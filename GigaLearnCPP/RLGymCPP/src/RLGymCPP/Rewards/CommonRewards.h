#pragma once
#include "Reward.h"
#include "../Math.h"

namespace RLGC {

	template<bool PlayerEventState::* VAR, bool NEGATIVE>
	class PlayerDataEventReward : public Reward {
		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) {
			bool val =  player.eventState.*VAR;

			if (NEGATIVE) {
				return -(float)val;
			} else {
				return (float)val;
			}
		}
	};

	typedef PlayerDataEventReward<&PlayerEventState::goal, false> PlayerGoalReward; // NOTE: Given only to the player who last touched the ball on the opposing team
	typedef PlayerDataEventReward<&PlayerEventState::assist, false> AssistReward;
	typedef PlayerDataEventReward<&PlayerEventState::shot, false> ShotReward;
	typedef PlayerDataEventReward<&PlayerEventState::shotPass, false> ShotPassReward;
	typedef PlayerDataEventReward<&PlayerEventState::save, false> SaveReward;
	typedef PlayerDataEventReward<&PlayerEventState::bump, false> BumpReward;
	typedef PlayerDataEventReward<&PlayerEventState::bumped, true> BumpedPenalty;
	typedef PlayerDataEventReward<&PlayerEventState::demo, false> DemoReward;
	typedef PlayerDataEventReward<&PlayerEventState::demoed, true> DemoedPenalty;

	// Rewards a goal by anyone on the team
	// NOTE: Already zero-sum
	class GoalReward : public Reward {
	public:
		float concedeScale;
		GoalReward(float concedeScale = -1) : concedeScale(concedeScale) {}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) {
			if (!state.goalScored)
				return 0;

			bool scored = (player.team != RS_TEAM_FROM_Y(state.ball.pos.y));
			return scored ? 1 : concedeScale;
		}
	};

	// https://github.com/AechPro/rocket-league-gym-sim/blob/main/rlgym_sim/utils/reward_functions/common_rewards/misc_rewards.py
	class VelocityReward : public Reward {
	public:
		bool isNegative;
		VelocityReward(bool isNegative = false) : isNegative(isNegative) {}
		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) {
			return player.vel.Length() / CommonValues::CAR_MAX_SPEED * (1 - 2 * isNegative);
		}
	};

	// https://github.com/AechPro/rocket-league-gym-sim/blob/main/rlgym_sim/utils/reward_functions/common_rewards/ball_goal_rewards.py
	class VelocityBallToGoalReward : public Reward {
	public:
		bool ownGoal = false;
		VelocityBallToGoalReward(bool ownGoal = false) : ownGoal(ownGoal) {}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) {
			bool targetOrangeGoal = player.team == Team::BLUE;
			if (ownGoal)
				targetOrangeGoal = !targetOrangeGoal;

			Vec targetPos = targetOrangeGoal ? CommonValues::ORANGE_GOAL_BACK : CommonValues::BLUE_GOAL_BACK;
			
			Vec ballDirToGoal = (targetPos - state.ball.pos).Normalized();
			return ballDirToGoal.Dot(state.ball.vel / CommonValues::BALL_MAX_SPEED);
		}
	};

	// https://github.com/AechPro/rocket-league-gym-sim/blob/main/rlgym_sim/utils/reward_functions/common_rewards/player_ball_rewards.py
	class VelocityPlayerToBallReward : public Reward {
	public:
		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) {
			Vec dirToBall = (state.ball.pos - player.pos).Normalized();
			Vec normVel = player.vel / CommonValues::CAR_MAX_SPEED;
			return dirToBall.Dot(normVel);
		}
	};

	// True-potential form of player->ball proximity (Nexto's liu_dist player quality):
	// r = gamma*Phi(s') - Phi(s), Phi = exp(-dist/LIU_DIST_SCALE). Exact PBRS, so movement
	// cycles and whack-and-chase loops telescope to ~0 and episode truncation can't be
	// harvested — unlike VelocityPlayerToBallReward, the Phi drop when the ball is knocked
	// away is charged immediately. gamma must match the learner's gaeGamma.
	//
	// TEAM-CLOSEST (4.0 team play): Phi is the CLOSEST alive team car's proximity, not the
	// player's own — "someone on our team is on the ball", the team analogue of the same
	// race. With one car per team this is algebraically identical to the old per-player
	// form (verified by test); in 2v2/3v3 it stops paying the 2nd/3rd man for crowding the
	// ball (per-player proximity paid every teammate for converging on it — the single
	// biggest ball-chasing gradient in the stack). Every teammate receives the same value,
	// so under ZeroSumReward the result is exactly Psi = Phi_ownTeam - Phi_oppTeam at any
	// teamSpirit. Still an exact potential (a function of state only): telescoping holds.
	class BallProximityPotentialReward : public Reward {
	public:
		constexpr static float LIU_DIST_SCALE = 1410; // Nexto: max driving speed without boost
		float gamma;
		BallProximityPotentialReward(float gamma = 0.99f) : gamma(gamma) {}

		// Closest ALIVE team car's proximity; 0 if the whole team is demoed (no car = no coverage)
		static float TeamPhi(const GameState& state, Team team) {
			float best = 0;
			for (const Player& p : state.players)
				if (p.team == team && !p.isDemoed)
					best = RS_MAX(best, expf(-(state.ball.pos - p.pos).Length() / LIU_DIST_SCALE));
			return best;
		}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (!state.prev)
				return 0; // First step of the episode: nothing to diff against

			// A demo/respawn teleport is not anyone's action; skip the step whenever a team
			// car's demo state flips (the team analogue of the old self-only guard — with one
			// car per team the returned values are identical: transitions return 0 here, and
			// steady-demoed steps diff an empty-team Phi of 0 against 0).
			for (const Player& p : state.players) {
				if (p.team != player.team)
					continue;
				if (!p.prev || p.isDemoed != p.prev->isDemoed)
					return 0;
			}

			return gamma * TeamPhi(state, player.team) - TeamPhi(*state.prev, player.team);
		}
	};

	// True-potential form of ball->goal progress (Nexto's goal-dist state quality):
	// Phi = 0.5*(exp(-dOpp/CAR_MAX_SPEED) - exp(-dOwn/CAR_MAX_SPEED)), r = gamma*Phi(s') - Phi(s).
	// Blue's Phi is the exact negative of orange's, so this is already zero-sum — do NOT
	// wrap it in ZeroSumReward (that silently doubles it). The exp shape is near-flat at
	// midfield and steep at the goal mouth, so credit concentrates on finishing positions
	// and rolled-back balls refund in full. gamma must match the learner's gaeGamma.
	class BallToGoalPotentialReward : public Reward {
	public:
		float gamma;
		BallToGoalPotentialReward(float gamma = 0.99f) : gamma(gamma) {}

		static float Phi(const Vec& ballPos, Team team) {
			Vec oppGoal = (team == Team::BLUE) ? CommonValues::ORANGE_GOAL_BACK : CommonValues::BLUE_GOAL_BACK;
			Vec ownGoal = (team == Team::BLUE) ? CommonValues::BLUE_GOAL_BACK : CommonValues::ORANGE_GOAL_BACK;
			return 0.5f * (
				expf(-(ballPos - oppGoal).Length() / CommonValues::CAR_MAX_SPEED) -
				expf(-(ballPos - ownGoal).Length() / CommonValues::CAR_MAX_SPEED));
		}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (!state.prev)
				return 0;

			return gamma * Phi(state.ball.pos, player.team) - Phi(state.prev->ball.pos, player.team);
		}
	};

	// https://github.com/AechPro/rocket-league-gym-sim/blob/main/rlgym_sim/utils/reward_functions/common_rewards/player_ball_rewards.py
	class FaceBallReward : public Reward {
	public:
		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) {
			Vec dirToBall = (state.ball.pos - player.pos).Normalized();
			return player.rotMat.forward.Dot(dirToBall);
		}
	};

	class TouchBallReward : public Reward {
	public:
		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) {
			return player.ballTouchedStep;
		}
	};

	class SpeedReward : public Reward {
	public:
		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) {
			return player.vel.Length() / CommonValues::CAR_MAX_SPEED;
		}
	};

	class WavedashReward : public Reward {
	public:
		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) {
			if (!player.prev)
				return 0;

			if (player.isOnGround && (player.prev->isFlipping && !player.prev->isOnGround)) {
				return 1;
			} else {
				return 0;
			}
		}
	};

	class PickupBoostReward : public Reward {
	public:
		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (!player.prev)
				return 0;

			if (player.boost > player.prev->boost) {
				return sqrtf(player.boost / 100.f) - sqrtf(player.prev->boost / 100.f);
			} else {
				return 0;
			}
		}
	};

	// https://github.com/AechPro/rocket-league-gym-sim/blob/main/rlgym_sim/utils/reward_functions/common_rewards/misc_rewards.py
	class SaveBoostReward : public Reward {
	public:
		float exponent;
		SaveBoostReward(float exponent = 0.5f) : exponent(exponent) {}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) {
			return RS_CLAMP(powf(player.boost / 100, exponent), 0, 1);
		}
	};


	class AirReward : public Reward {
	public:
		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			return !player.isOnGround;
		}
	};

	// Mostly based on the classic Necto rewards
	// Total reward output for speeding the ball up to MAX_REWARDED_BALL_SPEED is 1.0
	// The bot can do this slowly (putting) or quickly (shooting)
	class TouchAccelReward : public Reward {
	public:
		constexpr static float MAX_REWARDED_BALL_SPEED = RLGC::Math::KPHToVel(110);

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (!state.prev)
				return 0;

			if (player.ballTouchedStep) {
				float prevSpeedFrac = RS_MIN(1, state.prev->ball.vel.Length() / MAX_REWARDED_BALL_SPEED);
				float curSpeedFrac = RS_MIN(1, state.ball.vel.Length() / MAX_REWARDED_BALL_SPEED);

				if (curSpeedFrac > prevSpeedFrac) {
					return (curSpeedFrac - prevSpeedFrac);
				} else {
					// Not speeding up the ball so we don't care
					return 0;
				}
			} else {
				return 0;
			}
		}
	};

	// Touch height (Nexto's touch_height), impulse-scaled: on touch, pays
	// (2 - onGround) * avg(ballZ, carZ)/CEILING_Z * min(1, |delta ballVel|/FULL_CREDIT_DELTA_V).
	// ballTouchedStep fires EVERY step of sustained contact, so without the impulse factor
	// this would be a carry annuity (wall pins / ceiling carries out-earn goals). A carry
	// only adds ~67 uu/s of ball speed per step (impulse factor ~0.07 -> pays ~0); a real
	// strike gets the full height credit. Airborne touches pay double.
	class TouchHeightReward : public Reward {
	public:
		constexpr static float FULL_CREDIT_DELTA_V = 1000; // uu/s of ball delta-v for full credit

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (!state.prev || !player.ballTouchedStep)
				return 0;

			float heightFrac = 0.5f * (state.ball.pos.z + player.pos.z) / CommonValues::CEILING_Z;
			float airMult = player.isOnGround ? 1.0f : 2.0f;
			float impulseFrac = RS_MIN(1.0f, (state.ball.vel - state.prev->ball.vel).Length() / FULL_CREDIT_DELTA_V);
			return airMult * heightFrac * impulseFrac;
		}
	};

	class StrongTouchReward : public Reward {
	public:
		float minRewardedVel, maxRewardedVel;
		StrongTouchReward(float minSpeedKPH = 20, float maxSpeedKPH = 130) {
			minRewardedVel = RLGC::Math::KPHToVel(minSpeedKPH);
			maxRewardedVel = RLGC::Math::KPHToVel(maxSpeedKPH);
		}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (!state.prev)
				return 0;

			if (player.ballTouchedStep) {
				float hitForce = (state.ball.vel - state.prev->ball.vel).Length();
				if (hitForce < minRewardedVel)
					return 0;

				return RS_MIN(1, hitForce / maxRewardedVel);
			} else {
				return 0;
			}
		}
	};

	// ========================================================================
	// FRONTIER-9 additions (workflow-designed, red-team verified against the
	// actual GameEventTracker/Arena source before shipping).
	// ========================================================================

	// Aerial STRIKE reward - replaces TouchHeightReward as the aerial gradient.
	// Why TouchHeight failed (Aerial Touch Ratio stuck at 0.00125 after 17B+ steps):
	//   (a) avg(ballZ, carZ) paid ground strikes and hop-pokes - no DIFFERENTIAL
	//       between the current 193uu-touch behavior and the target skill;
	//   (b) it was gated, and the HER-trained gate structurally discounts states
	//       the policy has never achieved; positive-part-only gating also made
	//       the zero-sum pair net-negative on every aerial touch.
	// This version is an AND (min, not avg): full credit requires SUSTAINED
	// FLIGHT and a GENUINELY HIGH BALL simultaneously. isOnGround is true on
	// walls, so wall-pinned touches pay exactly 0; the ball-height ramp starts
	// at 150uu (rest ball = 93, current mean touch = 193 -> pays ~0). The
	// impulse factor kills carry annuities; the refire cooldown caps juggle
	// self-rally at ~1.25 payouts/s. Wrap in ZeroSumReward(_, 0). Do NOT gate.
	class AerialTouchReward : public Reward {
	public:
		constexpr static float BALL_MIN_Z = 150;            // below this pays 0
		constexpr static float BALL_FULL_Z = 1450;           // full height credit at/above
		constexpr static float MAX_CREDIT_AIR_TIME = 1.75f;  // seconds of flight for full air credit
		constexpr static float FULL_CREDIT_DELTA_V = 500;    // uu/s of ball delta-v for full credit
		constexpr static int REFIRE_COOLDOWN_STEPS = 24;     // ~0.8s at tickSkip 4 (30Hz steps); re-derive if tickSkip changes

		std::vector<int> stepsSincePay; // per player.index; per-arena instance, so safe

		virtual void Reset(const GameState& initialState) override {
			stepsSincePay.assign(initialState.players.size(), REFIRE_COOLDOWN_STEPS + 1);
		}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if ((size_t)player.index >= stepsSincePay.size())
				stepsSincePay.resize(player.index + 1, REFIRE_COOLDOWN_STEPS + 1);
			int& sincePay = stepsSincePay[player.index];
			sincePay++;

			if (!state.prev || !player.ballTouchedStep || player.isOnGround)
				return 0;
			if (sincePay <= REFIRE_COOLDOWN_STEPS)
				return 0;

			float ballFrac = RS_CLAMP((state.ball.pos.z - BALL_MIN_Z) / (BALL_FULL_Z - BALL_MIN_Z), 0, 1);
			float airFrac = RS_MIN(1.0f, player.airTime / MAX_CREDIT_AIR_TIME);
			float heightFrac = RS_MIN(ballFrac, airFrac);
			if (heightFrac <= 0)
				return 0;

			float impulseFrac = RS_MIN(1.0f, (state.ball.vel - state.prev->ball.vel).Length() / FULL_CREDIT_DELTA_V);
			float r = heightFrac * impulseFrac;
			if (r > 0)
				sincePay = 0;
			return r;
		}
	};

	// Pre-touch aerial-approach potential - the gradient BEFORE the first air
	// touch ever lands. Phi = ballHighFrac * carUpFrac * exp(-dist/LIU_DIST_SCALE);
	// r = gamma*Phi(s') - Phi(s). Phi is 0 whenever the car is grounded or the
	// ball is low, so parking under a dropping ball earns nothing, while
	// jumping/boosting toward a high ball pays +dPhi immediately and a whiff or
	// landing refunds it - attempts become gradient-visible at ~zero net cost.
	// Exact PBRS: telescopes to ~0 over any cycle, cannot be farmed. NEVER gate.
	class AirInterceptPotentialReward : public Reward {
	public:
		constexpr static float LIU_DIST_SCALE = 1410;
		constexpr static float BALL_MIN_Z = 300, BALL_FULL_Z = 1600;
		constexpr static float CAR_MIN_Z = 17, CAR_FULL_Z = 517;
		float gamma;
		AirInterceptPotentialReward(float gamma = 0.99f) : gamma(gamma) {}

		static float Phi(const Vec& ballPos, const Vec& carPos) {
			float ballFrac = RS_CLAMP((ballPos.z - BALL_MIN_Z) / (BALL_FULL_Z - BALL_MIN_Z), 0, 1);
			float carFrac = RS_CLAMP((carPos.z - CAR_MIN_Z) / (CAR_FULL_Z - CAR_MIN_Z), 0, 1);
			return ballFrac * carFrac * expf(-(ballPos - carPos).Length() / LIU_DIST_SCALE);
		}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (!state.prev || !player.prev)
				return 0; // First step of the episode: nothing to diff against

			// A demo/respawn teleport is not the player's action; don't charge/pay the jump
			if (player.isDemoed || player.prev->isDemoed)
				return 0;

			return gamma * Phi(state.ball.pos, player.pos) - Phi(state.prev->ball.pos, player.prev->pos);
		}
	};

	// PickupBoostReward + the demo guard the original is missing. Demo-respawn
	// boost (a fresh spawn tank) is not a pickup the player earned: without this
	// guard, a ZeroSumReward wrapper CHARGES the demoer for the victim's respawn
	// tank, silently refunding part of the demo pair swing. Guard mirrors
	// BallProximityPotentialReward's demo/respawn-teleport convention.
	class GuardedPickupBoostReward : public Reward {
	public:
		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (!player.prev)
				return 0;

			// Respawn refill is a teleport side effect, not a pickup
			if (player.isDemoed || player.prev->isDemoed)
				return 0;

			if (player.boost > player.prev->boost) {
				return sqrtf(player.boost / 100.f) - sqrtf(player.prev->boost / 100.f);
			} else {
				return 0;
			}
		}
	};

	// Save event with an OPPOSED-SHOT guard.
	//
	// The raw engine save is farmable in 1v1 (verified against GameEventTracker.cpp):
	// GameEventTracker credits the shot to the team OPPOSITE the threatened net
	// (shooterTeam = RS_OPPOSITE_TEAM(goalTeam)), so a player who blasts the ball
	// at their OWN net some time after any opponent touch arms a phantom "shot"
	// credited to the opponent, then collects a "save" by touching it away again -
	// a repeatable zero-sum income stream decoupled from winning.
	//
	// The guard: only pay the save if the last ball touch STRICTLY BEFORE the
	// save step came from the OPPONENT. Genuine save (opponent shoots -> I block)
	// passes; self-served shots (my own touch armed the trajectory) pay 0, because
	// curTouchTeam persists as MY team from my arming touch onward until someone
	// else touches it - so prevTouchTeam is mine, not the opponent's, at the
	// moment I'd collect a self-farmed save. Conservative by design: a double-
	// touch save (my first contact fails to clear, second completes it) is also
	// blocked - under-crediting, never over-crediting.
	//
	// Stateful: last-touch tracking updates in PreStep (RewardWrapper forwards
	// PreStep/Reset to the child, so this works inside ZeroSumReward). Per-arena
	// instance -> no cross-arena state. Wrap in ZeroSumReward(_, 0). Do NOT gate:
	// the reachability gate's attack-oriented level is lowest exactly in the
	// own-half states where saves fire.
	class OpposedSaveReward : public Reward {
	public:
		// -1 = none yet, -2 = ambiguous (both teams touched the same step)
		int curTouchTeam = -1;  // team of the most recent touch, up to and including this step
		int prevTouchTeam = -1; // team of the last touch strictly before this step

		virtual void Reset(const GameState& initialState) override {
			curTouchTeam = prevTouchTeam = -1;
		}

		virtual void PreStep(const GameState& state) override {
			prevTouchTeam = curTouchTeam;

			int touchTeam = -1;
			bool ambiguous = false;
			for (const Player& p : state.players) {
				if (!p.ballTouchedStep)
					continue;
				int t = (int)p.team;
				if (touchTeam == -1)
					touchTeam = t;
				else if (touchTeam != t)
					ambiguous = true;
			}

			if (ambiguous)
				curTouchTeam = -2;
			else if (touchTeam != -1)
				curTouchTeam = touchTeam;
		}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (!player.eventState.save)
				return 0;

			// Pay only if the touch that created/carried the shot was the opponent's
			return (prevTouchTeam == (int)RS_OPPOSITE_TEAM(player.team)) ? 1.0f : 0.0f;
		}
	};
}