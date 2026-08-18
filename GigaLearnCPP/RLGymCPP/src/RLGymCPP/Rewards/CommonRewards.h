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

		// Closest ALIVE team car's proximity; 0 if the whole team is demoed (no car = no coverage).
		// 5.0 FAR-FIELD TERM (PULSAR5.md): the pure exp saturates beyond ~2000uu - the
		// measured "gradient desert" behind ~9% dead far-field frames (behavior_symptoms).
		// A small linear component gives the potential a nonzero slope everywhere; still
		// an exact potential (any Phi is), so still unfarmable and policy-invariant.
		static float TeamPhi(const GameState& state, Team team) {
			float best = 0;
			for (const Player& p : state.players)
				if (p.team == team && !p.isDemoed) {
					float d = (state.ball.pos - p.pos).Length();
					float phi = expf(-d / LIU_DIST_SCALE) + 0.08f * RS_MAX(0.f, 1.f - d / 12000.f);
					best = RS_MAX(best, phi);
				}
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

		/** Computes each team's potential and demo guard once for the full player batch. **/
		virtual std::vector<float> GetAllRewards(const GameState& state, bool isFinal) override {
			std::vector<float> result(state.players.size(), 0.f);
			if (!state.prev)
				return result;

			bool blueValid = true;
			bool orangeValid = true;
			for (const Player& p : state.players) {
				bool& teamValid = p.team == Team::BLUE ? blueValid : orangeValid;
				if (!p.prev || p.isDemoed != p.prev->isDemoed)
					teamValid = false;
			}

			const float blueReward = blueValid
				? gamma * TeamPhi(state, Team::BLUE) - TeamPhi(*state.prev, Team::BLUE)
				: 0.f;
			const float orangeReward = orangeValid
				? gamma * TeamPhi(state, Team::ORANGE) - TeamPhi(*state.prev, Team::ORANGE)
				: 0.f;
			for (size_t i = 0; i < state.players.size(); i++)
				result[i] = state.players[i].team == Team::BLUE ? blueReward : orangeReward;
			return result;
		}
	};

	// TIME COST (2026-07-16, user-directed): a small constant per-step penalty -
	// urgency pressure that taxes stalling (the measured 73%-airborne hover next to
	// uncontested balls) and slow play generally. NOT zero-sum in the strict sum
	// (both players pay), but a uniform constant CANCELS in every competitive margin
	// (PSD/league fitness = return DIFFERENCES), so fitness purity holds. The weight
	// must stay small vs GoalReward: at weight w a 30s episode costs 450w
	// (15Hz steps at tickSkip 8; re-derive if tickSkip changes) undiscounted -
	// keep 450w << 150 or ending episodes (including CONCEDING) becomes attractive.
	class TimeCostReward : public Reward {
	public:
		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			return -1.f;
		}
	};

	// TEMPO CREDIT (2026-07-16, user-directed "energy reward"): the player's total
	// MECHANICAL energy as an exact PBRS potential — Phi ~ (0.5|v|^2 + g*z), r =
	// gamma*Phi(s') - Phi(s). The farmability the raw form invites is removed by
	// construction, not by tuning: telescoping makes every energy-pump loop worth
	// exactly zero (speed up / flip about / slow down nets 0), the ZeroSum wrapper
	// kills co-farming, and gamma MUST match the learner's gaeGamma (stack rule).
	// What it buys is INSTANT, LOCAL credit for momentum decisions: a flip that dies
	// in place prices negative within a step instead of diffusing across the horizon
	// (the measured wavedash-class SNR starvation, MECHANICS.md). PE is included so
	// KE->PE conversion (jumping, climbing) is energy-neutral — this term must never
	// tax aerials. Ball energy is deliberately absent: TouchAccel IS the ball-energy
	// reward. STORED BOOST is in the sum since 2026-07-19 (see Phi): fuel is energy,
	// so the potential is indifferent between holding and spending it — the dump-
	// into-ground-speed incentive the boost-free form created is gone.
	class CarEnergyPotentialReward : public Reward {
	public:
		float gamma;
		CarEnergyPotentialReward(float gamma = 0.99f) : gamma(gamma) {}

		static float Phi(const Player& p) {
			constexpr float KE_NORM = 0.5f * 2300.f * 2300.f; // KE at supersonic ~ 1.0
			float ke = 0.5f * p.vel.LengthSq();
			float pe = 650.f * RS_MAX(0.f, p.pos.z - 17.f);   // g = 650 uu/s^2
			// STORED BOOST IS ENERGY (2026-07-19, measured fix): without this term the
			// potential valued fuel at zero, so dumping boost into ground speed was free
			// energy gain and holding it was worthless - and the 29.3B aerial census
			// caught the consequence: mean boost at aerial opportunities collapsed
			// 54 -> 18 while takeoff ATTEMPTS rose 8x (the bot wants to go up, arrives
			// empty). SQRT-shaped (RLGym-PPO guide's scarcity curve, adopted same day):
			// the marginal value of fuel is highest near empty - exactly where the
			// census caught the failure - and cheap near full, so the term defends a
			// working reserve without paying for hoarding, and small pads become
			// automatically worth detouring for when low. Full tank = 1.0 = supersonic
			// KE. Still exact PBRS: ANY Phi telescopes - unfarmable; pad pickups are
			// real player actions (the demo-respawn tank stays guarded in GetReward).
			// Pickup credit intentionally overlaps GuardedPickupBoost - this term adds
			// the SPEND side of the ledger, which a pickup reward cannot express.
			float be = sqrtf(0.01f * p.boost) * KE_NORM;      // boost 0-100 -> 0..KE_NORM, concave
			return (ke + pe + be) / KE_NORM;
		}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (!state.prev || !player.prev)
				return 0;
			// Demo/respawn teleports are not the player's action
			if (player.isDemoed || player.isDemoed != player.prev->isDemoed)
				return 0;
			return gamma * Phi(player) - Phi(*player.prev);
		}
	};

	// TEAM PRESSURE (2026-07-19, user-directed, RLGym-PPO-guide item): -1 per step
	// while NO alive teammate (self included) is pressuring the ball - "pressuring"
	// = within nearDist of it, OR closing on it at closeSpeed+ from within closeDist.
	// nearDist is deliberately wide (2500) so shadow defense COUNTS as pressure: the
	// penalty fires only on genuine collective disengagement (everyone far, nobody
	// approaching) - the measured dead-play / collective-decline pathology - never on
	// defensive shape. Wrap in ZeroSumReward like every event term; mutual
	// non-pressure then cancels (a zero-sum stack cannot charge symmetric passivity -
	// accepted, TimeCost still taxes both sides; real states are asymmetric).
	class TeamPressureReward : public Reward {
	public:
		float nearDist, closeDist, closeSpeed;
		TeamPressureReward(float nearDist = 2500, float closeDist = 4000, float closeSpeed = 500)
			: nearDist(nearDist), closeDist(closeDist), closeSpeed(closeSpeed) {}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			for (auto& p : state.players) {
				if (p.team != player.team || p.isDemoed)
					continue;
				Vec to = state.ball.pos - p.pos;
				float d = to.Length();
				if (d < nearDist)
					return 0;
				if (d < closeDist && p.vel.Dot(to * (1.f / RS_MAX(d, 1.f))) > closeSpeed)
					return 0;
			}
			return -1;
		}

		/** Evaluates pressure once per car and reuses the team result for every teammate. **/
		virtual std::vector<float> GetAllRewards(const GameState& state, bool isFinal) override {
			bool bluePressuring = false;
			bool orangePressuring = false;
			for (const Player& p : state.players) {
				if (p.isDemoed)
					continue;
				Vec to = state.ball.pos - p.pos;
				float d = to.Length();
				bool pressuring = d < nearDist
					|| (d < closeDist && p.vel.Dot(to * (1.f / RS_MAX(d, 1.f))) > closeSpeed);
				if (p.team == Team::BLUE)
					bluePressuring |= pressuring;
				else
					orangePressuring |= pressuring;
			}

			std::vector<float> result(state.players.size());
			for (size_t i = 0; i < state.players.size(); i++) {
				bool pressuring = state.players[i].team == Team::BLUE
					? bluePressuring : orangePressuring;
				result[i] = pressuring ? 0.f : -1.f;
			}
			return result;
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
	// self-rally at 0.2 payouts/s (1 / the 5s cooldown - genuinely tickSkip-invariant
	// since 2026-08-17, when it became a deltaTime accumulation instead of a step count
	// that claimed invariance it did not have; see the cooldown constant below).
	// Wrap in ZeroSumReward(_, 0). Do NOT gate.
	// KICKOFF RACE (2026-07-20, user-directed: Pulsar wins most play but loses NET on
	// conceded kickoff goals - it loses the kickoff). A kickoff is a symmetric sprint
	// to the ball; on a CONTESTED kickoff, getting the first touch fast is the game.
	// Rewards that first touch, TIME-DECAYED (fast ~1, dawdled ~0), zero-sum wrapped
	// so the winner gains and the loser mirrors it.
	//
	// THE CONTESTED GATE (the user's real safeguard): a "delay kickoff" is the
	// OPPONENT declining the 50/50 to bait us into committing so they can counter.
	// An UNCONDITIONAL first-touch reward would farm us there - the bot would rush in
	// for the touch reward and hand them the counter. So the touch pays ONLY when an
	// opponent is actually contesting the ball (near it or closing on it) at the
	// moment of contact. Against a hang-back, taking the touch pays 0, so the reward
	// never drives us into the trap; the real objective (Goal/B2G/possession) governs
	// how we play a delay instead. Detection: at the touch, is any opponent within
	// CONTEST_DIST of the ball, or closing on it above CONTEST_SPEED. Symmetric - it
	// pays whichever side wins a genuine contest and mirrors the loss.
	//
	// Kickoff is detected ONLY at Reset (ball spawned at field center at rest;
	// KickoffState / FuzzedKickoffState both leave the ball at (0,0,rest)), so a ball
	// passing through center mid-play can never false-fire it. Fires once per kickoff.
	// WINDOW RE-DERIVATION (2026-08-17, measurement-convicted). This reward had NEVER
	// FIRED in any lineage: a wandb sweep of `Rewards/KickoffRace` across every run
	// since it shipped 2026-07-20 found exactly 2 nonzero samples ever (max 0.0011).
	// Cause: the window was `WINDOW_STEPS = 30` with payout `1 - steps/30`, i.e. decaying
	// to ZERO at 2.0s - but a corner-spawn kickoff car is ~3280uu from a resting ball, a
	// pro-grade speedflip touch is ~1.9-2.1s (paying ~3%), and this bot's healthy median
	// first touch is ~3.4s (the boot probe's own number). The window expired and
	// `resolved` latched before any physically realistic touch, so the 5.3 weight raise
	// 25 -> 60 doubled a term that could not pay.
	// Now: FULL credit out to FULL_CREDIT_SECS (a near-optimal kickoff is not penalised
	// for being 1.9s instead of 0s - the decay's job is to punish dawdling, not to demand
	// superhuman speed), then a linear ramp to zero at WINDOW_SECS, which sits above the
	// measured median so a normal contested kickoff lands on the ramp with real gradient.
	// SECONDS, NOT STEPS: the old constant carried "re-derive if tickSkip changes" and
	// was then not re-derived - on the ts1 lineages (6.0-6.2) the same 30 steps was a
	// 0.25s window. Accumulating state.deltaTime makes it tickSkip-invariant by
	// construction, so the landmine cannot be re-armed. Revert = FULL_CREDIT_SECS 0,
	// WINDOW_SECS 2 (the old shape).
	class KickoffRaceReward : public Reward {
	public:
		constexpr static float FULL_CREDIT_SECS = 2.0f;  // at/below this, a contested win pays in full
		constexpr static float WINDOW_SECS = 5.0f;       // ...ramping to 0 here; median touch ~3.4s
		constexpr static float CONTEST_DIST = 1500;  // opponent within this of the ball = contesting
		constexpr static float CONTEST_SPEED = 500;  // ...or closing on it faster than this
		bool active = false, resolved = false;
		float elapsed = 0;

		static bool IsKickoffBall(const GameState& s) {
			return fabsf(s.ball.pos.x) < 200 && fabsf(s.ball.pos.y) < 200
				&& s.ball.vel.Length() < 50;
		}

		// Is an opponent of the toucher actually going for the ball? (Not a delay.)
		bool Contested(const Player& toucher, const GameState& s) const {
			for (auto& p : s.players) {
				if (p.team == toucher.team || p.isDemoed)
					continue;
				Vec to = s.ball.pos - p.pos;
				float d = to.Length();
				if (d < CONTEST_DIST)
					return true;
				if (d > 1e-3f && p.vel.Dot(to * (1.f / d)) > CONTEST_SPEED)
					return true;
			}
			return false;
		}

		virtual void Reset(const GameState& initialState) override {
			active = IsKickoffBall(initialState);
			resolved = false;
			elapsed = 0;
		}

		// EnvSet calls PreStep once per arena per step, before the reward pass - so this
		// replaces the old `if (player.index == 0) steps++` hack (which bumped inside the
		// per-player loop and depended on index 0 existing).
		virtual void PreStep(const GameState& state) override {
			if (active && !resolved)
				elapsed += state.deltaTime;
		}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (!active || resolved)
				return 0;
			if (elapsed > WINDOW_SECS) { // window elapsed unresolved -> no race reward
				resolved = true;
				return 0;
			}
			if (!player.ballTouchedStep)
				return 0;
			resolved = true; // first touch of the kickoff, either team
			if (!Contested(player, state))
				return 0; // uncontested (delay kickoff) -> no drive to commit into a counter
			float over = elapsed - FULL_CREDIT_SECS;
			if (over <= 0)
				return 1.f;
			return RS_CLAMP(1.f - over / (WINDOW_SECS - FULL_CREDIT_SECS), 0.f, 1.f);
		}

		virtual std::string GetName() override { return "KickoffRace"; }
	};

	class AerialTouchReward : public Reward {
	public:
		constexpr static float BALL_MIN_Z = 150;            // below this pays 0
		constexpr static float BALL_FULL_Z = 1450;           // full height credit at/above
		constexpr static float MAX_CREDIT_AIR_TIME = 1.75f;  // seconds of flight for full air credit
		constexpr static float FULL_CREDIT_DELTA_V = 500;    // uu/s of ball delta-v for full credit

		// REFIRE COOLDOWN 0.8s -> 5s (2026-08-17). The cooldown, not the weight, is the
		// term's anti-farm mechanism, and at 0.8s it was not rate-limiting anything: an
		// air-dribble juggle re-touches the ball far faster than that, so the farm ran at
		// the full 1.25 payouts/s ceiling. That farm was then measured TWICE - at weight
		// 120 (wandb pkljg9g1: raw income ~5x from 22B while windowed Nexto goal share
		// collapsed 0.875 -> 0.32) and again at 40 on the AiMOS 690-GPU run - and both
		// times it was answered by cutting the WEIGHT (120 -> 40 -> 15), which suppresses
		// the farm and the acquisition signal in equal measure. Lengthening the cooldown
		// is the asymmetric fix instead: a *legitimate* aerial play (leave the ground,
		// climb, strike, recover) has a natural cadence already well above 5s, so it loses
		// almost nothing, while a sustained juggle's income ceiling drops ~6x. This is
		// what buys back the headroom to run a meaningful weight without re-arming the
		// annuity - see the WEIGHT RULE note at the term's site in ExampleMain, which is
		// deliberately left at 15 here (raising it is a separate, user-owned lever).
		// SECONDS, NOT STEPS: the old `REFIRE_COOLDOWN_STEPS = 12` carried "re-derive if
		// tickSkip changes" and was not re-derived across the ts1 lineages, where the same
		// 12 steps meant 0.1s. Accumulating deltaTime is tickSkip-invariant by
		// construction. Revert = REFIRE_COOLDOWN_SECS 0.8f.
		constexpr static float REFIRE_COOLDOWN_SECS = 5.0f;

		std::vector<float> secsSincePay; // per player.index; per-arena instance, so safe

		virtual void Reset(const GameState& initialState) override {
			secsSincePay.assign(initialState.players.size(), REFIRE_COOLDOWN_SECS + 1);
		}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if ((size_t)player.index >= secsSincePay.size())
				secsSincePay.resize(player.index + 1, REFIRE_COOLDOWN_SECS + 1);
			float& sincePay = secsSincePay[player.index];
			sincePay += state.deltaTime;

			if (!state.prev || !player.ballTouchedStep || player.isOnGround)
				return 0;
			if (sincePay <= REFIRE_COOLDOWN_SECS)
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

	// FLIP RESET (2026-07-20, user-directed: a ZERO-RATE mechanic - the bot never
	// enters the precursor state, so amplification can't help and it needs both
	// SEEDING (AirPlayState) and this gradient). A flip reset = restoring your flip
	// by contacting the ball with your WHEELS while airborne. There is no honest PBRS
	// potential for it (its value is purely instrumental - the flick AFTER the
	// reset), so it is a gated EVENT reward, farm-hardened four ways: (1) a GENUINE
	// reset only - the engine's own HasFlipReset() must transition false->true via a
	// ball touch this step; (2) roof pointing at the ball (rotMat.up . toBall > 0.3),
	// i.e. wheels-first contact, not a nose poke; (3) ball genuinely high; (4) ~1s
	// cooldown so a tight juggle can't rack it up. Zero-sum wrapped like every event
	// term. The residual farm (ceiling-ball juggling for resets) advances the ball
	// nowhere, so B2G/Goal/TimeCost dominate its gradient once scoring exists.
	// SCAFFOLD weight - anneal once mechanic_census shows a stable flip-reset rate.
	class FlipResetReward : public Reward {
	public:
		constexpr static float BALL_MIN_Z = 500;             // genuinely aerial
		constexpr static float BALL_FULL_Z = 1400;
		constexpr static float ROOF_TO_BALL_MIN = 0.3f;      // wheels-toward-ball geometry
		constexpr static int   COOLDOWN_STEPS = 15;          // ~1s at 15Hz (tickSkip 8)
		std::vector<int> sincePay; // per player.index; per-arena instance, safe

		virtual void Reset(const GameState& initialState) override {
			sincePay.assign(initialState.players.size(), COOLDOWN_STEPS + 1);
		}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if ((size_t)player.index >= sincePay.size())
				sincePay.resize(player.index + 1, COOLDOWN_STEPS + 1);
			int& sp = sincePay[player.index];
			sp++;

			if (!state.prev || !player.prev)
				return 0;
			// The reset EVENT: touched the ball, airborne, flip restored THIS step
			// (transition), not the whole airborne-reset window.
			if (!player.ballTouchedStep || player.isOnGround)
				return 0;
			if (!player.HasFlipReset() || player.prev->HasFlipReset())
				return 0;
			if (sp <= COOLDOWN_STEPS)
				return 0;

			Vec toBall = state.ball.pos - player.pos;
			float d = toBall.Length();
			if (d < 1e-3f)
				return 0;
			// Roof toward ball = wheels made the contact (the reset geometry)
			if (player.rotMat.up.Dot(toBall * (1.f / d)) < ROOF_TO_BALL_MIN)
				return 0;

			float ballFrac = RS_CLAMP((state.ball.pos.z - BALL_MIN_Z) / (BALL_FULL_Z - BALL_MIN_Z), 0, 1);
			if (ballFrac <= 0)
				return 0;
			sp = 0;
			return ballFrac;
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

	// CONSECUTIVE AIR TOUCHES (2026-07-21, user-directed): rewards CHAINING several
	// airborne ball touches without landing in between - sustained aerial control
	// (aerial dribbling / juggling), the skill AerialTouchReward's single-strike
	// event cannot express. Built as an exact-telescoping PBRS so it is UNFARMABLE
	// BY CONSTRUCTION (the user's requirement), not by tuning: Phi is a saturating
	// function of a per-player consecutive-air-touch STREAK, r = gamma*Phi(s') -
	// Phi(s). Extending a chain raises Phi (positive, LOCAL credit for the extra
	// touch); the moment the player lands, the streak resets to 0 and the whole
	// chain is refunded (gamma*Phi(0) - Phi(n) = -Phi(n)), so a tap-land-tap grind
	// nets ~0 - the discounted sum over ANY trajectory telescopes to -Phi(start).
	// gamma MUST equal the learner's gaeGamma (stack rule) or it stops telescoping
	// against GAE. Wrap in ZeroSumReward: a linear map of a per-player potential is
	// still a telescoping potential, so this keeps the whole-stack zero-sum
	// invariant while staying exact PBRS. NEVER gate (positive-part gating breaks
	// telescoping).
	//
	// Phi(streak) = 1 - exp(-max(0, streak-1)/TAU): streak 0 AND 1 both score 0, so
	// a lone airborne touch pays nothing here (that is AerialTouchReward's job) and
	// credit begins on the 2nd consecutive air touch, saturating toward 1 so an
	// endless juggle cannot pay unboundedly (marginal credit shrinks each touch).
	// The streak counts this player's airborne (isOnGround==false) ballTouchedStep
	// touches and resets to 0 whenever the player is on the ground OR a wall (any
	// isOnGround - "consecutive AIR touches" ends the instant you are supported).
	// A demo/respawn is a teleport, not the player's action: the streak resets and
	// the step is not charged (mirrors the other potentials' demo guard - a
	// negligible, standard telescoping leak). Per-arena instance -> no cross-arena
	// state.
	class ConsecutiveAirTouchReward : public Reward {
	public:
		constexpr static float TAU = 2.0f; // touches past the first to reach ~63% saturation
		float gamma;
		ConsecutiveAirTouchReward(float gamma = 0.99f) : gamma(gamma) {}

		std::vector<int> streak; // per player.index

		static float Phi(int streak) {
			int chained = streak - 1; // 0 and 1 -> no chain yet
			if (chained <= 0)
				return 0.f;
			return 1.f - expf(-(float)chained / TAU);
		}

		virtual void Reset(const GameState& initialState) override {
			streak.assign(initialState.players.size(), 0);
		}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if ((size_t)player.index >= streak.size())
				streak.resize(player.index + 1, 0);
			int& s = streak[player.index];
			int prevStreak = s;

			// Demo/respawn teleport is not the player's action: drop the chain, don't charge it
			if (player.isDemoed || (player.prev && player.prev->isDemoed)) {
				s = 0;
				return 0;
			}

			if (player.isOnGround) {
				s = 0; // supported (ground OR wall): the airborne chain ends
			} else if (player.ballTouchedStep) {
				s = prevStreak + 1; // another airborne touch extends the chain
			}
			// else: airborne, no touch -> streak unchanged (Phi bleeds by (gamma-1)*Phi)

			return gamma * Phi(s) - Phi(prevStreak);
		}
	};

	// WALL-JUMP-TO-BALL (2026-07-21, user-directed): rewards going for the ball in
	// the air after launching off a wall - the wall-read aerial (drive up a wall,
	// leave it, strike the ball). Built as an exact-telescoping PBRS so it is
	// UNFARMABLE BY CONSTRUCTION (the user's requirement): while the player is
	// airborne AND has left a wall since it was last supported, Phi = exp(-|ball -
	// car| / LIU_DIST_SCALE); otherwise Phi = 0. r = gamma*Phi(s') - Phi(s). Closing
	// on the ball after a wall launch pays +dPhi immediately; landing clears the
	// latch and refunds the approach in full (Phi -> 0 is charged as -Phi at the
	// landing step), so climb-and-retreat cycles telescope to ~0 - the discounted
	// sum over ANY trajectory is -Phi(start). Leaving a wall AWAY from the ball pays
	// ~0 (dist large -> Phi ~ 0), so it rewards wall exits TOWARD the ball only.
	// gamma MUST equal the learner's gaeGamma. Wrap in ZeroSumReward (a linear map
	// of a per-player potential stays a telescoping potential -> keeps the stack
	// zero-sum). NEVER gate.
	//
	// "Launched off a wall" = a per-player latch set the step the player becomes
	// airborne having been ON A WALL the previous step, held through the WHOLE
	// aerial (so the full wall-to-ball flight is shaped, not just the launch tick),
	// and cleared on any ground/wall contact or demo. It is intentionally NOT
	// cleared on the ball touch: clearing mid-air would drop Phi without charging
	// the refund and reopen a farm. A wall = world contact whose normal is closer
	// to horizontal than the engine's own ground/wall split (|normal.z| < 1/sqrt(2),
	// the autoflip NORM_Z_THRESH): flat ground (normal.z~1) and ceiling (normal.z~-1)
	// are excluded, side/back/corner walls (normal.z~0) included. Being a plain state
	// latch, Phi stays a function of (augmented) state, so telescoping holds.
	// Per-arena instance -> no cross-arena state.
	class WallJumpToBallReward : public Reward {
	public:
		constexpr static float LIU_DIST_SCALE = 1410;            // matches the other proximity potentials
		constexpr static float WALL_NORMAL_Z_MAX = 0.7071068f;   // 1/sqrt(2) = engine ground/wall split
		float gamma;
		WallJumpToBallReward(float gamma = 0.99f) : gamma(gamma) {}

		std::vector<char> launched; // per player.index (char = bool latch)

		static bool OnWall(const Player& p) {
			return p.isOnGround && p.worldContact.hasContact
				&& fabsf(p.worldContact.contactNormal.z) < WALL_NORMAL_Z_MAX;
		}

		virtual void Reset(const GameState& initialState) override {
			launched.assign(initialState.players.size(), 0);
		}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if ((size_t)player.index >= launched.size())
				launched.resize(player.index + 1, 0);
			char& latch = launched[player.index];
			bool prevLatch = latch != 0;

			// Need both previous states to diff a potential; a demo/respawn teleport is
			// not the player's action. In any of these cases, drop the latch and skip.
			if (!state.prev || !player.prev || player.isDemoed || player.prev->isDemoed) {
				latch = 0;
				return 0;
			}

			bool curLatch;
			if (player.isOnGround)
				curLatch = false;             // supported (ground OR wall): no active launch
			else if (OnWall(*player.prev))
				curLatch = true;              // just left a wall into the air
			else
				curLatch = prevLatch;         // persist through the aerial
			latch = curLatch ? 1 : 0;

			float phiPrev = prevLatch
				? expf(-(state.prev->ball.pos - player.prev->pos).Length() / LIU_DIST_SCALE) : 0.f;
			float phiCur = curLatch
				? expf(-(state.ball.pos - player.pos).Length() / LIU_DIST_SCALE) : 0.f;

			return gamma * phiCur - phiPrev;
		}
	};
}