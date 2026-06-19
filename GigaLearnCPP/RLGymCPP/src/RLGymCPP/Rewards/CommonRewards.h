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

	namespace ScoringRewardUtil {
		inline float SafeExpDist(Vec a, Vec b, float scale) {
			return expf(-a.Dist(b) / scale);
		}

		inline float CosineSimilarity(Vec a, Vec b) {
			float denom = a.Length() * b.Length();
			if (denom <= 1e-6f)
				return 0;

			return a.Dot(b) / denom;
		}

		inline float HeightActivation(float z) {
			return cbrtf((z - CommonValues::GOAL_HEIGHT) / CommonValues::CEILING_Z);
		}

		inline float BallGoalDistanceQuality(const GameState& state) {
			Vec blueGoal = (CommonValues::BLUE_GOAL_BACK + CommonValues::BLUE_GOAL_CENTER) / 2;
			Vec orangeGoal = (CommonValues::ORANGE_GOAL_BACK + CommonValues::ORANGE_GOAL_CENTER) / 2;

			return 0.25f * (
				SafeExpDist(orangeGoal, state.ball.pos, CommonValues::CAR_MAX_SPEED) -
				SafeExpDist(blueGoal, state.ball.pos, CommonValues::CAR_MAX_SPEED)
			);
		}

		inline Team GetScoringTeam(const GameState& state) {
			return RS_TEAM_FROM_Y(state.ball.pos.y) == Team::BLUE ? Team::ORANGE : Team::BLUE;
		}

		inline const Player* GetPrevPlayer(const GameState& state, const Player& player) {
			if (!state.prev)
				return NULL;

			for (const Player& prevPlayer : state.prev->players)
				if (prevPlayer.carId == player.carId)
					return &prevPlayer;

			return NULL;
		}
	}

	class BallGoalDistanceReward : public Reward {
	public:
		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (!state.prev)
				return 0;

			float delta =
				ScoringRewardUtil::BallGoalDistanceQuality(state) -
				ScoringRewardUtil::BallGoalDistanceQuality(*state.prev);
			return player.team == Team::BLUE ? delta : -delta;
		}
	};

	class TeamGoalReward : public Reward {
	public:
		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (!state.goalScored)
				return 0;

			return player.team == ScoringRewardUtil::GetScoringTeam(state) ? 1 : 0;
		}
	};

	class GoalSpeedBonusReward : public Reward {
	public:
		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (!state.prev || !state.goalScored || player.team != ScoringRewardUtil::GetScoringTeam(state))
				return 0;

			return state.prev->ball.vel.Length() / CommonValues::BALL_MAX_SPEED;
		}
	};

	class ConcedeDistanceReward : public Reward {
	public:
		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (!state.prev || !state.goalScored || player.team == ScoringRewardUtil::GetScoringTeam(state))
				return 0;

			float distance = player.pos.Dist(state.prev->ball.pos);
			return -(1 - expf(-distance / CommonValues::CAR_MAX_SPEED));
		}
	};

	class BoostGainReward : public Reward {
	public:
		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			const Player* last = ScoringRewardUtil::GetPrevPlayer(state, player);
			if (!last)
				return 0;

			float boostDiff = sqrtf(player.boost / 100.f) - sqrtf(last->boost / 100.f);
			return boostDiff >= 0 ? boostDiff : 0;
		}
	};

	class BoostLoseReward : public Reward {
	public:
		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			const Player* last = ScoringRewardUtil::GetPrevPlayer(state, player);
			if (!last || player.pos.z >= CommonValues::GOAL_HEIGHT)
				return 0;

			float boostDiff = sqrtf(player.boost / 100.f) - sqrtf(last->boost / 100.f);
			return boostDiff < 0 ? boostDiff * (1 - player.pos.z / CommonValues::GOAL_HEIGHT) : 0;
		}
	};

	class TouchHeightReward : public Reward {
	public:
		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (!state.prev || !player.ballTouchedStep)
				return 0;

			float avgHeight = 0.5f * (player.pos.z + state.ball.pos.z);
			float h0 = ScoringRewardUtil::HeightActivation(0);
			float h1 = ScoringRewardUtil::HeightActivation(CommonValues::CEILING_Z);
			float heightFactor = (ScoringRewardUtil::HeightActivation(avgHeight) - h0) / (h1 - h0);
			return (2 - (float)player.isOnGround) * heightFactor;
		}
	};

	class LowTouchAccelReward : public Reward {
	public:
		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (!state.prev || !player.ballTouchedStep)
				return 0;

			float avgHeight = 0.5f * (player.pos.z + state.ball.pos.z);
			float h0 = ScoringRewardUtil::HeightActivation(0);
			float h1 = ScoringRewardUtil::HeightActivation(CommonValues::CEILING_Z);
			float heightFactor = (ScoringRewardUtil::HeightActivation(avgHeight) - h0) / (h1 - h0);
			return
				(1 - heightFactor) *
				(state.ball.vel - state.prev->ball.vel).Length() /
				CommonValues::CAR_MAX_SPEED;
		}
	};

	class FlipResetReward : public Reward {
	public:
		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			const Player* last = ScoringRewardUtil::GetPrevPlayer(state, player);
			if (!last || !player.ballTouchedStep)
				return 0;

			bool gainedFlip = player.HasFlipOrJump() && !last->HasFlipOrJump();
			bool closeToBall = player.pos.Dist(state.ball.pos) < 2 * CommonValues::BALL_RADIUS;
			bool underBall = ScoringRewardUtil::CosineSimilarity(state.ball.pos - player.pos, -player.rotMat.up) > 0.9f;
			return gainedFlip && player.pos.z > 3 * CommonValues::BALL_RADIUS && closeToBall && underBall ? 1 : 0;
		}
	};
}
