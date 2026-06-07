#pragma once
#include "Reward.h"
#include "../Math.h"
#include "../../../RocketSim/src/Sim/BallPredTracker/BallPredTracker.h"
#include <unordered_map>
#include <memory>
#include <cfloat>
#include <cmath>

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

	inline float GoalwardTouchQuality(
		const Player& player,
		const GameState& state,
		float maxRewardedGoalwardAccel = Math::KPHToVel(100),
		float maxRewardedBallSpeed = Math::KPHToVel(120)
	) {
		if (!state.prev || !player.ballTouchedStep)
			return 0;

		Vec targetGoal = player.team == Team::BLUE ? CommonValues::ORANGE_GOAL_BACK : CommonValues::BLUE_GOAL_BACK;
		Vec ballDirToGoal = (targetGoal - state.ball.pos).Normalized();
		Vec touchDeltaVel = state.ball.vel - state.prev->ball.vel;
		float goalwardAccel = touchDeltaVel.Dot(ballDirToGoal);
		if (goalwardAccel <= 0)
			return 0;

		float accelScore = RS_CLAMP(goalwardAccel / maxRewardedGoalwardAccel, 0, 1);
		float directionScore = RS_CLAMP(state.ball.vel.Normalized().Dot(ballDirToGoal), 0, 1);
		float speedScore = RS_CLAMP(state.ball.vel.Length() / maxRewardedBallSpeed, 0, 1);

		return 0.6f * accelScore + 0.25f * directionScore + 0.15f * speedScore;
	}

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

	class TimePenalty : public Reward {
	public:
		float penalty;

		TimePenalty(float penalty = -0.5f) : penalty(penalty) {}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			return state.goalScored ? 0 : penalty;
		}
	};

	class EnergyReward : public Reward {
	public:
		float speedScale, boostScale, flipScale, forwardVelScale;

		EnergyReward(float speedScale = 0.45f, float boostScale = 0.25f, float flipScale = 0.2f, float forwardVelScale = 0.1f)
			: speedScale(speedScale), boostScale(boostScale), flipScale(flipScale), forwardVelScale(forwardVelScale) {}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			float speed = RS_CLAMP(player.vel.Length() / CommonValues::CAR_MAX_SPEED, 0, 1);
			float boost = RS_CLAMP(sqrtf(player.boost / 100.f), 0, 1);
			float flipAvailable = player.HasFlipOrJump() ? 1 : 0;
			float forwardVel = RS_CLAMP(player.rotMat.forward.Dot(player.vel) / CommonValues::CAR_MAX_SPEED, 0, 1);

			return speedScale * speed + boostScale * boost + flipScale * flipAvailable + forwardVelScale * forwardVel;
		}
	};

	class AirTouchReward : public Reward {
	public:
		float minBallHeight, maxBallHeight;
		bool requireGoalwardTouch;

		AirTouchReward(float minBallHeight = 500, float maxBallHeight = CommonValues::CEILING_Z, bool requireGoalwardTouch = true)
			: minBallHeight(minBallHeight), maxBallHeight(maxBallHeight), requireGoalwardTouch(requireGoalwardTouch) {}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (!state.prev || !player.ballTouchedStep || player.isOnGround || state.ball.pos.z < minBallHeight)
				return 0;

			if (requireGoalwardTouch) {
				Vec targetGoal = player.team == Team::BLUE ? CommonValues::ORANGE_GOAL_BACK : CommonValues::BLUE_GOAL_BACK;
				Vec ballDirToGoal = (targetGoal - state.ball.pos).Normalized();
				Vec touchDeltaVel = state.ball.vel - state.prev->ball.vel;

				if (touchDeltaVel.Dot(ballDirToGoal) <= 0)
					return 0;
			}

			float heightRange = RS_MAX(1, maxBallHeight - minBallHeight);
			return RS_CLAMP((state.ball.pos.z - minBallHeight) / heightRange, 0, 1);
		}
	};

	class UsefulAirTouchReward : public Reward {
	public:
		float minBallHeight, maxBallHeight, heightExponent;

		UsefulAirTouchReward(float minBallHeight = 500, float maxBallHeight = CommonValues::CEILING_Z, float heightExponent = 1.35f)
			: minBallHeight(minBallHeight), maxBallHeight(maxBallHeight), heightExponent(heightExponent) {}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (player.isOnGround || state.ball.pos.z < minBallHeight)
				return 0;

			float quality = GoalwardTouchQuality(player, state);
			if (quality <= 0)
				return 0;

			float heightRange = RS_MAX(1, maxBallHeight - minBallHeight);
			float heightScore = RS_CLAMP((state.ball.pos.z - minBallHeight) / heightRange, 0, 1);
			return quality * powf(heightScore, heightExponent);
		}
	};

	class SecondTouchReward : public Reward {
	public:
		float minTouchDelay, followupWindow;
		std::unordered_map<int, float> touchTimers;

		SecondTouchReward(float minTouchDelay = 0.15f, float followupWindow = 3.0f)
			: minTouchDelay(minTouchDelay), followupWindow(followupWindow) {}

		virtual void Reset(const GameState& initialState) override {
			touchTimers.clear();
		}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			auto insertResult = touchTimers.insert({ player.index, followupWindow + 1 });
			float& touchTimer = insertResult.first->second;
			float reward = 0;

			if (player.ballTouchedStep) {
				if (touchTimer >= minTouchDelay && touchTimer <= followupWindow) {
					float quality = GoalwardTouchQuality(player, state);
					float timeScore = 1 - RS_CLAMP((touchTimer - minTouchDelay) / RS_MAX(0.001f, followupWindow - minTouchDelay), 0, 1);
					reward = quality * (0.5f + 0.5f * timeScore);
				}

				touchTimer = 0;
			} else if (touchTimer <= followupWindow) {
				touchTimer += state.deltaTime;
			}

			return reward;
		}
	};

	class PossessionReward : public Reward {
	public:
		float maxDistance, maxRelativeSpeed;

		PossessionReward(float maxDistance = 350, float maxRelativeSpeed = 1400)
			: maxDistance(maxDistance), maxRelativeSpeed(maxRelativeSpeed) {}

		float GetPossessionScore(const Player& candidate, const GameState& state) const {
			float distance = candidate.pos.Dist(state.ball.pos);
			if (distance > maxDistance)
				return 0;

			float relativeSpeed = (state.ball.vel - candidate.vel).Length();
			float proximityScore = 1 - (distance / maxDistance);
			float controlScore = 1 - RS_CLAMP(relativeSpeed / maxRelativeSpeed, 0, 1);

			return proximityScore * controlScore;
		}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			float playerScore = GetPossessionScore(player, state);
			if (playerScore <= 0)
				return 0;

			for (const Player& other : state.players) {
				if (other.index != player.index && GetPossessionScore(other, state) > playerScore)
					return 0;
			}

			return playerScore;
		}
	};

	class OpponentPossessionPenalty : public PossessionReward {
	public:
		OpponentPossessionPenalty(float maxDistance = 350, float maxRelativeSpeed = 1400)
			: PossessionReward(maxDistance, maxRelativeSpeed) {}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			float bestOpponentScore = 0;

			for (const Player& other : state.players) {
				if (other.team != player.team)
					bestOpponentScore = RS_MAX(bestOpponentScore, GetPossessionScore(other, state));
			}

			return -bestOpponentScore;
		}
	};

	class FlipResetReward : public Reward {
	public:
		float minBallHeight;

		FlipResetReward(float minBallHeight = 500) : minBallHeight(minBallHeight) {}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (!player.prev || !player.ballTouchedStep || player.isOnGround || state.ball.pos.z < minBallHeight)
				return 0;

			return player.GotFlipReset() && !player.prev->GotFlipReset();
		}
	};

	class FlipResetFollowupReward : public Reward {
	public:
		float minBallHeight, followupWindow;
		std::unordered_map<int, float> resetTimers;

		FlipResetFollowupReward(float minBallHeight = 500, float followupWindow = 2.0f)
			: minBallHeight(minBallHeight), followupWindow(followupWindow) {}

		virtual void Reset(const GameState& initialState) override {
			resetTimers.clear();
		}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (!player.prev || !state.prev)
				return 0;

			float& resetTimer = resetTimers[player.index];
			bool gotResetNow = player.GotFlipReset() && !player.prev->GotFlipReset() && !player.isOnGround && state.ball.pos.z >= minBallHeight;
			float reward = 0;

			if (resetTimer > 0 && !gotResetNow && player.ballTouchedStep && !player.isOnGround && state.ball.pos.z >= minBallHeight) {
				float quality = GoalwardTouchQuality(player, state);
				if (quality > 0) {
					reward = quality;
					resetTimer = 0;
				}
			}

			resetTimer = RS_MAX(0, resetTimer - state.deltaTime);
			if (gotResetNow)
				resetTimer = followupWindow;

			return reward;
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

	class AerialApproachReward : public Reward {
	public:
		float minBallHeight;

		AerialApproachReward(float minBallHeight = 500) : minBallHeight(minBallHeight) {}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (state.ball.pos.z < minBallHeight || player.isOnGround)
				return 0;

			Vec dirToBall = (state.ball.pos - player.pos).Normalized();
			return RS_CLAMP(player.vel.Dot(dirToBall) / CommonValues::CAR_MAX_SPEED, 0, 1);
		}
	};

	class HeightWeightedAerialApproachReward : public Reward {
	public:
		float minBallHeight, maxBallHeight, heightExponent;

		HeightWeightedAerialApproachReward(float minBallHeight = 500, float maxBallHeight = 1600, float heightExponent = 1.5f)
			: minBallHeight(minBallHeight), maxBallHeight(maxBallHeight), heightExponent(heightExponent) {}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (state.ball.pos.z < minBallHeight || player.isOnGround)
				return 0;

			Vec dirToBall = (state.ball.pos - player.pos).Normalized();
			float approach = RS_CLAMP(player.vel.Dot(dirToBall) / CommonValues::CAR_MAX_SPEED, 0, 1);
			if (approach <= 0)
				return 0;

			float heightRange = RS_MAX(1, maxBallHeight - minBallHeight);
			float heightScore = RS_CLAMP((state.ball.pos.z - minBallHeight) / heightRange, 0, 1);
			return approach * powf(heightScore, heightExponent);
		}
	};

	class BallPredInterceptReward : public Reward {
	public:
		float minBallHeight, maxPredTime, predTimeStep, reachSpeed, reachSlack;
		std::unique_ptr<RocketSim::BallPredTracker> ballPredTracker;
		Arena* ballPredArena = nullptr;

		BallPredInterceptReward(
			float minBallHeight = 500,
			float maxPredTime = 2.0f,
			float predTimeStep = 0.25f,
			float reachSpeed = 1500.0f,
			float reachSlack = 250.0f
		) : minBallHeight(minBallHeight), maxPredTime(maxPredTime), predTimeStep(predTimeStep), reachSpeed(reachSpeed), reachSlack(reachSlack) {}

		virtual void Reset(const GameState& initialState) override {
			ballPredTracker.reset();
			ballPredArena = nullptr;
		}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (!state.lastArena || state.ball.pos.z < minBallHeight)
				return 0;

			if (!ballPredTracker || ballPredArena != state.lastArena) {
				ballPredArena = state.lastArena;
				ballPredTracker = std::make_unique<RocketSim::BallPredTracker>(state.lastArena, 256);
			} else {
				ballPredTracker->UpdatePredFromArena(state.lastArena);
			}

			Vec target = state.ball.pos;
			bool foundReachable = false;
			float bestScore = -FLT_MAX;

			for (float t = predTimeStep; t <= maxPredTime + 0.001f; t += predTimeStep) {
				BallState predBall = ballPredTracker->GetBallStateForTime(t);
				float dist = player.pos.Dist(predBall.pos);
				float reachableDist = reachSpeed * t + reachSlack;
				float score = reachableDist - dist;

				if (score >= 0) {
					target = predBall.pos;
					foundReachable = true;
					break;
				}

				if (score > bestScore) {
					bestScore = score;
					target = predBall.pos;
				}
			}

			Vec dirToTarget = (target - player.pos).Normalized();
			float approach = RS_CLAMP(player.vel.Dot(dirToTarget) / CommonValues::CAR_MAX_SPEED, 0, 1);
			return foundReachable ? approach : approach * 0.5f;
		}
	};

	class KickoffTouchReward : public Reward {
	public:
		float maxTouchTime;
		bool finished = false;
		float elapsed = 0;

		KickoffTouchReward(float maxTouchTime = 3.0f) : maxTouchTime(maxTouchTime) {}

		virtual void Reset(const GameState& initialState) override {
			finished = false;
			elapsed = 0;
		}

		virtual void PreStep(const GameState& state) override {
			if (!finished)
				elapsed += state.deltaTime;
		}

		virtual std::vector<float> GetAllRewards(const GameState& state, bool isFinal) override {
			std::vector<float> rewards(state.players.size(), 0);
			if (finished)
				return rewards;

			bool anyTouch = false;
			for (int i = 0; i < state.players.size(); i++) {
				if (state.players[i].ballTouchedStep) {
					anyTouch = true;
					rewards[i] = RS_CLAMP(1.0f - elapsed / maxTouchTime, 0, 1);
				}
			}

			if (anyTouch)
				finished = true;

			return rewards;
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

	class UsefulFlickReward : public Reward {
	public:
		float maxSetupBallHeight, minUpwardVel, maxRewardedUpwardVel;

		UsefulFlickReward(float maxSetupBallHeight = 450, float minUpwardVel = 300, float maxRewardedUpwardVel = 1400)
			: maxSetupBallHeight(maxSetupBallHeight), minUpwardVel(minUpwardVel), maxRewardedUpwardVel(maxRewardedUpwardVel) {}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (!state.prev || !player.ballTouchedStep || state.prev->ball.pos.z > maxSetupBallHeight)
				return 0;

			float addedUpwardVel = state.ball.vel.z - state.prev->ball.vel.z;
			if (addedUpwardVel < minUpwardVel)
				return 0;

			float quality = GoalwardTouchQuality(player, state);
			if (quality <= 0)
				return 0;

			float upwardScore = RS_CLAMP((addedUpwardVel - minUpwardVel) / RS_MAX(1, maxRewardedUpwardVel - minUpwardVel), 0, 1);
			return quality * upwardScore;
		}
	};

	class ShotOnFrameReward : public Reward {
	public:
		float maxPredTime, maxRewardedBallSpeed;

		ShotOnFrameReward(float maxPredTime = 2.0f, float maxRewardedBallSpeed = RLGC::Math::KPHToVel(120))
			: maxPredTime(maxPredTime), maxRewardedBallSpeed(maxRewardedBallSpeed) {}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (!state.prev || !player.ballTouchedStep)
				return 0;

			float targetY = player.team == Team::BLUE ? CommonValues::BACK_WALL_Y : -CommonValues::BACK_WALL_Y;
			float velY = state.ball.vel.y;
			float yDelta = targetY - state.ball.pos.y;
			if (velY * yDelta <= 0)
				return 0;

			float t = yDelta / velY;
			if (t <= 0 || t > maxPredTime)
				return 0;

			Vec targetPos = state.ball.pos + state.ball.vel * t;
			if (fabsf(targetPos.x) > CommonValues::GOAL_WIDTH_FROM_CENTER || targetPos.z > CommonValues::GOAL_HEIGHT || targetPos.z < 0)
				return 0;

			Vec targetGoal = player.team == Team::BLUE ? CommonValues::ORANGE_GOAL_BACK : CommonValues::BLUE_GOAL_BACK;
			Vec ballDirToGoal = (targetGoal - state.ball.pos).Normalized();
			float alignment = RS_CLAMP(state.ball.vel.Normalized().Dot(ballDirToGoal), 0, 1);
			float speedFrac = RS_CLAMP(state.ball.vel.Length() / maxRewardedBallSpeed, 0, 1);
			return alignment * speedFrac;
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
}
