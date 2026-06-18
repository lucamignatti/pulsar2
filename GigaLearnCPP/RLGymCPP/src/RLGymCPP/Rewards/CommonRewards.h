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

	// Port of the last pre-Tecko Nexto reward from Rolv-Arild/Necto commit ff1ced0.
	class NextoReward : public Reward {
	public:
		float teamSpirit, goalW, goalDistW, goalSpeedBonusW, goalDistBonusW, demoW;
		float distW, alignW, boostGainW, boostLoseW, touchGrassW, touchHeightW;
		float touchAccelW, flipResetW, opponentPunishW;

		float stateQuality = 0;
		std::vector<float> playerQualities;
		std::vector<float> rewards;

		NextoReward(
			float teamSpirit = 0.6f,
			float goalW = 12.5f,
			float goalDistW = 2.5f,
			float goalSpeedBonusW = 1.25f,
			float goalDistBonusW = 1.25f,
			float demoW = 5.f,
			float distW = 0.f,
			float alignW = 0.f,
			float boostGainW = 0.7f,
			float boostLoseW = 0.4f,
			float touchGrassW = 0.f,
			float touchHeightW = 1.f,
			float touchAccelW = 0.f,
			float flipResetW = 5.f,
			float opponentPunishW = 1.f
		) :
			teamSpirit(teamSpirit), goalW(goalW), goalDistW(goalDistW),
			goalSpeedBonusW(goalSpeedBonusW), goalDistBonusW(goalDistBonusW), demoW(demoW),
			distW(distW), alignW(alignW), boostGainW(boostGainW), boostLoseW(boostLoseW),
			touchGrassW(touchGrassW), touchHeightW(touchHeightW), touchAccelW(touchAccelW),
			flipResetW(flipResetW), opponentPunishW(opponentPunishW) {
		}

		virtual void Reset(const GameState& initialState) override {
			rewards.clear();
			auto qualities = GetStateQualities(initialState);
			stateQuality = qualities.first;
			playerQualities = qualities.second;
		}

		virtual void PreStep(const GameState& state) override {
			CalculateRewards(state);
		}

		virtual std::vector<float> GetAllRewards(const GameState& state, bool isFinal) override {
			if (rewards.size() != state.players.size())
				CalculateRewards(state);

			return rewards;
		}

	private:
		static float SafeExpDist(Vec a, Vec b, float scale) {
			return expf(-a.Dist(b) / scale);
		}

		static float CosineSimilarity(Vec a, Vec b) {
			float denom = a.Length() * b.Length();
			if (denom <= 1e-6f)
				return 0;

			return a.Dot(b) / denom;
		}

		static float HeightActivation(float z) {
			return cbrtf((z - CommonValues::GOAL_HEIGHT) / CommonValues::CEILING_Z);
		}

		static float MeanTeamReward(const std::vector<float>& teamRewards, const std::vector<int>& indices) {
			if (indices.empty())
				return 0;

			float total = 0;
			for (int idx : indices)
				total += teamRewards[idx];

			return total / indices.size();
		}

		static void GetTeamIndices(const GameState& state, std::vector<int>& blue, std::vector<int>& orange) {
			blue.clear();
			orange.clear();
			for (int i = 0; i < state.players.size(); i++) {
				if (state.players[i].team == Team::BLUE)
					blue.push_back(i);
				else
					orange.push_back(i);
			}
		}

		std::pair<float, std::vector<float>> GetStateQualities(const GameState& state) const {
			Vec ballPos = state.ball.pos;
			Vec blueGoal = (CommonValues::BLUE_GOAL_BACK + CommonValues::BLUE_GOAL_CENTER) / 2;
			Vec orangeGoal = (CommonValues::ORANGE_GOAL_BACK + CommonValues::ORANGE_GOAL_CENTER) / 2;

			float quality =
				0.5f * goalDistW *
				(SafeExpDist(orangeGoal, ballPos, CommonValues::CAR_MAX_SPEED) -
					SafeExpDist(blueGoal, ballPos, CommonValues::CAR_MAX_SPEED));

			std::vector<float> qualities(state.players.size(), 0);
			for (int i = 0; i < state.players.size(); i++) {
				const Player& player = state.players[i];
				Vec pos = player.pos;

				float alignment =
					0.5f *
					(CosineSimilarity(ballPos - pos, CommonValues::ORANGE_GOAL_BACK - pos) -
						CosineSimilarity(ballPos - pos, CommonValues::BLUE_GOAL_BACK - pos));
				if (player.team == Team::ORANGE)
					alignment *= -1;

				float liuDist = SafeExpDist(ballPos, pos, 1410.f);
				qualities[i] = distW * liuDist + alignW * alignment;
			}

			return { quality / 2, qualities };
		}

		void ApplyTeamMix(std::vector<float>& playerRewards, const GameState& state) const {
			std::vector<int> blue, orange;
			GetTeamIndices(state, blue, orange);

			float blueMean = MeanTeamReward(playerRewards, blue);
			float orangeMean = MeanTeamReward(playerRewards, orange);

			for (int idx : blue)
				playerRewards[idx] =
					(1 - teamSpirit) * playerRewards[idx] +
					teamSpirit * blueMean -
					opponentPunishW * orangeMean;

			for (int idx : orange)
				playerRewards[idx] =
					(1 - teamSpirit) * playerRewards[idx] +
					teamSpirit * orangeMean -
					opponentPunishW * blueMean;
		}

		void ApplyGoalRewards(std::vector<float>& playerRewards, const GameState& state) const {
			if (!state.prev || !state.goalScored)
				return;

			Team scoringTeam = RS_TEAM_FROM_Y(state.ball.pos.y) == Team::BLUE ? Team::ORANGE : Team::BLUE;
			float goalSpeed = state.prev->ball.vel.Length();

			for (int i = 0; i < state.players.size(); i++) {
				const Player& player = state.players[i];
				if (player.team == scoringTeam) {
					playerRewards[i] = goalW + goalSpeedBonusW * goalSpeed / CommonValues::BALL_MAX_SPEED;
				} else {
					float distance = player.pos.Dist(state.prev->ball.pos);
					playerRewards[i] = -goalDistBonusW * (1 - expf(-distance / CommonValues::CAR_MAX_SPEED));
				}
			}
		}

		void CalculateRewards(const GameState& state) {
			if (!state.prev || playerQualities.size() != state.players.size()) {
				Reset(state);
				rewards = std::vector<float>(state.players.size(), 0);
				return;
			}

			auto qualities = GetStateQualities(state);
			float newStateQuality = qualities.first;
			std::vector<float>& newPlayerQualities = qualities.second;

			std::vector<float> playerRewards(state.players.size(), 0);
			float ballHeight = state.ball.pos.z;
			float h0 = HeightActivation(0);
			float h1 = HeightActivation(CommonValues::CEILING_Z);

			for (int i = 0; i < state.players.size(); i++) {
				const Player& player = state.players[i];
				const Player& last = state.prev->players[i];
				float carHeight = player.pos.z;

				if (player.ballTouchedStep) {
					float avgHeight = 0.5f * (carHeight + ballHeight);
					float heightFactor = (HeightActivation(avgHeight) - h0) / (h1 - h0);
					playerRewards[i] += touchHeightW * (2 - (float)player.isOnGround) * heightFactor;

					bool gainedFlip = player.HasFlipOrJump() && !last.HasFlipOrJump();
					bool closeToBall = player.pos.Dist(state.ball.pos) < 2 * CommonValues::BALL_RADIUS;
					bool underBall = CosineSimilarity(state.ball.pos - player.pos, -player.rotMat.up) > 0.9f;
					if (gainedFlip && carHeight > 3 * CommonValues::BALL_RADIUS && closeToBall && underBall)
						playerRewards[i] += flipResetW;

					playerRewards[i] +=
						touchAccelW *
						(1 - heightFactor) *
						(state.ball.vel - state.prev->ball.vel).Length() /
						CommonValues::CAR_MAX_SPEED;
				}

				float boostDiff = sqrtf(player.boost / 100.f) - sqrtf(last.boost / 100.f);
				if (boostDiff >= 0) {
					playerRewards[i] += boostGainW * boostDiff;
				} else if (carHeight < CommonValues::GOAL_HEIGHT) {
					playerRewards[i] += boostLoseW * boostDiff * (1 - carHeight / CommonValues::GOAL_HEIGHT);
				}

				playerRewards[i] -= (float)player.isOnGround * touchGrassW;

				if (player.isDemoed && !last.isDemoed)
					playerRewards[i] -= demoW / 2;
				if (player.eventState.demo)
					playerRewards[i] += demoW / 2;
			}

			for (int i = 0; i < state.players.size(); i++) {
				playerRewards[i] += newPlayerQualities[i] - playerQualities[i];
				if (state.players[i].team == Team::BLUE)
					playerRewards[i] += newStateQuality - stateQuality;
				else
					playerRewards[i] -= newStateQuality - stateQuality;
			}

			playerQualities = newPlayerQualities;
			stateQuality = newStateQuality;

			ApplyGoalRewards(playerRewards, state);
			ApplyTeamMix(playerRewards, state);

			rewards = playerRewards;
		}
	};
}
