#pragma once
#include "RewardWrapper.h"

namespace RLGC {
	// This is a wrapper class that makes another reward function zero-sum and team-distributed
	// Per-player reward is calculated using: ownReward*(1-teamSpirit) + avgTeamReward*teamSpirit - avgOpponentReward
	class ZeroSumReward : public RewardWrapper {
	public:

		float teamSpirit, opponentScale;

		// Team spirit is the fraction of reward shared between teammates
		// Opponent scale is the scale of punishment for opponent rewards (normally 1, non-1 is no longer zero-sum)
		ZeroSumReward(Reward* child, float teamSpirit, float opponentScale = 1, bool ownsFunc = true)
			: RewardWrapper(child), teamSpirit(teamSpirit), opponentScale(opponentScale) {

		}

	protected:

		// Inner (pre-zero-sum) child rewards from the last GetAllRewards, kept for logging
		std::vector<float> _lastRewards = {};

		// Get all rewards for all players
		virtual std::vector<float> GetAllRewards(const GameState& state, bool final) override;

		// Log the inner child reward, not the team-distributed output
		virtual float GetLoggableReward(int playerIndex, const std::vector<float>& lastOutput) override {
			return (playerIndex < (int)_lastRewards.size()) ? _lastRewards[playerIndex] : lastOutput[playerIndex];
		}
	};
}