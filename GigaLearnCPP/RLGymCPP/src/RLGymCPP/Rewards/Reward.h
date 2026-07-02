#pragma once
#include "../Gamestates/GameState.h"
#include "../BasicTypes/Action.h"

// https://github.com/AechPro/rocket-league-gym-sim/blob/main/rlgym_sim/utils/reward_functions/reward_function.py
namespace RLGC {
	class Reward {
	private:
		std::string _cachedName = {};

	public:
		virtual void Reset(const GameState& initialState) {}

		virtual void PreStep(const GameState& state) {}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) {
			throw std::runtime_error("GetReward() is unimplemented");
			return 0;
		}

		// Get all rewards for all players
		virtual std::vector<float> GetAllRewards(const GameState& state, bool isFinal) {

			std::vector<float> rewards = std::vector<float>(state.players.size());
			for (int i = 0; i < state.players.size(); i++) {
				rewards[i] = GetReward(state.players[i], state, isFinal);
			}

			return rewards;
		}

		virtual std::string GetName() {

			if (!_cachedName.empty())
				return _cachedName;

			std::string rewardName = typeid(*this).name();

			// Trim the string to after cetain keys
			{
				constexpr const char* TRIM_KEYS[] = {
					"::", // Namespace separator
					" " // Any spaces
				};
				for (const char* key : TRIM_KEYS) {
					size_t idx = rewardName.rfind(key);
					if (idx == std::string::npos)
						continue;

					rewardName.erase(rewardName.begin(), rewardName.begin() + idx + strlen(key));
				}
			}

			_cachedName = rewardName;
			return rewardName;
		}

		virtual ~Reward() {};
	};

	struct WeightedReward {
		Reward* reward;
		float weight;
		// Gated rewards can be scaled down by the learner's reachability gate (when enabled);
		// ungated rewards always pay in full
		bool gated;

		WeightedReward(Reward* reward, float scale, bool gated = false) : reward(reward), weight(scale), gated(gated) {}
		WeightedReward(Reward* reward, int scale, bool gated = false) : reward(reward), weight((float)scale), gated(gated) {}
	};
}