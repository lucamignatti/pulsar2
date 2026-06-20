#pragma once
#include "Reward.h"

// https://github.com/AechPro/rocket-league-gym-sim/blob/main/rlgym_sim/utils/reward_functions/reward_function.py
namespace RLGC {
	template<typename T>
	class PlayerReward : public Reward {
	private:
		std::vector<T*> _instances;

	public:
		virtual void Reset(const GameState& initialState) {
			if (_instances.empty()) {
				// Generate instances
				for (int i = 0; i < (int)initialState.players.size(); i++)
					_instances.push_back(new T());
			}

			for (auto inst : _instances)
				inst->Reset(initialState);
		}

		virtual void PreStep(const GameState& state) {
			for (auto inst : _instances)
				inst->PreStep(state);
		}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) {
			return _instances[player.index]->GetReward(player, state, isFinal);
		}

		// Get all rewards for all players
		virtual std::vector<float> GetAllRewards(const GameState& state, bool isFinal) {

			std::vector<float> rewards = std::vector<float>(state.players.size());
			for (int i = 0; i < state.players.size(); i++) {
				rewards[i] = GetReward(state.players[i], state, isFinal);
			}

			return rewards;
		}

		virtual ~PlayerReward() {
			for (auto inst : _instances)
				delete inst;
		};
	};
}