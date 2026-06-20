#pragma once
#include "../Gamestates/GameState.h"
#include "../BasicTypes/Action.h"

#include <typeinfo>
#include <cstdlib>
#if defined(__GNUC__) || defined(__clang__)
#include <cxxabi.h>
#endif

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

#if defined(__GNUC__) || defined(__clang__)
			// Under the Itanium ABI (GCC/Clang) typeid().name() is mangled,
			// e.g. "N4RLGC14TeamGoalRewardE". Demangle it to a readable form
			// like "RLGC::TeamGoalReward" before we trim the namespace.
			{
				int status = 0;
				char* demangled = abi::__cxa_demangle(rewardName.c_str(), nullptr, nullptr, &status);
				if (status == 0 && demangled)
					rewardName = demangled;
				std::free(demangled);
			}
#endif

			// Strip the leading namespace qualifier (e.g. "RLGC::") so we log
			// "TeamGoalReward" rather than the fully-qualified name. Only trim
			// "::" occurring before the first '<' so template arguments are
			// preserved (e.g. PlayerDataEventReward<&PlayerEventState::demo,
			// false> stays distinct from the demoed instantiation).
			{
				size_t templateStart = rewardName.find('<');
				size_t searchLimit = (templateStart == std::string::npos) ? rewardName.size() : templateStart;
				size_t nsIdx = rewardName.rfind("::", searchLimit);
				if (nsIdx != std::string::npos)
					rewardName.erase(rewardName.begin(), rewardName.begin() + nsIdx + 2);
			}

			_cachedName = rewardName;
			return rewardName;
		}

		virtual ~Reward() {};
	};

	struct WeightedReward {
		Reward* reward;
		float weight;

		WeightedReward(Reward* reward, float scale) : reward(reward), weight(scale) {}
		WeightedReward(Reward* reward, int scale) : reward(reward), weight(scale) {}
	};
}