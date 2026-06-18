#pragma once
#include "Reward.h"

namespace RLGC {
	// This is a base wrapper class that wraps around an existing reward
	class RewardWrapper : public Reward {
	public:

		Reward* child;

		// Assumes ownership
		RewardWrapper(Reward* child)
			: child(child) {}

		~RewardWrapper() {
			delete child;
		}

	protected:
		virtual void Reset(const GameState& initialState) override {
			child->Reset(initialState);
		}

		virtual void PreStep(const GameState& state) override {
			child->PreStep(state);
		}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) {
			return child->GetReward(player, state, isFinal);
		}

		virtual std::string GetName() {
			return child->GetName();
		}
	};
}