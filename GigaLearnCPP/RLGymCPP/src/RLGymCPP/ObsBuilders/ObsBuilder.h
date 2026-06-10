#pragma once
#include "../Gamestates/GameState.h"
#include "../BasicTypes/Action.h"
#include "../BasicTypes/Lists.h"

// https://github.com/AechPro/rocket-league-gym-sim/blob/main/rlgym_sim/utils/obs_builders/obs_builder.py
namespace RLGC {
	class ObsBuilder {
	public:
		virtual void Reset(const GameState& initialState) {}

		// NOTE: May be called once during environment initialization to determine policy neuron size
		virtual FList BuildObs(const Player& player, const GameState& state) = 0;

		// Returns the index of the first car-local ball component (pos+vel, 6 floats) in a built obs.
		// Returns -1 if not supported by this obs builder (GCRL car_critic will be skipped).
		virtual int GetCarLocalBallOffset() const { return -1; }

		// Whether this builder x-mirrors the obs for this player in the current state.
		// Must be evaluated against the same game state the obs was built from; used by
		// GCRL hindsight relabeling to keep goals in a consistent mirror frame.
		virtual bool IsObsMirroredX(const Player& player) const { return false; }
	};
}