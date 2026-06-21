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

		// Returns the index of the first car-local ball component (pos+vel, 6 floats) within a built obs.
		// Used by the GCRL car critic's hindsight relabeling. Returns -1 if this builder does not expose
		// a car-local ball block, in which case the car critic is skipped.
		virtual int GetCarLocalBallOffset() const { return -1; }
	};
}