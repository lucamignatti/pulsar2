#pragma once
#include "AdvancedObs.h"

namespace RLGC {
	// AdvancedObs with a FIXED obs size for any team size up to maxPlayersPerTeam, so one
	// policy net can train across 1v1/2v2/3v3 (and carry a checkpoint between them).
	//
	// Layout: AdvancedObs's header and self block are kept byte-identical (ball at 0,
	// self right after the header — offsets the steering landing sims and analysis
	// tooling read), then teammate/opponent slots are padded to fixed counts with
	// all-zero rows, then one presence flag per slot (1 = real player) is appended at
	// the very end so the 29-float per-player stride is preserved.
	//
	// Real players are shuffled across their slots every build to prevent slot bias
	// (same rationale as DefaultObsPadded; safe because the policy is feedforward, and
	// it means every slot's weights train even while the arena mix is mostly 1v1).
	class AdvancedObsPadded : public AdvancedObs {
	public:
		int maxPlayersPerTeam;

		AdvancedObsPadded(int maxPlayersPerTeam) : maxPlayersPerTeam(maxPlayersPerTeam) {}

		virtual FList BuildObs(const Player& player, const GameState& state) override;
	};
}
