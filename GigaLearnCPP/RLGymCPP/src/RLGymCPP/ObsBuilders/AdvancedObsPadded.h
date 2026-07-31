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
	//
	// That shuffle draws from a clock-seeded engine, so the observation is
	// NONDETERMINISTIC for a fixed game state. Harmless while training; fatal for the
	// viewer, where it means restoring a state and playing forward twice gives two
	// different plays (measured: the obs diverged on the very first inference after a
	// byte-identical restore). Set shuffleSlots false to fill slots in order — the
	// policy is trained to be slot-invariant precisely so this costs nothing.
	// Keep it ON for training, or every slot past the first stops receiving gradient.
	class AdvancedObsPadded : public AdvancedObs {
	public:
		int maxPlayersPerTeam;
		bool shuffleSlots;

		AdvancedObsPadded(int maxPlayersPerTeam, bool shuffleSlots = true)
			: maxPlayersPerTeam(maxPlayersPerTeam), shuffleSlots(shuffleSlots) {}

		virtual FList BuildObs(const Player& player, const GameState& state) override;
	};
}
