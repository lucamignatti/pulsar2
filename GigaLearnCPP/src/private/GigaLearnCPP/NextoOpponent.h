#pragma once
#include "FrameworkTorch.h"

#include <RLGymCPP/Gamestates/GameState.h>

#include <torch/script.h>

namespace GGL {

	// EXTERNAL TRAINING OPPONENT: Nexto (2026-07-20, user-directed "play better
	// bots to force the aerials"). Nexto is a public SSL-level RLGym bot that
	// plays the air constantly - contesting it fills the buffer with the high-ball
	// states our self-play never generates, and it doubles as a FIXED external
	// yardstick (Nexto/Goals For/Against) immune to the version-pool inflation
	// documented in H2_TRUNCATION.md. Same tick_skip (8) as this run; its EARL
	// attention net handles any team size, so it serves 1v1/2v2/3v3 alike.
	//
	// This class adapts Nexto's interface to ours: its TorchScript module
	// (rlbot-run/nexto/nexto-model.pt, loaded verbatim - never trained here), a
	// C++ port of its per-entity obs builder (rlbot-run/nexto/nexto_obs.py is the
	// reference, replicated EXACTLY - including its shipped quirks, e.g. orange
	// observers see mirrored pad positions paired with absolute-order pad flags;
	// fidelity to the trained distribution beats local notions of correctness),
	// and its 90-row action lookup mapped to OUR DefaultAction indices by tuple
	// equality at construction (hard-fails if any row has no match, so table
	// drift can never silently scramble its controls).
	//
	// Threading: owned by the Learner, used ONLY from the collection path (the
	// pipelined worker) - weights are frozen so there is nothing to snapshot.
	class NextoOpponent {
	public:
		torch::jit::script::Module model;
		torch::Device device;
		// Nexto lookup row -> our DefaultAction index (both 90 entries)
		std::vector<int> actionMap;
		// Nexto's raw 8-float rows (its q vector wants the PARSED previous action)
		std::vector<std::array<float, 8>> lookup;
		// Per-global-player previous parsed action (zeroed on episode reset/serve start)
		std::vector<std::array<float, 8>> prevActions;

		NextoOpponent(const std::string& modelPath, torch::Device device);

		// Zero all previous-action memory (call once when a serve iteration begins:
		// Nexto picks up arenas mid-episode, same cold start its RLBot boot has)
		void BeginServe(int numPlayers);

		// Fill outActionIdx (OUR action-table indices) for every player marked in
		// isOld. prevArenaTerminals = envSet->state.terminals at the loop top (the
		// PREVIOUS step's flags): a nonzero entry means that arena's current obs is
		// a fresh episode, so its players' previous actions are zeroed first.
		void Act(const std::vector<RLGC::GameState>& states,
			const std::vector<bool>& isOld,
			const std::vector<uint8_t>& prevArenaTerminals,
			std::vector<int>& outActionIdx);
	};
}
