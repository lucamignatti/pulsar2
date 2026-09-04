#pragma once
// Config for the dip-triggered search-and-imitate actuator (Util/DipSearch.h). Lives in its
// own header so LearnerConfig can carry it without pulling the pool/engine headers in.

namespace GGL {
	struct DipSearchConfig {
		bool enabled = false;

		// -- trigger --
		float arenaFrac = 0.25f;     // fraction of arenas that bank a snapshot every step
		int k = 15;                  // realised-advantage window, decision steps (1 s at 15 Hz)
		float dipQuantile = 0.03f;   // bottom quantile of A_k that counts as a dip
		int maxStates = 32;          // states searched per iteration (cap)

		// -- search --
		int poolSize = 512;          // private arenas stepped in lockstep
		int horizon = 60;            // rollout length, decision steps (4 s at 15 Hz)
		int macro = 3;               // steps an open-loop action is held
		int nBaseline = 16;          // fresh policy rollouts per state (the "what I do" estimate)
		int nRandom = 24;            // uniform-valid open-loop prefixes, L in {2,4,8}
		int nTemp = 24;              // policy-at-temperature prefixes (temp 2/L4, 4/L4, 2/L8)
		int nFinalists = 8;          // top prefixes re-evaluated fresh
		int kFinal = 8;              // evaluations per finalist
		int kHeldout = 32;           // held-out evaluations of the winner (the reported gain)
		float budgetSecs = 6.0f;     // wall-clock cap per iteration; stops between state groups
		int threads = 8;             // std::threads stepping the pool

		// -- consumer --
		float minGain = 0.5f;        // critic units (~1/15 goal at Returns STD 20): accept below this = noise
		float weightCap = 3.0f;      // imitation weight cap, critic units (SIL caps at 2 sigma of R-V)
		int imitateRollouts = 4;     // held-out rollouts whose prefix steps become imitation rows
	};
}
