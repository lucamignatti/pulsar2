#pragma once
#include "Framework.h"

namespace GGL {
	// GGL_EXPAND_K/_IN/_OUT: net2net width expansion of a saved dense policy.
	RG_IMEXPORT int RunExpandCheckpoint();
	// Offline pure-sim goal-share eval vs the embedded Nexto (the qualifier yardstick:
	// "score 42 before Nexto scores 28" is a race over at most 69 goals, so the whole
	// question reduces to the per-goal win rate p - P(qualify per attempt) =
	// P(Binomial(69, p) >= 42)). Runs INSTEAD of the trainer when GGL_NEXTO_EVAL=1,
	// CPU-only, no RLBot, no learn pass; safe beside a live training process when
	// niced. The Pulsar side goes through InferUnit exactly like GigaLearnRLBot
	// (weights-derived layer sizes, argmax by default), so the measured p is the
	// DEPLOYMENT policy's, not the training-time sampling policy's.
	//
	// Env knobs: GGL_CHECKPOINT (required, a saved checkpoint dir - copy it out of the
	// live rotation first), GGL_NEXTO_MODEL (default rlbot-run/nexto/nexto-model.pt),
	// GGL_EVAL_ARENAS (8), GGL_EVAL_GOALS (200, stop target), GGL_EVAL_EP_CAP_S (300),
	// GGL_SAMPLE_ACTIONS (0 = argmax, deployment parity), GGL_TICK_SKIP (8),
	// GGL_EVAL_TORCH_THREADS (4).
	RG_IMEXPORT int RunNextoEval();
}
