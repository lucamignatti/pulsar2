#pragma once

#include "Framework.h"

#include <RLGymCPP/EnvSet/EnvSet.h>

namespace GGL {
	struct SkillTrackerConfig {
		bool enabled = false;

		// Number of arenas for evaluation
		// Don't put this much higher than your CPU thread count
		int numArenas = 16;

		// Optional create-func for the eval arenas. Unset (default) clones the TRAINING
		// create-func over numArenas LOW indices — with a mixed-mode fleet whose team arenas
		// sit on trailing indices, that means the tracker only ever plays (and reports Elo
		// for) the primary mode. Set this to lay out the eval fleet's own mode mix so team
		// modes get evaluated too (each mode's goals feed its own lazily-created
		// "Rating/<mode>" key). Keep index 0 on the primary mode: it feeds the render sender,
		// and Rating/<primary> should stay continuous when the training mix changes.
		// Only the arena (team size), obs builder and action parser matter here — the version
		// manager overwrites eval rewards, state setters and terminal conditions.
		RLGC::EnvCreateFn envCreateFn = nullptr;

		// Time (in seconds) to simulate each skill rating run, per arena
		float simTime = 45;

		// Maximum time (in seconds) to simulate games for (games are reset after the maximum time is exceeded)
		// If this max time is too low, the rating quality may be poor because the bots wont have enough time to score goals
		float maxSimTime = 240; 

		// Number of iterations between running skill rating games
		int updateInterval = 16;

		// Rating increment scale per-goal
		float ratingInc = 5; 

		// Initial rating of the first version
		float initialRating = 0; 

		// Policies will be inferred deterministically
		// Off by default since the learning algorithm is trying to optimize the stochastic policy
		bool deterministic = false;

		// === PERMANENT REFERENCE SET: the non-inflating skill read (2026-07-25) ===
		//
		// Rating/1v1 above is a TREADMILL. AddVersion copies the main's CURRENT rating into every
		// new version, and each goal moves both sides, so the pool's rating tracks the agent and
		// the number rises whether or not real skill does - measured ~6x overstatement (54% real
		// win share against a predicted 75%). Enabling trainAgainstOldVersions makes it strictly
		// worse, because the run is then optimizing directly against its own measuring stick.
		//
		// References fix that by construction: a SEPARATE vector, log-spaced over the run,
		// NEVER trained against, and with the OLDEST entry never evicted - so `Ref/Oldest Share`
		// is measured against a genuinely fixed opponent and cannot drift. Goal share, not Elo:
		// share against a frozen opponent needs no rating bookkeeping and is exactly the method
		// research/tools/anchor_battery.py already uses as the honest offline yardstick. Counters
		// are cumulative and persisted, so (as with Nexto/Goals *) the SLOPE is the signal.
		//
		// 0 disables the whole reference path.
		int maxReferences = 8;

		// Iterations between reference battery matches. Slower than updateInterval: this is a
		// slow, honest signal and it borrows the barrier budget the league's evolve step used
		// to spend (~3.76 s per 16 iterations), so a slower cadence keeps the change cost-negative.
		int referenceUpdateInterval = 64;
	};
}