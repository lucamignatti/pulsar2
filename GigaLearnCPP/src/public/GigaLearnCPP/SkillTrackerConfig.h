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
	};
}