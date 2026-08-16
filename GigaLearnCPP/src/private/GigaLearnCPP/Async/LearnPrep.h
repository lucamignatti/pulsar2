#pragma once

#include "Fragment.h"
#include <private/GigaLearnCPP/PPO/ExperienceBuffer.h>
#include <GigaLearnCPP/Util/Report.h>
#include <GigaLearnCPP/LearnerConfig.h>
#include <GigaLearnCPP/Distributed/Session.h>

#include <functional>
#include <vector>

namespace GGL {

class PPOLearner;
struct WelfordStat;

// Cat collector columns, one InferValueFamily on the cat, temporal GAE per fragment,
// then after-cat goal blend + adv-filter. SIL/vdag-inject/hull/gap stay in titan
// Learner.cpp until those helpers are hoisted (need gapSensor / Trajectory).
void PrepareExperienceBatched(
	PPOLearner* ppo,
	const LearnerConfig& config,
	std::vector<TrajectoryFragment>& frags,
	ExperienceTensors& out,
	Report& report,
	WelfordStat* returnStat,
	Dist::Session* dist,
	int obsSize,
	const std::function<void()>& onChunkDrain);

}
