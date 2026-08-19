// Entry point for the RLBot v5 bot executable (GigaLearnRLBot).
//
// Builds an InferUnit from a trained checkpoint using the SAME obs builder, action
// parser and model shape the bot was trained with (see src/ExampleMain.cpp's
// cfg.ppo.* / cfg.gapSensor.* setup), then connects to the RLBotServer / RLBotSim
// and plays until the match ends.

#include "RLBotClient.h"

#include <RLGymCPP/ObsBuilders/AdvancedObsPadded.h>
#include <RLGymCPP/ActionParsers/DefaultAction.h>
#include <RLGymCPP/Gamestates/GameState.h>
#include <GigaLearnCPP/Util/InferUnit.h>

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace RLGC;
using namespace GGL;

// Must match ExampleMain.cpp's MAX_PLAYERS_PER_TEAM (not exported from that file).
static constexpr int MAX_PLAYERS_PER_TEAM = 3;

// Determine the policy's obs size the way the framework does during environment
// setup: build a representative (1v1) state and measure a single obs vector. This
// must match the obs size the checkpoint was trained with, or InferUnit will abort
// with a clear size-mismatch message.
static int ComputeObsSize(ObsBuilder* obs) {
	GameState gs; // default ctor sizes the boost-pad arrays
	Player a = {}; a.carId = 1; a.team = Team::BLUE;   a.index = 0;
	Player b = {}; b.carId = 2; b.team = Team::ORANGE; b.index = 1;
	gs.players = { a, b };
	return (int)obs->BuildObs(gs.players[0], gs).size();
}

static std::string JoinSizes(const std::vector<int>& v) {
	std::string s;
	for (size_t i = 0; i < v.size(); i++) s += (i ? ", " : "") + std::to_string(v[i]);
	return s;
}

static bool EnvFlag(const char* name) {
	const char* v = std::getenv(name);
	return v && (std::string(v) == "1" || std::string(v) == "true");
}

int main(int argc, char** argv) {
	std::cout << std::unitbuf; // flush every insertion (log files block-buffer otherwise)

	// Checkpoint folder (a saved checkpoints_5.0v3/<step> dir, containing SHARED_HEAD.lt +
	// POLICY.lt): argv[1], else $GGL_CHECKPOINT, else "checkpoint" relative to the
	// executable's working dir.
	std::string checkpoint = "checkpoint";
	if (argc > 1) {
		checkpoint = argv[1];
	} else if (const char* env = std::getenv("GGL_CHECKPOINT")) {
		checkpoint = env;
	}

	bool useGPU = EnvFlag("GGL_USE_GPU");

	// Team-canonical, fixed-width padded obs (4.0/5.0 team-play lineage) - same builder
	// AdvancedObsPadded(3) ExampleMain.cpp's MakeEnv() uses, so 1v1/2v2/3v3 checkpoints
	// all load through the same obs shape.
	auto* obsBuilder = new AdvancedObsPadded(MAX_PLAYERS_PER_TEAM);
	auto* actionParser = new DefaultAction();

	int obsSize = ComputeObsSize(obsBuilder);
	RG_LOG("RLBot bot obs size: " << obsSize);

	// Must mirror ExampleMain.cpp's cfg.ppo.sharedHead / cfg.ppo.policy exactly, or the
	// saved weights won't load into a matching architecture. Current lineage: the
	// residual/asymmetric net (trunk 1152 kept wide, policy SHRUNK to 768).
	//
	// addResiduals IS LOAD-INVISIBLE AND BEHAVIOUR-CRITICAL: the module list stays flat,
	// so a residual checkpoint loads shape-clean into a non-residual model and then plays
	// WRONG silently (Model::Forward applies the skips from recorded index spans). A
	// mismatch here produces no error at all - only a bot that looks lobotomized.
	bool addResiduals = true;

	// Derived from the checkpoint's own weights (see ReadLayerSizesFromModule). The shared
	// head has no output projection; the policy head's last 2-D weight IS its output layer,
	// so drop it -- addOutputLayer re-adds it.
	std::vector<int> sharedSizes, policySizes;
	try {
		sharedSizes = ReadLayerSizesFromModule(checkpoint + "/SHARED_HEAD.lt", false);
		policySizes = ReadLayerSizesFromModule(checkpoint + "/POLICY.lt", true);
	} catch (const std::exception& e) {
		RG_ERR_CLOSE("Could not read layer sizes from checkpoint \"" << checkpoint
			<< "\": " << e.what());
	}
	if (sharedSizes.empty() || policySizes.empty())
		RG_ERR_CLOSE("Checkpoint \"" << checkpoint << "\" has no 2-D weights to size from");
	RG_LOG("Architecture derived from checkpoint: trunk { " << JoinSizes(sharedSizes)
		<< " }, policy { " << JoinSizes(policySizes) << " }");

	PartialModelConfig sharedHeadConfig;
	sharedHeadConfig.layerSizes = sharedSizes;
	sharedHeadConfig.addResiduals = addResiduals;
	sharedHeadConfig.activationType = ModelActivationType::LEAKY_RELU;
	sharedHeadConfig.addLayerNorm = true;
	// The shared head is a pure feature trunk; the policy head owns the output layer
	// (matches PPOLearnerConfig's own default - InferUnit builds models directly from
	// PartialModelConfig, bypassing that default, so it must be set explicitly here).
	sharedHeadConfig.addOutputLayer = false;

	PartialModelConfig policyConfig;
	policyConfig.layerSizes = policySizes;
	policyConfig.addResiduals = addResiduals;
	policyConfig.activationType = ModelActivationType::LEAKY_RELU;
	policyConfig.addLayerNorm = true;

	// MoE trunk: geometry comes from the 3-D expert stacks, not from the 2-D layer scan
	// (which misreads each block's [E, dim] router as an E-wide layer and then trips
	// addResiduals' uniform-width check). Without this no MoE checkpoint can be played.
	// GGL_MOE_TOPK must match training - it is a routing scalar, not a saved parameter.
	{
		const char* tk = std::getenv("GGL_MOE_TOPK");
		if (ReadMoEConfigFromModule(checkpoint + "/SHARED_HEAD.lt", sharedHeadConfig,
				(tk && *tk) ? std::atoi(tk) : 4)) {
			policyConfig.addResiduals = false;
			if (policySizes.size() > 1)
				policyConfig.layerSizes = { policySizes.front() };
		}
	}

	// NOTE: the policy head is at plain trunk width. The Optimistic-Critic Ladder's
	// 5-column policy wire (which made this head's input 512+5=517 on the 5.0v3 lineage
	// from 18.88B on) was REMOVED 2026-07-25 with the rest of the Ladder actuation; the
	// composition critic that replaced it never feeds H to the policy. Ladder-era 5.0v3
	// checkpoints therefore DO NOT load here - they abort with 264704 vs 262144 on
	// POLICY.lt layer 0. Restore tag for that architecture: pre-strip-20260725.
	RG_LOG("Loading GigaLearn checkpoint from \"" << checkpoint << "\" (useGPU=" << useGPU << ")...");
	auto* inferUnit = new InferUnit(
		obsBuilder, obsSize, actionParser,
		sharedHeadConfig, policyConfig,
		checkpoint, useGPU);

	// tickSkip MUST match the lineage the checkpoint was trained at, and unlike the layer
	// sizes a mismatch is SILENT: the bot simply decides at the wrong rate and plays badly
	// with no error. 5.x lineages are ts8 (15 Hz); 6.0 is ts1 (120 Hz). It is not recorded
	// in the checkpoint, so it is an explicit env knob and is always logged.
	RLBotParams params = {};
	params.tickSkip = 8;
	if (const char* ts = std::getenv("GGL_TICK_SKIP")) {
		int v = std::atoi(ts);
		if (v >= 1 && v <= 120) params.tickSkip = v;
		else RG_ERR_CLOSE("GGL_TICK_SKIP=\"" << ts << "\" is out of range (1..120)");
	}
	params.actionDelay = 0;   // 0 on every current lineage: the game's ~2-tick send->applied
	                          // delay is cancelled by the ~2-tick staleness of the packet the
	                          // policy acts on (SIM2REAL_AUDIT.md S4).
	RG_LOG("Decision rate: tickSkip=" << params.tickSkip << " ("
		<< (120.0 / params.tickSkip) << " Hz), actionDelay=" << params.actionDelay
		<< "  [override with GGL_TICK_SKIP]");
	params.inferUnit = inferUnit;

	RLBotClient::Run(params);
	return EXIT_SUCCESS;
}
