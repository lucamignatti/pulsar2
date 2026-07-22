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

#include <cstdlib>
#include <iostream>
#include <string>

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
	// saved weights won't load into a matching architecture.
	PartialModelConfig sharedHeadConfig;
	sharedHeadConfig.layerSizes = { 512, 512 };
	sharedHeadConfig.activationType = ModelActivationType::LEAKY_RELU;
	sharedHeadConfig.addLayerNorm = true;
	// The shared head is a pure feature trunk; the policy head owns the output layer
	// (matches PPOLearnerConfig's own default - InferUnit builds models directly from
	// PartialModelConfig, bypassing that default, so it must be set explicitly here).
	sharedHeadConfig.addOutputLayer = false;

	PartialModelConfig policyConfig;
	policyConfig.layerSizes = { 512, 512, 512 };
	policyConfig.activationType = ModelActivationType::LEAKY_RELU;
	policyConfig.addLayerNorm = true;

	// Optimistic-Critic Ladder wire: the policy head's input is widened by 5 columns
	// ([V_real, V_exp, gap_KD, V_metric, gap_PK], tanh-squashed - see CLAUDE.md's Ladder
	// section). InferUnit/PPOLearner::InferActionsFromModels default to a NULL ladder,
	// which is exact zeros - the same convention every eval/opponent/render path uses,
	// correct here too (Rating-style play, not training collection).
	constexpr int LADDER_WIRE_COLUMNS = 5;

	RG_LOG("Loading GigaLearn checkpoint from \"" << checkpoint << "\" (useGPU=" << useGPU << ")...");
	auto* inferUnit = new InferUnit(
		obsBuilder, obsSize, actionParser,
		sharedHeadConfig, policyConfig,
		checkpoint, useGPU,
		LADDER_WIRE_COLUMNS
	);

	RLBotParams params = {};
	params.tickSkip = 8;      // matches cfg.tickSkip in ExampleMain.cpp
	params.actionDelay = 0;   // matches cfg.actionDelay (5.0v3: zero actuation latency)
	params.inferUnit = inferUnit;

	RLBotClient::Run(params);
	return EXIT_SUCCESS;
}
