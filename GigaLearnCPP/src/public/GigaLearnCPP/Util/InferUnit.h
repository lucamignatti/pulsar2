#pragma once

#include "ModelConfig.h"

namespace GGL {
	struct RG_IMEXPORT InferUnit {
		int obsSize;
		RLGC::ObsBuilder* obsBuilder;
		RLGC::ActionParser* actionParser;
		struct ModelSet* models;
		bool useGPU;

		// NOTE: Reset() will never be called on your obs
		InferUnit(
			RLGC::ObsBuilder* obsBuilder, int obsSize, RLGC::ActionParser* actionParser,
			PartialModelConfig sharedHeadConfig, PartialModelConfig policyConfig,
			std::filesystem::path modelsFolder, bool useGPU,
			// Ladder wire columns appended to the policy head input (PPOLearnerConfig::extraPolicyInputs).
			// Must match the checkpoint's cfg.gapSensor.wireEnabled state or LoadFrom's size check
			// rejects the weights (0 for pre-Ladder checkpoints, 5 for the current 5.0v3 run).
			int extraPolicyInputs = 0);


		RLGC::Action InferAction(const RLGC::Player& player, const RLGC::GameState& state, bool deterministic, float temperature = 1);
		std::vector<RLGC::Action> BatchInferActions(const std::vector<RLGC::Player>& players, const std::vector<RLGC::GameState>& states, bool deterministic, float temperature = 1);

		// TODO: Add deconstructor (make sure to free models too)
	};
}