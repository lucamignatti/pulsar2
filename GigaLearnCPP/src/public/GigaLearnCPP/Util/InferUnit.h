#pragma once

#include "ModelConfig.h"

namespace GGL {
	// Hidden-layer widths read straight out of a saved .lt module's 2-D (Linear) weights.
	//
	// The RLBot bot and the viz both used to hand-copy these from ExampleMain.cpp, which
	// broke silently on every lineage that resized the net (1152/768 on 5.3, 896/640 on
	// 6.0 ts1). The weights already carry the answer. Set dropOutput for a head whose last
	// 2-D weight is its output projection (addOutputLayer re-adds it).
	RG_IMEXPORT std::vector<int> ReadLayerSizesFromModule(const std::string& ltPath, bool dropOutput);

	struct RG_IMEXPORT InferUnit {
		int obsSize;
		RLGC::ObsBuilder* obsBuilder;
		RLGC::ActionParser* actionParser;
		struct ModelSet* models;
		bool useGPU;

		// Optional per-row capture of exactly what inference consumed and chose. The obs
		// here is THE vector fed to the net, not a rebuild - AdvancedObsPadded's slot
		// shuffle draws from a clock-seeded engine, so building the obs a second time
		// outside this call can assign players to different slots and silently log
		// something the policy never saw (the same trap the viz obs-hash caught).
		struct InferDebug {
			RLGC::FList obs;
			std::vector<uint8_t> actionMask;
			int actionIndex = -1;
		};

		// NOTE: Reset() will never be called on your obs
		InferUnit(
			RLGC::ObsBuilder* obsBuilder, int obsSize, RLGC::ActionParser* actionParser,
			PartialModelConfig sharedHeadConfig, PartialModelConfig policyConfig,
			std::filesystem::path modelsFolder, bool useGPU);


		RLGC::Action InferAction(const RLGC::Player& player, const RLGC::GameState& state, bool deterministic, float temperature = 1, InferDebug* debugOut = nullptr);
		std::vector<RLGC::Action> BatchInferActions(const std::vector<RLGC::Player>& players, const std::vector<RLGC::GameState>& states, bool deterministic, float temperature = 1, std::vector<InferDebug>* debugOut = nullptr);

		~InferUnit();
	};
}