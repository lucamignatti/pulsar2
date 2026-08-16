#pragma once

#include "../Framework.h"

namespace GGL {
	// ADAM is the DEFAULT (first member) - the reachability heads rely on that rather than
	// naming it, because contrastive InfoNCE embeddings train poorly under orthogonalized
	// updates. MUON is what every dense net asks for explicitly.
	// ADAMW / ADAGRAD / RMSPROP / MAGSGD removed 2026-07-25: zero users, and MagSGD carried a
	// whole implementation file. Safe to renumber - optimType is never serialized, it is
	// rebuilt from ExampleMain on every boot.
	enum class ModelOptimType {
		ADAM,
		MUON
	};

	enum class ModelActivationType {
		RELU,
		LEAKY_RELU,
		SIGMOID,
		TANH
	};

	// Doesn't include inputs or outputs
	struct PartialModelConfig {
		std::vector<int> layerSizes = {};
		ModelActivationType activationType = ModelActivationType::RELU;
		ModelOptimType optimType = ModelOptimType::ADAM;
		bool addLayerNorm = true;
		bool addOutputLayer = true;

		// Residual (BroNet-style) blocks. When true, layerSizes[0] is a plain "stem" layer
		// and every following PAIR of layers becomes a residual block:
		//     x <- Act( x + LN(W2 Act(LN(W1 x))) )
		// A trailing unpaired layer stays plain, so {W} and {W,W} are unchanged by this flag
		// and {W,W,W} = stem + 1 block, {W,W,W,W,W} = stem + 2 blocks.
		// Requires uniform layerSizes (the skip needs matching dims) - asserted in Model().
		// The module list stays FLAT (Linear/LayerNorm/Act in order, exactly as before); the
		// skips are applied by Model::Forward from recorded index spans, so every consumer
		// that walks seq looking for Linears (PSD::LinearLayers, PolicySlots) still works.
		bool addResiduals = false;

		// DeepSeek-V3-style MoE trunk (research/reports/MOE_POLICY.md). When moeBlocks > 0
		// the model is: Linear(numInputs -> layerSizes[0]) embed + LN + Act, then
		// moeBlocks x MoEBlock(dim=layerSizes[0], moeHidden, moeExperts, moeTopK)
		// (each block is pre-LN residual internally), then LN, then the output layer.
		// layerSizes must have exactly one entry (the trunk width). Incompatible with
		// addResiduals (blocks carry their own skips).
		int moeBlocks = 0;
		int moeExperts = 0;
		int moeTopK = 0;
		int moeHidden = 0;

		bool IsValid() const {
			return !layerSizes.empty()
				&& (moeBlocks == 0 || (moeExperts > 0 && moeTopK > 0 && moeHidden > 0
					&& layerSizes.size() == 1 && !addResiduals));
		}
	};

	struct ModelConfig : PartialModelConfig {
		int numInputs = -1;
		int numOutputs = -1;

		bool IsValid() const {
			return PartialModelConfig::IsValid() && numInputs > 0 && (numOutputs > 0 || !addOutputLayer);
		}

		ModelConfig(const PartialModelConfig& partialConfig) : PartialModelConfig(partialConfig) {}
	};
}