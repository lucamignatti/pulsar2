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

		// INTENT CLASS (research/reports/NATIVE_INTENT_PROTOCOL.md). When intentBiasRows > 0
		// the model carries an extra registered parameter `intent_bias` of shape
		// [rows, numOutputs, 1] (3-D on purpose: Muon orthogonalizes 2-D params and a 6x90
		// action-preference table is not a matrix to orthogonalize; 3-D falls to Muon's Adam
		// path). Init N(0, intentBiasStd). Saved/loaded as <NAME>_INTENT_BIAS.lt; absent on
		// old checkpoints -> kept at init (a zero std makes it exactly inert). The caller
		// (InferPolicyProbsFromModels) adds row z to the logits before masking.
		int intentBiasRows = 0;
		float intentBiasStd = 0.f;

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