#pragma once
#include "../Util/Models.h"

namespace GGL {

	// Deliberate-practice goal proposer: g_t = clamp(g_{t-1} + Delta(feat_t, g_{t-1})), goals living
	// in the SAME 6D canonical-ball space as the reachability BALL head. Trained by advantage-weighted
	// hindsight: targets are achieved ball states N steps ahead, weighted by whether the N-step
	// advantage A^(N) beats an aspiration percentile of the batch (CRR-binary weights). This is a
	// hierarchy over the LEARNING SIGNAL, never the policy — the proposed goal never becomes a
	// policy input; it only ever feeds advantage-level shaping (Stage 2) and practice drills (Stage 3).
	// Stage 1 (this module trained + unrolled every iteration) is passive: it never touches
	// rewards/advantages/returns/values on its own.
	class ProposerModule {
	public:
		ProposerConfig config;
		torch::Device device;

		// Owned by the PPOLearner's ModelSet (saved/loaded with everything else). groupStepExempt is
		// set so the PPO minibatch loop's models.StepOptims() never steps it — Train() steps it itself.
		Model* delta;

		ProposerModule(int trunkOutSize, const ProposerConfig& config, torch::Device device, ModelSet& outModels,
			const char* modelName = "proposer_delta");

		// Detached (no-grad) fp32 trunk features for every row, chunked like Reachability's EvalRho.
		// Returns [n, trunkOutSize] on `device`. Never differentiable -> the proposer's own gradients
		// can never reach shared_head/policy/critic.
		torch::Tensor ComputeFeatures(Model* sharedHead, torch::Tensor obs, int64_t chunkSize);

		// Episode boundaries from a terminals column (combinedTraj = complete episodes back-to-back;
		// a nonzero terminal marks the LAST row of its episode - same invariant the reachability gate
		// relies on). outEpStart[e]/outEpEnd[e] are the first/last row index of episode e.
		static void SegmentEpisodes(
			const std::vector<int8_t>& terminals,
			std::vector<int64_t>& outEpStart, std::vector<int64_t>& outEpEnd);

		struct UnrollResult {
			torch::Tensor goals;     // [n,6] CPU float32 - g_t per row
			torch::Tensor prevGoals; // [n,6] CPU float32 - g_{t-1} used to produce g_t (regression anchor)
			float meanDeltaNorm = 0; // mean ||Delta|| per step, for logging
		};

		// No-BPTT (semi-gradient) unroll: g_{-1} = curBall[episode start] (i.e. "propose staying put"
		// as the neutral start), then g_t = clamp(g_{t-1} + Delta(feat_t, g_{t-1})) walking forward
		// through each episode. Ragged-batched by local step (one batched Delta forward per step index,
		// over whichever episodes are still alive), so cost is one forward per timestep of the LONGEST
		// episode, not per row. Always no-grad; Train() re-derives gradients from prevGoals separately.
		UnrollResult Unroll(
			torch::Tensor features, torch::Tensor curBall,
			const std::vector<int64_t>& epStart, const std::vector<int64_t>& epEnd);

		// N-step advantage, in the SAME standardized/clipped units GAE trains the critic against
		// (rHat = clip(r/returnStd, +-clipRange) when returnStd != 0, else raw r), so it's comparable
		// to V. Window clamps at episode end; bootstraps gamma^N * V[t+N] when the window stays inside
		// the episode, gamma^K * (truncated ? truncVal : 0) when it runs off the end (K = steps actually
		// summed). truncValPreds is consumed front-to-back, one per truncated episode IN ORDER OF
		// APPEARANCE (chronological) - the same population GAE consumes back-to-front over its reverse
		// scan, so both land on the same trajectory's bootstrap value.
		static torch::Tensor ComputeNStepAdvantages(
			torch::Tensor rews /* post-gate, CPU */, const std::vector<int8_t>& terminals,
			torch::Tensor valPreds /* CPU */, torch::Tensor truncValPreds /* CPU, may be undefined */,
			const std::vector<int64_t>& epStart, const std::vector<int64_t>& epEnd,
			int horizonSteps, float gamma, float returnStd, float clipRange);

		struct TrainResult {
			float loss = 0;
		};

		// Semi-gradient regression: prevGoals is treated as a constant (no grad flows through it, it
		// was produced by Unroll under RG_NO_GRAD); only Delta's own parameters are trained. Ends every
		// minibatch with delta->StepOptim() (grads are left as none afterward, so the PPO loop's
		// later models.StepOptims() call is a no-op on this model even without groupStepExempt).
		TrainResult Train(
			torch::Tensor features /* [n,trunk] on device, detached */,
			torch::Tensor prevGoals /* [n,6] CPU */,
			torch::Tensor targets /* [n,6] CPU */,
			torch::Tensor weights /* [n] CPU, CRR-binary aspiration weights */);
	};
}
