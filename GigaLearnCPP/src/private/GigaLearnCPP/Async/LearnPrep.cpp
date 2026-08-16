#include "LearnPrep.h"
#include <private/GigaLearnCPP/PPO/PPOLearner.h>
#include <private/GigaLearnCPP/PPO/GAE.h>
#include <private/GigaLearnCPP/Util/WelfordStat.h>
#include <GigaLearnCPP/Util/Timer.h>
#include <RLGymCPP/TerminalConditions/TerminalCondition.h>

#ifdef RG_CUDA_SUPPORT
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <cmath>

using namespace RLGC;

namespace GGL {

void PrepareExperienceBatched(
	PPOLearner* ppo,
	const LearnerConfig& config,
	std::vector<TrajectoryFragment>& frags,
	ExperienceTensors& out,
	Report& report,
	WelfordStat* returnStat,
	Dist::Session* dist,
	int obsSize,
	const std::function<void()>& onChunkDrain) {

	std::vector<torch::Tensor> st, ac, lp, am, pv, rew;
	std::vector<torch::Tensor> nxt, goalRewParts;
	bool anyGoalRews = false;
	for (auto& f : frags) {
		st.push_back(f.states);
		ac.push_back(f.actions);
		lp.push_back(f.logProbs);
		am.push_back(f.actionMasks);
		pv.push_back(f.policy_version);
		rew.push_back(f.rewards);
		if (f.nextStates.defined() && f.nextStates.size(0) > 0)
			nxt.push_back(f.nextStates);
		if (f.goalRews.defined() && f.goalRews.numel() > 0) {
			goalRewParts.push_back(f.goalRews);
			anyGoalRews = true;
		}
	}
	auto tStates = torch::cat(st, 0);
	auto tActions = torch::cat(ac, 0).to(torch::kInt64);
	auto tLogProbs = torch::cat(lp, 0);
	auto tActionMasks = torch::cat(am, 0);
	auto tRewards = torch::cat(rew, 0);
	float stepRew = tRewards.numel() > 0 ? tRewards.mean().item<float>() : 0.f;
	if (dist)
		dist->avg_host(&stepRew, 1);
	report["Average Step Reward"] = stepRew;
	auto tPolicyVer = torch::cat(pv, 0);
	torch::Tensor tNextTrunc;
	if (!nxt.empty())
		tNextTrunc = torch::cat(nxt, 0);

	const bool goalOn = config.ppo.goalCritic.enabled && anyGoalRews;
	const int64_t n = tStates.size(0);
	(void)obsSize;

	auto ensureCap1d = [](torch::Tensor& buf, int64_t minElems, torch::TensorOptions opts) {
		if (!buf.defined() || buf.numel() < minElems || buf.dtype() != opts.dtype() || buf.device() != opts.device())
			buf = torch::empty({ minElems }, opts);
		return buf.narrow(0, 0, minElems);
	};
	auto ensure2d = [](torch::Tensor& buf, int64_t rows, int64_t cols, torch::TensorOptions opts) {
		if (!buf.defined() || buf.size(0) < rows || buf.size(1) != cols
			|| buf.dtype() != opts.dtype() || buf.device() != opts.device())
			buf = torch::empty({ rows, cols }, opts);
		return buf.narrow(0, 0, rows);
	};

	thread_local static torch::Tensor gpu_states, gpu_masks;
	thread_local static torch::Tensor pinned_val_preds, pinned_goal_preds;
	thread_local static torch::Tensor pinned_trunc_vals, pinned_goal_trunc_vals;

	torch::Tensor tDeviceStates = tStates;
	torch::Tensor tDeviceMasks = tActionMasks;
	if (ppo->device.is_cuda()) {
		tDeviceStates = ensure2d(gpu_states, n, tStates.size(1),
			torch::TensorOptions().dtype(torch::kFloat32).device(ppo->device));
		tDeviceStates.copy_(tStates, /*non_blocking=*/true);
		tDeviceMasks = ensure2d(gpu_masks, n, tActionMasks.size(1),
			torch::TensorOptions().dtype(tActionMasks.dtype()).device(ppo->device));
		tDeviceMasks.copy_(tActionMasks, /*non_blocking=*/true);
	}

	torch::Tensor tValPreds, tGoalValPreds;
	torch::Tensor tTruncValPreds, tGoalTruncValPreds;
	{
		RG_NO_GRAD;
		Timer valTimer = {};
		auto pinOpts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU).pinned_memory(true);
		tValPreds = ensureCap1d(pinned_val_preds, n, pinOpts);
		if (goalOn)
			tGoalValPreds = ensureCap1d(pinned_goal_preds, n, pinOpts);
		torch::Tensor vCrit, vGoal;
		// Async GAE does not consume vdag/geo outputs; skip those heads (was a full extra
		// V-dagger pair on 160k rows, discarded).
		ppo->InferValueFamily(
			tDeviceStates, &vCrit, goalOn ? &vGoal : nullptr, nullptr, nullptr);
		tValPreds.copy_(vCrit, /*non_blocking=*/true);
		if (goalOn && vGoal.defined())
			tGoalValPreds.copy_(vGoal, /*non_blocking=*/true);
		if (tNextTrunc.defined() && tNextTrunc.size(0) > 0) {
			int64_t nt = tNextTrunc.size(0);
			tTruncValPreds = ensureCap1d(pinned_trunc_vals, nt, pinOpts);
			if (goalOn)
				tGoalTruncValPreds = ensureCap1d(pinned_goal_trunc_vals, nt, pinOpts);
			auto g = tNextTrunc.to(ppo->device, /*non_blocking=*/true);
			torch::Tensor vT, vG;
			ppo->InferValueFamily(g, &vT, goalOn ? &vG : nullptr, nullptr, nullptr);
			tTruncValPreds.copy_(vT, /*non_blocking=*/true);
			if (goalOn && vG.defined())
				tGoalTruncValPreds.copy_(vG, /*non_blocking=*/true);
		}
#ifdef RG_CUDA_SUPPORT
		if (ppo->device.is_cuda()) {
			cudaEvent_t ev = nullptr;
			cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
			cudaEventRecord(ev, c10::cuda::getCurrentCUDAStream(ppo->device.index()).stream());
			while (cudaEventQuery(ev) == cudaErrorNotReady) {
				if (onChunkDrain)
					onChunkDrain();
			}
			cudaEventSynchronize(ev);
			cudaEventDestroy(ev);
		} else if (onChunkDrain) {
			onChunkDrain();
		}
#else
		if (onChunkDrain)
			onChunkDrain();
#endif
		report["Value Pred Time"] = valTimer.Elapsed();
	}

	std::vector<torch::Tensor> advParts, tgtParts, retParts, goalAdvParts, goalTgtParts;
	int64_t rowOff = 0;
	int64_t truncOff = 0;
	double clipAcc = 0;
	int64_t clipRows = 0;
	const float retStd = returnStat ? (float)returnStat->GetSTD() : 1.f;
	Timer gaeTimer = {};
	int gaeIdx = 0;
	for (auto& f : frags) {
		if (onChunkDrain && ((gaeIdx++) & 15) == 0)
			onChunkDrain();
		const int64_t fn = f.rows();
		auto vp = tValPreds.narrow(0, rowOff, fn);
		const int64_t ntr = (f.nextStates.defined() && f.nextStates.size(0) > 0)
			? f.nextStates.size(0) : 0;
		torch::Tensor tvp;
		if (ntr > 0 && tTruncValPreds.defined()
			&& truncOff + ntr <= tTruncValPreds.size(0)) {
			tvp = tTruncValPreds.narrow(0, truncOff, ntr);
		}
		torch::Tensor a, tv, r;
		float clipP = 0.f;
		GAE::Compute(
			f.rewards, f.terminals, vp, tvp,
			a, tv, r, clipP,
			config.ppo.gaeGamma, config.ppo.gaeLambda,
			retStd, config.ppo.rewardClipRange);
		advParts.push_back(std::move(a));
		tgtParts.push_back(std::move(tv));
		retParts.push_back(std::move(r));
		clipAcc += (double)clipP * (double)fn;
		clipRows += fn;

		if (goalOn && f.goalRews.defined() && tGoalValPreds.defined()) {
			torch::Tensor gtvp;
			if (ntr > 0 && tGoalTruncValPreds.defined()
				&& truncOff + ntr <= tGoalTruncValPreds.size(0)) {
				gtvp = tGoalTruncValPreds.narrow(0, truncOff, ntr);
			}
			torch::Tensor ga, gtv, gr;
			float gclip = 0.f;
			GAE::Compute(
				f.goalRews, f.terminals, tGoalValPreds.narrow(0, rowOff, fn), gtvp,
				ga, gtv, gr, gclip,
				config.ppo.goalCritic.gamma, config.ppo.gaeLambda, 0.f, 0.f);
			goalAdvParts.push_back(std::move(ga));
			goalTgtParts.push_back(std::move(gtv));
		}

		rowOff += fn;
		truncOff += ntr;
	}
	report["GAE Time"] = gaeTimer.Elapsed();

	auto tAdvantages = torch::cat(advParts, 0);
	auto tTargetVals = torch::cat(tgtParts, 0);
	auto tReturns = torch::cat(retParts, 0);
	float rewClipPortion = clipRows > 0 ? (float)(clipAcc / (double)clipRows) : 0.f;
	report["GAE/Reward Clip Portion"] = rewClipPortion;
	report["GAE/Avg Advantage"] = tAdvantages.abs().mean().item<float>();
	float rawAdvAbsMean = tAdvantages.abs().mean().item<float>();

	if (returnStat) {
		auto cpuR = tReturns.contiguous().to(torch::kCPU).flatten();
		FList tmp(cpuR.data_ptr<float>(), cpuR.data_ptr<float>() + cpuR.numel());
		returnStat->Increment(tmp);
		returnStat->SyncAcrossRanks(dist);
	}

	torch::Tensor tGoalTargetVals;
	if (goalOn && !goalAdvParts.empty()) {
		auto tGoalAdvantages = torch::cat(goalAdvParts, 0);
		tGoalTargetVals = torch::cat(goalTgtParts, 0);
		float advStd = tAdvantages.std().item<float>();
		float goalAdvStd = tGoalAdvantages.std().item<float>();
		if (config.ppo.goalCritic.beta > 0 && goalAdvStd > 1e-8f && advStd > 1e-8f) {
			float betaEff = config.ppo.goalCritic.beta * advStd / goalAdvStd;
			torch::Tensor injected = betaEff * (tGoalAdvantages - tGoalAdvantages.mean());
			tAdvantages = tAdvantages + injected;
			report["GoalCritic/Blend BetaEff"] = betaEff;
			report["GoalCritic/Injected Abs Mean"] = injected.abs().mean().item<float>();
		}
	}

	{
		float postAdvAbsMean = tAdvantages.abs().mean().item<float>();
		report["GAE/Avg Advantage Post-Inj"] = postAdvAbsMean;
		if (rawAdvAbsMean > 1e-8f)
			report["GAE/Injected Frac"] = (postAdvAbsMean - rawAdvAbsMean) / rawAdvAbsMean;
	}

	torch::Tensor tAdvFilterMask;
	if (config.ppo.advFilterFrac < 1.f && n > 1) {
		RG_NO_GRAD;
		float frac = config.ppo.advFilterFrac;
		if (frac < 0.01f) frac = 0.01f;
		if (frac > 1.f) frac = 1.f;
		auto advF = tAdvantages.to(torch::kFloat32).flatten();
		bool byMagnitude = config.ppo.advFilterMode == AdvFilterMode::MAGNITUDE;
		auto advKey = byMagnitude ? advF.abs() : advF;
		float thresh = advKey.quantile(1.f - frac).item<float>();
		auto keep = (advKey >= thresh).to(torch::kFloat32);
		tAdvFilterMask = keep;
		report["AdvFilter/Kept Frac"] = keep.sum().item<float>() / (float)advF.numel();
	}

	out = {};
	out.states = tDeviceStates;
	out.actions = tActions;
	out.logProbs = tLogProbs;
	out.targetValues = tTargetVals;
	out.actionMasks = tDeviceMasks;
	out.advantages = tAdvantages;
	if (tGoalTargetVals.defined())
		out.goalTargetValues = tGoalTargetVals;
	if (tAdvFilterMask.defined())
		out.advFilterMask = tAdvFilterMask;

	const int64_t learnVer = (int64_t)ppo->policyVersion;
	auto pvCpu = tPolicyVer.to(torch::kCPU).contiguous();
	const int32_t* pver = pvCpu.data_ptr<int32_t>();
	int64_t vdMin = 1 << 30, vdMax = 0;
	double vdSum = 0;
	int64_t nFresh = 0;
	int64_t nMuChange = 0;
	int64_t muOff = 0;
	double pullSum = 0, spanSum = 0;
	int64_t nFrags = 0;
	int64_t pullMax = 0;
	for (auto& f : frags) {
		const int64_t fn = f.rows();
		int32_t first = (fn > 0) ? pver[muOff] : 0;
		int64_t pulls = 0;
		for (int64_t i = 0; i < fn; i++) {
			int64_t lag = learnVer - (int64_t)(uint32_t)pver[muOff + i];
			if (lag < vdMin) vdMin = lag;
			if (lag > vdMax) vdMax = lag;
			vdSum += (double)lag;
			if (lag <= 1) nFresh++;
			if (pver[muOff + i] != first) nMuChange++;
			if (i > 0 && pver[muOff + i] != pver[muOff + i - 1])
				pulls++;
		}
		int64_t span = 0;
		if (fn > 0)
			span = (int64_t)f.hdr.max_policy_version - (int64_t)f.hdr.min_policy_version;
		if (span < 0) span = 0;
		pullSum += (double)pulls;
		spanSum += (double)span;
		if (pulls > pullMax) pullMax = pulls;
		nFrags++;
		muOff += fn;
		if (onChunkDrain && (muOff & 16383) == 0)
			onChunkDrain();
	}
	if (n > 0) {
		report["Async/VersionDiff Min"] = (float)vdMin;
		report["Async/VersionDiff Avg"] = (float)(vdSum / (double)n);
		report["Async/VersionDiff Max"] = (float)vdMax;
		report["Async/FreshFrac"] = (float)nFresh / (float)n;
		report["Async/MuChangeFrac"] = (float)nMuChange / (float)n;
	}
	float pullsAvg = nFrags > 0 ? (float)(pullSum / (double)nFrags) : 0.f;
	float spanAvg = nFrags > 0 ? (float)(spanSum / (double)nFrags) : 0.f;
	int pullMaxI = (int)pullMax;
	if (dist) {
		dist->avg_host(&pullsAvg, 1);
		dist->avg_host(&spanAvg, 1);
		dist->max_host(&pullMaxI, 1);
	}
	report["Async/MidTraj Pulls Avg"] = pullsAvg;
	report["Async/MidTraj Pulls Max"] = (float)pullMaxI;
	report["Async/MidTraj Span Avg"] = spanAvg;
}

}
