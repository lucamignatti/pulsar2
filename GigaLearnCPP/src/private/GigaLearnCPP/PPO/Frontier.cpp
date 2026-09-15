#include "Frontier.h"

namespace GGL {

	FrontierModule::FrontierModule(int obsSize, const FrontierConfig& config, torch::Device device, ModelSet& outModels)
		: config(config), device(device), obsSize(obsSize) {

		// The bound is not optional. Without it the toy's value map ran 0.25 -> 38,187 in 32
		// sweeps; twin nets trained on identical targets are too correlated for their minimum to
		// act as pessimism. Fail loudly at boot rather than diverge silently 3 hours in.
		if (!(config.valueAbsMax > 0))
			RG_ERR_CLOSE("FrontierConfig::valueAbsMax must be > 0 (bounded bootstrap target is required)");
		if (!(config.valueExpectile > 0.5f && config.valueExpectile < 1.0f))
			RG_ERR_CLOSE("FrontierConfig::valueExpectile must be in (0.5, 1) - 0.5 is a mean, which is the wrong question");
		if (config.bandLowDecisions <= 0 || config.bandHighDecisions <= config.bandLowDecisions)
			RG_ERR_CLOSE("FrontierConfig band must satisfy 0 < bandLowDecisions < bandHighDecisions");
		if (config.goalTtl < config.silMinWindow)
			RG_ERR_CLOSE("FrontierConfig::goalTtl (" << config.goalTtl << ") < silMinWindow ("
				<< config.silMinWindow << "): every window would be rejected and SIL would be inert");

		ModelConfig valueConfig = config.value;
		valueConfig.numInputs = obsSize;
		valueConfig.numOutputs = 1;
		valueConfig.addOutputLayer = true;

		ModelConfig quasiConfig = config.quasi;
		quasiConfig.numInputs = obsSize;
		quasiConfig.numOutputs = config.latentAsym + config.latentSym;
		quasiConfig.addOutputLayer = true;

		valueA = new Model("frontier_value_a", valueConfig, device);
		valueB = new Model("frontier_value_b", valueConfig, device);
		quasi = new Model("frontier_quasi", quasiConfig, device);

		outModels.Add(valueA);
		outModels.Add(valueB);
		outModels.Add(quasi);
	}

	torch::Tensor FrontierModule::Value(torch::Tensor obs) {
		torch::NoGradGuard noGrad;
		auto a = valueA->Forward(obs, false).squeeze(-1);
		auto b = valueB->Forward(obs, false).squeeze(-1);
		return torch::minimum(a, b);
	}

	// Expectile regression: asymmetric squared loss. tau > 0.5 weights positive residuals more, so
	// the fit tracks the upper part of the target distribution - a soft max over whatever actions
	// the behaviour actually took, without ever querying an action nobody executed.
	static torch::Tensor ExpectileLoss(torch::Tensor pred, torch::Tensor target, float tau) {
		auto diff = target - pred;
		auto weight = torch::abs(torch::full_like(diff, tau) - (diff < 0).to(diff.dtype()));
		return (weight * diff.pow(2)).mean();
	}

	FrontierModule::ValueStats FrontierModule::TrainValue(
		torch::Tensor obs, torch::Tensor nextObs, torch::Tensor reward, torch::Tensor done) {

		ValueStats stats = {};
		const float bound = config.valueAbsMax;

		torch::Tensor target;
		{
			torch::NoGradGuard noGrad;
			auto nextV = torch::minimum(
				valueA->Forward(nextObs, false).squeeze(-1),
				valueB->Forward(nextObs, false).squeeze(-1)
			).clamp(-bound, bound);                       // bounded BEFORE it is bootstrapped
			auto raw = reward + config.gamma * (1.0f - done) * nextV;
			stats.maxAbsTarget = raw.abs().max().item<float>();
			stats.clamped = (raw.abs() > bound).sum().item<int>();
			target = raw.clamp(-bound, bound);            // and bounded again after
		}

		auto predA = valueA->Forward(obs, false).squeeze(-1);
		auto predB = valueB->Forward(obs, false).squeeze(-1);
		auto loss = ExpectileLoss(predA, target, config.valueExpectile)
		          + ExpectileLoss(predB, target, config.valueExpectile);
		loss.backward();

		stats.loss = loss.item<float>();
		stats.meanValue = torch::minimum(predA, predB).mean().item<float>();
		return stats;
	}

	torch::Tensor FrontierModule::Encode(torch::Tensor obs) {
		return quasi->Forward(obs, false);
	}

	torch::Tensor FrontierModule::Distance(torch::Tensor fromLatent, torch::Tensor toLatent) {
		const int k = config.latentAsym;
		// Asymmetric part: zero when every coordinate of `to` is already below `from`, positive by
		// the largest coordinate that must rise. This is what makes leaving a manoeuvre cheap and
		// re-entering it expensive, which a symmetric metric cannot express at all.
		auto asym = torch::relu(toLatent.narrow(-1, 0, k) - fromLatent.narrow(-1, 0, k)).amax(-1);
		auto sym = (toLatent.narrow(-1, k, config.latentSym) - fromLatent.narrow(-1, k, config.latentSym)).norm(2, -1);
		return asym + sym;
	}

	FrontierModule::QuasiStats FrontierModule::TrainQuasi(
		torch::Tensor obs, torch::Tensor nextObs, torch::Tensor pairObs, torch::Tensor done) {

		QuasiStats stats = {};

		auto zFrom = Encode(obs);
		auto zNext = Encode(nextObs);
		auto zPair = Encode(pairObs);

		// Local constraint: one real decision costs at most 1. Terminal rows are teleports
		// (their stored successor is a kickoff) and carry no constraint.
		auto keep = (1.0f - done.to(torch::kFloat32).flatten());
		auto nKeep = keep.sum().clamp_min(1.0f);
		auto local = Distance(zFrom, zNext);
		auto violation = (torch::relu(local - 1.0f).pow(2) * keep).sum() / nKeep;

		// Spread: push everything else apart, softly capped so it cannot run away.
		const float cap = config.quasiSpreadCap;
		auto spread = Distance(zFrom, zPair);
		auto spreadTerm = (cap * torch::tanh(spread / cap)).mean() / cap;

		auto loss = -spreadTerm + config.quasiConstraintWeight * violation;
		loss.backward();

		stats.loss = loss.item<float>();
		stats.meanLocal = ((local * keep).sum() / nKeep).item<float>();
		localDEma = 0.99f * localDEma + 0.01f * stats.meanLocal;
		stats.violation = violation.item<float>();
		stats.meanSpread = spread.mean().item<float>();
		return stats;
	}

	FrontierModule::GoalPick FrontierModule::SelectGoals(torch::Tensor obs, torch::Tensor candidates,
		float bandLo, float bandHi, int mode) {
		torch::NoGradGuard noGrad;
		GoalPick pick = {};

		const int n = obs.size(0);
		const int m = candidates.size(0);

		auto zObs = Encode(obs);                    // [n, latent]
		auto zCand = Encode(candidates);            // [m, latent]
		auto vObs = Value(obs);                     // [n]
		auto vCand = Value(candidates);             // [m]

		// [n, m] distances and value gains
		auto dist = Distance(zObs.unsqueeze(1).expand({ n, m, zObs.size(-1) }),
		                     zCand.unsqueeze(0).expand({ n, m, zCand.size(-1) }));
		auto gain = vCand.unsqueeze(0) - vObs.unsqueeze(1);

		auto inBand = (dist >= bandLo) & (dist <= bandHi);

		// Rarity: mean distance from the CURRENT state population to each candidate. Large means
		// the policy rarely gets near it. Free - it is a reduction of the matrix already built.
		auto rarity = dist.mean(0);                 // [m]

		torch::Tensor pref;
		switch (mode) {
			case FrontierConfig::GOALSEL_RANDOM:
				pref = torch::rand_like(gain);
				break;
			case FrontierConfig::GOALSEL_RARITY:
				pref = rarity.unsqueeze(0).expand_as(gain);
				break;
			default:
				pref = gain;                        // GOALSEL_VALUE
				break;
		}
		auto score = torch::where(inBand, pref, torch::full_like(pref, -1e18f));
		auto best = score.argmax(1);

		pick.valid = inBand.any(1);
		pick.goals = candidates.index_select(0, best);

		auto rows = torch::arange(n, torch::TensorOptions().dtype(torch::kLong).device(device));
		auto chosenDist = dist.index({ rows, best });
		auto chosenGain = gain.index({ rows, best });
		auto validF = pick.valid.to(torch::kFloat32);
		auto denom = validF.sum().clamp_min(1.0f);
		pick.meanDist = (chosenDist * validF).sum().item<float>() / denom.item<float>();
		pick.meanGain = (chosenGain * validF).sum().item<float>() / denom.item<float>();
		// Rarity of what was chosen, relative to the pool's mean rarity: > 1 means the goals are
		// less-visited than a random candidate. Under GOALSEL_VALUE this reads ~1 or below, which
		// is the diagnostic that the mechanism is exploiting rather than exploring.
		auto chosenRarity = rarity.index_select(0, best);
		float poolRarity = rarity.mean().item<float>();
		pick.meanRarity = poolRarity > 1e-6f
			? ((chosenRarity * validF).sum().item<float>() / denom.item<float>()) / poolRarity : 0.f;
		return pick;
	}

	FrontierModule::Progress FrontierModule::PrefixProgress(torch::Tensor prefixObs, torch::Tensor goal,
		torch::Tensor live) {
		torch::NoGradGuard noGrad;
		Progress out = {};

		// prefixObs: [n, T, obsSize]; goal: [n, obsSize]
		const int n = prefixObs.size(0);
		const int T = prefixObs.size(1);

		auto zPrefix = Encode(prefixObs.reshape({ n * T, obsSize })).reshape({ n, T, -1 });
		auto zGoal = Encode(goal).unsqueeze(1).expand({ n, T, -1 });
		auto dist = Distance(zPrefix, zGoal);                   // [n, T]
		if (live.defined())
			dist = torch::where(live, dist, torch::full_like(dist, 1e18f));

		auto dStart = dist.narrow(1, 0, 1).clamp_min(1e-6f);
		auto best = dist.min(1);
		auto dBest = std::get<0>(best);
		out.bestIndex = std::get<1>(best);

		auto progress = (1.0f - dBest / dStart.squeeze(1)).clamp(0.0f, 1.0f);
		out.weight = torch::exp(config.silSharpness * progress);
		// A prefix that never closed any distance contributes nothing.
		out.weight = torch::where(progress > 0, out.weight, torch::zeros_like(out.weight));
		return out;
	}
}
