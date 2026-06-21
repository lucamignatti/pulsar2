#pragma once
#include "../FrameworkTorch.h"
#include <GigaLearnCPP/Util/Utils.h>
#include <nlohmann/json.hpp>

namespace GGL {

	// SimBa-canonical RSNorm: a per-dimension running-observation standardizer.
	//
	//   o_norm = (o - mu) / sqrt(var + eps)
	//
	// mu/var are STANDARD count-based running statistics (OpenAI-baselines
	// RunningMeanStd): unbounded count, no EMA/momentum/cap. var = M2 / n
	// (population). No learnable params; Normalize() is a stop-gradient constant
	// affine transform of the input (mu/var are plain buffers, never in the graph).
	//
	// Discipline (enforced by the caller, not this class): Update() ONCE per
	// rollout from the freshly collected observations, BEFORE the K optimization
	// epochs; the stats are then frozen for the whole epoch pass. (mu, var, n) are
	// part of the policy state -- persist them in the checkpoint WITH the weights.
	//
	// Running stats are kept in double on CPU for precision over billions of steps;
	// the per-rollout batch reduction runs on the input's device in its own dtype
	// (so no float64 is required on the GPU/MPS), and only the [width] result is
	// moved to CPU double for accumulation. Normalize() uses a cached float
	// (mean, 1/sqrt(var+eps)) refreshed whenever the stats change.
	struct RSNorm {
		int width;
		double eps;
		double clipRange; // optional clamp of normalized obs; <= 0 disables (NOT part of SimBa)
		double count;

		torch::Tensor runningMean, runningM2; // [width] double, CPU
		torch::Tensor cacheMean, cacheInvStd; // [width] float, CPU (broadcast in Normalize)

		RSNorm(int width, double eps = 1e-8, double initVar = 1.0, double initCount = 1e-4, double clipRange = 0.0)
			: width(width), eps(eps), clipRange(clipRange), count(initCount) {
			// Init: mu = 0, var = 1 (M2 = initVar * n), small n so early steps are
			// well-defined with no warmup.
			runningMean = torch::zeros({ width }, torch::kFloat64);
			runningM2 = torch::full({ width }, initVar * initCount, torch::kFloat64);
			RefreshCache();
		}

		// Batched Welford merge (parallel-algorithm form; numerically stable, not
		// sum(x^2)/n - mu^2). batch: [N, width], any dtype/device.
		void Update(const torch::Tensor& batch) {
			RG_NO_GRAD;
			if (!batch.defined())
				return;
			if (batch.dim() != 2 || batch.size(1) != width)
				RG_ERR_CLOSE("RSNorm::Update expects [N, " << width << "], got shape with dim " << batch.dim());

			double nB = (double)batch.size(0);
			if (nB <= 0)
				return;

			// Per-batch reduction on the input device/dtype, then accumulate in double.
			torch::Tensor x = batch.detach();
			torch::Tensor muB_dev = x.mean(0);
			torch::Tensor M2B_dev = (x - muB_dev).pow(2).sum(0);

			torch::Tensor muB = muB_dev.to(torch::kCPU, torch::kFloat64);
			torch::Tensor M2B = M2B_dev.to(torch::kCPU, torch::kFloat64);

			torch::Tensor delta = muB - runningMean;
			double nNew = count + nB;
			runningMean = runningMean + delta * (nB / nNew);
			runningM2 = runningM2 + M2B + delta.pow(2) * (count * nB / nNew);
			count = nNew;

			RefreshCache();
		}

		// (o - mu) / sqrt(var + eps), stop-gradient. obs: [..., width].
		torch::Tensor Normalize(const torch::Tensor& obs) const {
			torch::Tensor m = cacheMean.to(obs.device(), obs.scalar_type());
			torch::Tensor s = cacheInvStd.to(obs.device(), obs.scalar_type());
			torch::Tensor out = (obs - m) * s;
			if (clipRange > 0)
				out = out.clamp(-clipRange, clipRange);
			return out;
		}

		void RefreshCache() {
			torch::Tensor var = runningM2 / std::max(count, 1e-12);
			cacheMean = runningMean.to(torch::kFloat32).clone();
			cacheInvStd = (1.0 / (var + eps).sqrt()).to(torch::kFloat32).clone();
		}

		// Summary scalars for logging.
		double GetMeanMu() const { return runningMean.mean().item<double>(); }
		double GetMeanVar() const { return (runningM2 / std::max(count, 1e-12)).mean().item<double>(); }
		double GetCount() const { return count; }

		nlohmann::json ToJSON() const {
			torch::Tensor mc = runningMean.contiguous();
			torch::Tensor vc = runningM2.contiguous();
			std::vector<double> meanV(mc.data_ptr<double>(), mc.data_ptr<double>() + mc.numel());
			std::vector<double> m2V(vc.data_ptr<double>(), vc.data_ptr<double>() + vc.numel());

			nlohmann::json result = {};
			result["mean"] = Utils::MakeJSONArray<double>(meanV);
			result["m2"] = Utils::MakeJSONArray<double>(m2V);
			result["count"] = count;
			result["width"] = width;
			return result;
		}

		void ReadFromJSON(const nlohmann::json& json) {
			std::vector<double> meanV = Utils::MakeVecFromJSON<double>(json["mean"]);
			std::vector<double> m2V = Utils::MakeVecFromJSON<double>(json["m2"]);
			width = (int)meanV.size();
			runningMean = torch::tensor(meanV, torch::kFloat64);
			runningM2 = torch::tensor(m2V, torch::kFloat64);
			count = json["count"].get<double>();
			RefreshCache();
		}
	};
}
