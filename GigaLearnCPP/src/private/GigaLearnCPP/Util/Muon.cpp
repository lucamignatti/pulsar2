#include "Muon.h"
#include "MoE.h"
#include <torch/version.h>
#if TORCH_VERSION_MAJOR < 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR < 2)
#include <c10/util/C++17.h>
#endif

using namespace torch;

// Orthogonalize via the quintic Newton-Schulz iteration. The coefficients trade exact
// orthogonality for convergence speed: 5 iterations land every singular value near 1,
// which is all Muon needs (the update's direction structure, not its exact spectrum).
torch::Tensor GGL::Muon::NewtonSchulz5(torch::Tensor g) {
	constexpr float A = 3.4445f, B = -4.7750f, C = 2.0315f;
	constexpr int NUM_ITERS = 5;

	// Batched form: a 3D input [E, m, n] (MoE expert stacks) orthogonalizes every slice
	// in the same ~15 matmuls via bmm — per-slice Frobenius normalization, batched grams.
	// 2D inputs take the identical math with a leading batch of 1 semantics-free.
	const bool batched = g.dim() == 3;
	Tensor x = batched
		? g / (g.norm(2, { -2, -1 }, /*keepdim=*/true) + 1e-7f)
		: g / (g.norm() + 1e-7f);

	// Keep the gram matrix below at the smaller dimension
	bool transposed = false;
	if (x.size(-2) > x.size(-1)) {
		x = x.transpose(-2, -1);
		transposed = true;
	}

	for (int i = 0; i < NUM_ITERS; i++) {
		Tensor a = torch::matmul(x, x.transpose(-2, -1));
		Tensor b = a * B + torch::matmul(a, a) * C;
		x = x * A + torch::matmul(b, x);
	}

	if (transposed)
		x = x.transpose(-2, -1);

	return x;
}

torch::Tensor GGL::Muon::step(LossClosure closure) {
	Tensor loss = {};
	if (closure != nullptr) {
		at::AutoGradMode enableGrad(true);
		loss = closure();
	}

	RG_NO_GRAD;

	for (auto& group : param_groups()) {
		auto& options = static_cast<MuonOptions&>(group.options());
		double lr = options.lr();
		double momentum = options.momentum();
		bool nesterov = options.nesterov();

		for (auto& param : group.params()) {
			if (!param.grad().defined())
				continue;

			Tensor grad = param.grad();

			// dim 3 = MoE expert stacks [E, m, n]: same Muon update, batched NS, RMS-matched
			// lr from the per-slice matrix dims (identical for every slice).
			if ((grad.dim() == 2 && grad.size(0) > 1 && grad.size(1) > 1)
				|| (grad.dim() == 3 && grad.size(1) > 1 && grad.size(2) > 1)) {
				// Momentum buffer in SGD's own param state so it checkpoints with the model.
				// Libtorch 2.1 keys Optimizer::state_ by stringified TensorImpl*; 2.2+
				// (PR 108748) keys by the pointer itself.
#if TORCH_VERSION_MAJOR < 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR < 2)
				const auto stateKey = c10::guts::to_string(param.unsafeGetTensorImpl());
#else
				const auto stateKey = param.unsafeGetTensorImpl();
#endif
				auto stateItr = state_.find(stateKey);
				if (stateItr == state_.end()) {
					auto newState = std::make_unique<optim::SGDParamState>();
					newState->momentum_buffer(torch::zeros_like(param));
					stateItr = state_.insert({ stateKey, std::move(newState) }).first;
				}

				Tensor buf = static_cast<optim::SGDParamState&>(*stateItr->second).momentum_buffer();
				buf.mul_(momentum).add_(grad);

				// NS shard (see header): momentum above always runs; the expensive
				// orthogonalization + apply only on the owning rank.
				const bool nsOwned = shardWorld <= 1
					|| (shardCounter++ % (int64_t)shardWorld) == (int64_t)shardRank;
				if (!nsOwned)
					continue;

				Tensor update = nesterov ? grad.add(buf, momentum) : buf;
				// EP: under expert parallelism this rank's gradient is EXACT ZERO outside
				// its owned expert slice (the owner computed the whole thing locally), so
				// orthogonalizing the full [E,m,n] stack is ~nL x wasted work — at 1B/24
				// ranks that was 1.5s of a 4.9s learn. Slice to the owned range; the rest
                                // is reconciled by the post-step owner broadcast anyway.
				Tensor target = param;
				if (grad.dim() == 3 && GGL::MoEExpertParallelOn()) {
					bool isExpert = false;
					for (auto& ep : GGL::MoEExpertParams())
						if (ep.is_same(param)) { isExpert = true; break; }
					if (isExpert) {
						auto own = GGL::MoEOwnedExpertRange((int)param.size(0));
						if (own.second > 0 && own.second < param.size(0)) {
							update = update.narrow(0, own.first, own.second);
							target = param.narrow(0, own.first, own.second);
						}
					}
				}
				update = NewtonSchulz5(update);

				// The RMS match (see header): per-element step ~= lr, like Adam
				double adjustedLR = lr * std::sqrt((double)std::max(update.size(-2), update.size(-1)));
				target.add_(update, -adjustedLR);
			} else {
				// Non-matrix param: plain Adam at the group lr
				AdamState& adam = adamStates[param.unsafeGetTensorImpl()];
				if (!adam.expAvg.defined()) {
					adam.expAvg = torch::zeros_like(param);
					adam.expAvgSq = torch::zeros_like(param);
				}
				adam.stepCount++;

				constexpr double BETA1 = 0.9, BETA2 = 0.999, EPS = 1e-8;
				adam.expAvg.mul_(BETA1).add_(grad, 1 - BETA1);
				adam.expAvgSq.mul_(BETA2).addcmul_(grad, grad, 1 - BETA2);

				double biasCorr1 = 1 - std::pow(BETA1, (double)adam.stepCount);
				double biasCorr2 = 1 - std::pow(BETA2, (double)adam.stepCount);
				Tensor denom = (adam.expAvgSq / biasCorr2).sqrt_().add_(EPS);
				param.addcdiv_(adam.expAvg / biasCorr1, denom, -lr);
			}
		}
	}

	return loss;
}

void GGL::Muon::save(torch::serialize::OutputArchive& archive) const {
    torch::optim::SGD::save(archive);
    torch::serialize::OutputArchive adam;
    adam.write("version", torch::tensor(int64_t(2)));
    // Torch 2.1's SGD loader does not restore group options. Persist explicitly.
    std::vector<double> options;
    for (const auto& group : param_groups()) {
        const auto& o = static_cast<const MuonOptions&>(group.options());
        options.insert(options.end(), {o.lr(), o.momentum(), o.dampening(), o.weight_decay(), double(o.nesterov())});
    }
    adam.write("options", torch::tensor(options, torch::kFloat64).reshape({(int64_t)param_groups().size(), 5}));
    std::vector<int64_t> present;
    int64_t index = 0;
    for (const auto& group : param_groups()) for (const auto& p : group.params()) {
        auto it = adamStates.find(p.unsafeGetTensorImpl());
        present.push_back(it != adamStates.end());
        if (it != adamStates.end()) {
            torch::serialize::OutputArchive entry;
            entry.write("mean", it->second.expAvg);
            entry.write("square", it->second.expAvgSq);
            entry.write("step", torch::tensor(it->second.stepCount));
            adam.write(std::to_string(index), entry);
        }
        ++index;
    }
    adam.write("present", torch::tensor(present, torch::kInt64));
    archive.write("muon_adam", adam);
}

void GGL::Muon::load(torch::serialize::InputArchive& archive) {
    torch::optim::SGD::load(archive);
    torch::serialize::InputArchive adam;
    if (!archive.try_read("muon_adam", adam)) {
        adamStates.clear();
        loadedLegacyAdamState = true;
        RG_LOG("MUON_LEGACY_RESUME: archive has no fallback Adam moments; unavailable historical state starts fresh");
        return;
    }
    torch::Tensor version, present, options;
    adam.read("version", version);
    adam.read("present", present);
    TORCH_CHECK(version.scalar_type() == torch::kInt64 && version.numel() == 1
        && version.item<int64_t>() == 2, "Unsupported Muon Adam checkpoint version");
    adam.read("options", options);
    TORCH_CHECK(options.scalar_type() == torch::kFloat64 && options.dim() == 2
        && options.size(0) == param_groups().size() && options.size(1) == 5
        && torch::isfinite(options).all().item<bool>(), "Invalid Muon optimizer options");
    options = options.cpu();
    for (int64_t g = 0; g < options.size(0); ++g) {
        auto o = options[g];
        const double lr=o[0].item<double>(), momentum=o[1].item<double>(), damp=o[2].item<double>(), decay=o[3].item<double>(), nest=o[4].item<double>();
        TORCH_CHECK(lr >= 0 && momentum >= 0 && damp >= 0 && decay >= 0
            && (nest == 0 || nest == 1) && (!nest || (momentum > 0 && damp == 0)), "Invalid Muon optimizer options");
    }
    int64_t count = 0;
    for (const auto& group : param_groups()) count += group.params().size();
    TORCH_CHECK(present.scalar_type() == torch::kInt64 && present.dim() == 1
        && present.numel() == count, "Muon Adam checkpoint parameter count mismatch");
    present = present.cpu();
    std::unordered_map<void*, AdamState> restored;
    int64_t index = 0;
    for (const auto& group : param_groups()) for (const auto& p : group.params()) {
        int64_t flag = present[index].item<int64_t>();
        TORCH_CHECK(flag == 0 || flag == 1, "Invalid Muon Adam checkpoint presence flag");
        if (flag) {
            const bool matrix = (p.dim() == 2 && p.size(0) > 1 && p.size(1) > 1)
                || (p.dim() == 3 && p.size(1) > 1 && p.size(2) > 1);
            TORCH_CHECK(!matrix, "Fallback Adam state attached to a Muon matrix");
            torch::serialize::InputArchive entry;
            adam.read(std::to_string(index), entry);
            AdamState state;
            torch::Tensor step;
            entry.read("mean", state.expAvg);
            entry.read("square", state.expAvgSq);
            entry.read("step", step);
            TORCH_CHECK(step.scalar_type() == torch::kInt64 && step.numel() == 1
                && step.item<int64_t>() > 0, "Invalid Muon Adam step count");
            TORCH_CHECK(state.expAvg.sizes() == p.sizes() && state.expAvgSq.sizes() == p.sizes()
                && state.expAvg.scalar_type() == p.scalar_type() && state.expAvgSq.scalar_type() == p.scalar_type(),
                "Muon Adam checkpoint tensor shape/type mismatch");
            TORCH_CHECK(torch::isfinite(state.expAvg).all().item<bool>()
                && torch::isfinite(state.expAvgSq).all().item<bool>()
                && (state.expAvgSq >= 0).all().item<bool>(), "Invalid Muon Adam moments");
            state.expAvg = state.expAvg.to(p.device());
            state.expAvgSq = state.expAvgSq.to(p.device());
            state.stepCount = step.item<int64_t>();
            restored.emplace(p.unsafeGetTensorImpl(), std::move(state));
        }
        ++index;
    }
    adamStates = std::move(restored);
    for (size_t g = 0; g < param_groups().size(); ++g) {
        auto v = options[(int64_t)g];
        auto& o = static_cast<MuonOptions&>(param_groups()[g].options());
        o.lr(v[0].item<double>()); o.momentum(v[1].item<double>());
        o.dampening(v[2].item<double>()); o.weight_decay(v[3].item<double>()); o.nesterov(v[4].item<double>() != 0);
    }
    loadedLegacyAdamState = false;
}
