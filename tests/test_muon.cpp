#include "TestHarness.h"

#include <GigaLearnCPP/Util/Muon.h>
#include <torch/torch.h>
#include <cmath>

// Muon optimizer, exercised through its public step() interface.
//
// The regression these pin: AdjustedLR scaled the orthogonalized update by
// sqrt(max(1, rows/cols)) (== 1 for square layers) instead of
// sqrt(max(rows, cols)).  That left Muon algorithmically correct (orthogonal,
// finite, descending) but ~sqrt(d) (~17x for 256-wide) too small a step for an
// Adam-tuned learning rate -- so training crawled and the policy looked frozen.
// Correctness checks pass on that bug; only a step-MAGNITUDE check catches it.

using namespace GGL;

static double RMS(const torch::Tensor& t) {
	return std::sqrt(t.detach().pow(2).mean().item<double>());
}

// Behaviour 1 (the regression): a square 2D param's per-step update is
// calibrated to the LR (Adam scale) -- per-element RMS ~= lr -- not orders of
// magnitude smaller.  Buggy scaling gives ~lr/sqrt(d) and fails the lower bound.
static void muon_step_magnitude_is_lr_calibrated() {
	torch::manual_seed(0);
	const int64_t d = 256;
	const double lr = 1e-3;

	auto p = torch::randn({ d, d });
	p.set_requires_grad(true);
	p.mutable_grad() = torch::randn({ d, d }); // fixed gradient

	auto before = p.detach().clone();

	// weight_decay 0 so the measured delta is purely the orthogonalized update;
	// decoupled weight decay would add a ~p*lr*wd term that masks the magnitude.
	MuonOptions opts(lr);
	opts.weight_decay(0.0);
	Muon opt({ p }, opts);
	opt.step();

	double updateRMS = RMS(before - p);

	// Fixed scaling: updateRMS ~= 0.9*lr.  Buggy: updateRMS ~= lr/sqrt(256) ~= lr/16.
	TCHECK(updateRMS > 0.4 * lr);
	TCHECK(updateRMS < 2.5 * lr);
}

// Behaviour 2: the step is a descent direction -- the param moves opposite the
// gradient (orthogonalization preserves positive correlation with the input), and
// stays finite.
static void muon_step_is_a_descent_direction() {
	torch::manual_seed(1);
	const int64_t d = 64;

	auto p = torch::randn({ d, d });
	p.set_requires_grad(true);
	auto g = torch::randn({ d, d });
	p.mutable_grad() = g.clone();

	auto before = p.detach().clone();

	MuonOptions opts(1e-2);
	opts.weight_decay(0.0);
	Muon opt({ p }, opts);
	opt.step();

	auto delta = p.detach() - before;                     // actual param change
	double dot = (delta * g).sum().item<double>();
	TCHECK(dot < 0.0);                                     // moved opposite the gradient
	TCHECK(torch::isfinite(p).all().item<bool>());
}

// Behaviour 3: 1D params (biases, norms) take the AdamW fallback path, not
// orthogonalization. With a constant gradient the first Adam step is ~ -lr*sign(g)
// per element -- which the Muon (orthogonal) path would NOT produce.
static void muon_uses_adamw_for_1d_params() {
	torch::manual_seed(2);
	const int64_t d = 128;
	const double lr = 1e-2;

	auto p = torch::randn({ d });          // 1D -> grad.dim() < 2 -> StepAdamW
	p.set_requires_grad(true);
	p.mutable_grad() = torch::full({ d }, 0.5f); // constant positive gradient

	auto before = p.detach().clone();

	MuonOptions opts(lr);
	opts.weight_decay(0.0);
	Muon opt({ p }, opts);
	opt.step();

	double meanDelta = (p.detach() - before).mean().item<double>();
	TCHECK_NEAR(meanDelta, -lr, lr * 0.05); // Adam first step ~= -lr*sign(g) = -lr
	TCHECK(torch::isfinite(p).all().item<bool>());
}

// Behaviour 4 (end-to-end): Muon actually minimizes a simple convex objective.
// This is the integration guard the magnitude bug slipped past -- with the
// ~17x-too-small step the loss would barely move in this many iterations.
static void muon_minimizes_a_quadratic() {
	torch::manual_seed(3);
	const int64_t d = 32;
	auto target = torch::randn({ d, d });

	auto p = torch::randn({ d, d });
	p.set_requires_grad(true);

	auto lossOf = [&](const torch::Tensor& w) {
		return (w.detach() - target).pow(2).mean().item<double>();
	};
	double initialLoss = lossOf(p);

	MuonOptions opts(0.02);
	opts.weight_decay(0.0); // so the minimum is exactly `target`
	Muon opt({ p }, opts);

	for (int i = 0; i < 300; i++) {
		// analytic gradient of mean((p - target)^2)
		p.mutable_grad() = (p.detach() - target) * (2.0 / (double)p.numel());
		opt.step();
	}

	double finalLoss = lossOf(p);
	TCHECK(finalLoss < 0.2 * initialLoss);
	TCHECK(torch::isfinite(p).all().item<bool>());
}

// Behaviour 5: the 2D update is the orthogonalization of the gradient -- its
// singular-value spectrum is compressed toward uniform (low relative spread)
// compared to the raw gradient. A no-op / broken Newton-Schulz would leave the
// update as spread-out as the gradient.
static double RelSpread(const torch::Tensor& m) {
	auto sv = std::get<1>(torch::svd(m.detach())); // singular values (U, S, V) -> S
	return (sv.std(false) / sv.mean().clamp_min(1e-12)).item<double>();
}

static void muon_orthogonalizes_the_update() {
	torch::manual_seed(4);
	const int64_t d = 96;

	auto p = torch::randn({ d, d });
	p.set_requires_grad(true);
	auto g = torch::randn({ d, d });
	p.mutable_grad() = g.clone();

	auto before = p.detach().clone();

	MuonOptions opts(1e-2);
	opts.weight_decay(0.0);
	Muon opt({ p }, opts);
	opt.step();

	auto update = before - p.detach();
	TCHECK(RelSpread(update) < RelSpread(g) * 0.75); // markedly more uniform than the gradient
}

void RunMuonTests() {
	RUN_SUITE("Muon", [] {
		muon_step_magnitude_is_lr_calibrated();
		muon_step_is_a_descent_direction();
		muon_uses_adamw_for_1d_params();
		muon_minimizes_a_quadratic();
		muon_orthogonalizes_the_update();
	});
}
