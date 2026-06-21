#include "TestHarness.h"

#include <GigaLearnCPP/Util/RSNorm.h>
#include <torch/torch.h>
#include <cmath>

// RSNorm (SimBa running observation normalization), exercised directly.
// Pins: count-based batched Welford (var = M2/n), batched == incremental,
// normalize -> ~zero-mean/unit-var, init = identity, JSON round-trip.

using namespace GGL;

static torch::Tensor popMean(const torch::Tensor& x) { return x.mean(0); }
static torch::Tensor popVar(const torch::Tensor& x) { return (x - x.mean(0)).pow(2).mean(0); }
static torch::Tensor popStd(const torch::Tensor& x) { return popVar(x).sqrt(); }

// Welford running mean/var match the analytic batch mean / population variance.
static void rsnorm_matches_analytic_mean_var() {
	torch::manual_seed(0);
	const int W = 4;
	auto scale = torch::tensor({ 1.0, 5.0, 0.1, 20.0 }, torch::kFloat64);
	auto offset = torch::tensor({ 0.0, 3.0, -1.0, 100.0 }, torch::kFloat64);
	auto x = torch::randn({ 2000, W }, torch::kFloat64) * scale + offset;

	RSNorm rs(W);
	rs.Update(x.to(torch::kFloat32)); // real path feeds float obs

	// init n=1e-4 is negligible vs N=2000
	TCHECK(torch::allclose(rs.runningMean, popMean(x), 1e-3, 1e-2));
	auto varR = rs.runningM2 / rs.count;
	TCHECK(torch::allclose(varR, popVar(x), 5e-3, 1e-1));
}

// Welford is exact regardless of batching: one [N,W] update == three chunked updates.
static void rsnorm_batched_equals_incremental() {
	torch::manual_seed(1);
	const int W = 4;
	auto x = torch::randn({ 600, W }) * 3.0 + 1.5;

	RSNorm a(W), b(W);
	a.Update(x);
	b.Update(x.slice(0, 0, 200));
	b.Update(x.slice(0, 200, 400));
	b.Update(x.slice(0, 400, 600));

	TCHECK(torch::allclose(a.runningMean, b.runningMean, 1e-4, 1e-4));
	TCHECK(torch::allclose(a.runningM2, b.runningM2, 1e-3, 1e-2));
	TCHECK(std::abs(a.count - b.count) < 1e-9);
}

// After updating on a distribution, normalizing it yields ~zero-mean / unit-var per dim.
static void rsnorm_normalizes_to_unit() {
	torch::manual_seed(2);
	const int W = 3;
	auto scale = torch::tensor({ 2.f, 10.f, 0.5f });
	auto offset = torch::tensor({ 1.f, -4.f, 7.f });
	auto x = torch::randn({ 4000, W }) * scale + offset;

	RSNorm rs(W);
	rs.Update(x);
	auto y = rs.Normalize(x);

	TCHECK(torch::allclose(popMean(y), torch::zeros({ W }), 1e-2, 1e-2));
	TCHECK(torch::allclose(popStd(y), torch::ones({ W }), 1e-2, 2e-2));
}

// Fresh normalizer (mu=0, var=1) is ~identity (no warmup needed).
static void rsnorm_init_is_identity() {
	const int W = 5;
	RSNorm rs(W);
	auto x = torch::randn({ 10, W });
	TCHECK(torch::allclose(rs.Normalize(x), x, 1e-3, 1e-3));
}

// JSON serialize/deserialize round-trips stats and the resulting normalization.
static void rsnorm_serialize_roundtrip() {
	torch::manual_seed(3);
	const int W = 6;
	auto x = torch::randn({ 500, W }) * 4.0 - 2.0;

	RSNorm a(W);
	a.Update(x);
	auto j = a.ToJSON();

	RSNorm b(W);
	b.ReadFromJSON(j);

	TCHECK(torch::allclose(a.runningMean, b.runningMean));
	TCHECK(torch::allclose(a.runningM2, b.runningM2));
	TCHECK(std::abs(a.count - b.count) < 1e-9);

	auto probe = torch::randn({ 3, W });
	TCHECK(torch::allclose(a.Normalize(probe), b.Normalize(probe)));
}

void RunRSNormTests() {
	RUN_SUITE("RSNorm", [] {
		rsnorm_matches_analytic_mean_var();
		rsnorm_batched_equals_incremental();
		rsnorm_normalizes_to_unit();
		rsnorm_init_is_identity();
		rsnorm_serialize_roundtrip();
	});
}
