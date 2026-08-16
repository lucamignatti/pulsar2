#include "MoE.h"
#include "Models.h"
#include <torch/cuda.h>
#ifdef GGL_MOE_KERNELS
#include "MoEKernels.h"
#include <c10/cuda/CUDAStream.h>
#include <ATen/autocast_mode.h>
#include <torch/version.h>

namespace {
	// The custom routed-FFN manages precision EXPLICITLY (fp32 routing, fp16 GEMMs).
	// The fleet's learn pass wraps forwards in autocast, which intercepted the
	// router mm and emitted fp16 logits that the route-plan kernel then read as
	// fp32 — 2x past the buffer, an illegal access that crash-looped the chain
	// (jobs 4631411-417; the selftest never saw it because it runs autocast-free).
	struct AutocastOffScope {
		bool prev = false;
		AutocastOffScope() {
#if TORCH_VERSION_MAJOR > 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 4)
			prev = at::autocast::is_autocast_enabled(at::kCUDA);
			if (prev)
				at::autocast::set_autocast_enabled(at::kCUDA, false);
#else
			prev = at::autocast::is_enabled();
			if (prev)
				at::autocast::set_enabled(false);
#endif
		}
		~AutocastOffScope() {
#if TORCH_VERSION_MAJOR > 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 4)
			if (prev)
				at::autocast::set_autocast_enabled(at::kCUDA, true);
#else
			if (prev)
				at::autocast::set_enabled(true);
#endif
		}
	};
}
#endif

using namespace torch;

// Isolated forward/backward/clone microtest (called from ExampleMain via extern decl;
// GGL_MOE_SELFTEST=1). Exists because the first in-trainer segfault took four bisect
// runs to localize — this reproduces the grad path in 2 seconds.
int GGL::RunMoESelfTest() {
	RG_LOG("MoE selftest: constructing block...");
	auto blk = GGL::MoEBlock(64, 32, 8, 2);
	{
		RG_LOG("MoE selftest: no-grad forward...");
		torch::NoGradGuard ng;
		auto y0 = blk->forward(torch::randn({ 96, 64 }));
		RG_LOG("  out " << y0.sizes() << " finite=" << y0.isfinite().all().item<bool>());
	}
	RG_LOG("MoE selftest: grad forward...");
	auto x = torch::randn({ 96, 64 });
	auto y = blk->forward(x);
	RG_LOG("MoE selftest: backward...");
	y.sum().backward();
	RG_LOG("MoE selftest: grad norms: routerW=" << blk->routerW.grad().norm().item<float>()
		<< " expertW1=" << blk->expertW1.grad().norm().item<float>()
		<< " sharedW1=" << blk->sharedW1.grad().norm().item<float>());
	RG_LOG("MoE selftest: clone...");
	auto c = blk->clone();
	RG_LOG("MoE selftest: bias update entropy=" << blk->UpdateRouterBias(1e-3f));

	// Full-buffer-scale phase: the trainer's gap/learn forwards run ~25k rows through a
	// stack of blocks with grad — the 96-row test above missed a segfault there once.
	RG_LOG("MoE selftest: big-R stacked grad forward (24832 x 3 blocks, w256)...");
	{
		std::vector<GGL::MoEBlock> blocks;
		for (int i = 0; i < 3; i++)
			blocks.push_back(GGL::MoEBlock(256, 64, 16, 2));
		auto xb = torch::randn({ 24832, 256 });
		auto yb = xb;
		for (auto& b : blocks)
			yb = b->forward(yb);
		RG_LOG("MoE selftest: big-R backward...");
		yb.sum().backward();
		RG_LOG("  big-R grads finite="
			<< blocks[0]->expertW1.grad().isfinite().all().item<bool>());
	}
	// Model-level phase: the trainer path is GGL::Model (Sequential + AnyModule +
	// embed/LN) -> Forward(x, false), repeated at buffer scale on the GPU — the raw
	// block tests above miss anything in that wrapping.
	if (torch::cuda::device_count() > 0) {
		RG_LOG("MoE selftest: GGL::Model GPU phase...");
		GGL::ModelConfig mc = GGL::ModelConfig(GGL::PartialModelConfig{});
		mc.numInputs = 230;
		mc.layerSizes = { 256 };
		mc.moeBlocks = 3;
		mc.moeExperts = 16;
		mc.moeTopK = 2;
		mc.moeHidden = 64;
		mc.addOutputLayer = false;
				mc.optimType = GGL::ModelOptimType::MUON;
		auto dev = torch::Device(torch::kCUDA, 0);
		GGL::Model model("moe_selftest", mc, dev);
		for (int it = 0; it < 30; it++) {
			RG_NO_GRAD;
			// value-pred-like: contiguous device batch
			auto x1 = torch::randn({ 24832, 230 }, dev);
			auto y1 = model.Forward(x1, false);
			// gap-like: CPU randperm gather then upload
			auto xc = torch::randn({ 24832, 230 });
			auto idx = torch::randperm(24832,
				torch::TensorOptions().dtype(torch::kLong)).slice(0, 0, 16384);
			auto y2 = model.Forward(xc.index_select(0, idx).to(dev, true), false);
			if (it % 10 == 0)
				RG_LOG("  it " << it << " y1 " << y1.mean().item<float>()
					<< " y2 " << y2.mean().item<float>());
		}
		RG_LOG("MoE selftest: GPU model phase OK");
	}
#ifdef GGL_MOE_KERNELS
	// Fast-path parity + speed (MOE_SPEED.md Stage 1 gate): the CUTLASS forward must
	// match the eager baddbmm forward on identical fp16 inputs before it may serve
	// the fleet. capacityFactor is raised so the eager path drops nothing (the fast
	// path never drops), making the comparison exact up to fp16 rounding; routerW is
	// boosted so selection is decisive (at raw init all 128 affinities are near-ties
	// and fp16-vs-fp32 logit rounding would flip selections spuriously).
	if (torch::cuda::device_count() > 0 && ggl_moe_kernels_available()) {
		RG_LOG("MoE selftest: CUTLASS fast-path parity (live dims 1024/512/128/4)...");
		auto dev = torch::Device(torch::kCUDA, 0);
		auto pb = GGL::MoEBlock(1024, 512, 128, 4);
		pb->capacityFactor = 8.f;
		{
			torch::NoGradGuard ng0;
			pb->routerW.mul_(10.0);
		}
		pb->to(dev);
		pb->to(torch::kHalf);
		auto xp = (torch::randn({ 777, 1024 }, dev) * 0.5).to(torch::kHalf);
		torch::NoGradGuard ng;
		pb->fastPathForce = 0;
		auto yRef = pb->forward(xp).to(torch::kFloat);
		pb->fastPathForce = 1;
		auto yFast = pb->forward(xp).to(torch::kFloat);
		float rel = (yFast - yRef).norm().item<float>()
			/ std::max(yRef.norm().item<float>(), 1e-6f);
		RG_LOG("  parity relRMS=" << rel);
		if (!(rel < 2e-2f))
			RG_ERR_CLOSE("MoE CUTLASS fast-path parity FAILED: relRMS=" << rel);
		auto fnTime = [&](int force) {
			pb->fastPathForce = force;
			for (int rep = 0; rep < 3; rep++)
				pb->forward(xp);
			torch::cuda::synchronize();
			auto t0 = std::chrono::steady_clock::now();
			for (int rep = 0; rep < 20; rep++)
				pb->forward(xp);
			torch::cuda::synchronize();
			return std::chrono::duration<double, std::milli>(
				std::chrono::steady_clock::now() - t0).count() / 20.0;
		};
		double eagerMs = fnTime(0), fastMs = fnTime(1);
		RG_LOG("  block fwd 777 rows: eager " << eagerMs << " ms, fast " << fastMs
			<< " ms (" << (eagerMs / fastMs) << "x)");
	}
	// Stage 1b gate: LEARN-path grad parity vs eager autograd, per param family.
	// fp32 eager vs fp16-compute custom -> tolerance 3e-2. cf=8 so eager drops nothing.
	if (torch::cuda::device_count() > 0 && ggl_moe_kernels_available()) {
		RG_LOG("MoE selftest: CUTLASS learn-path grad parity...");
		auto dev = torch::Device(torch::kCUDA, 0);
		auto lb = GGL::MoEBlock(1024, 512, 128, 4);
		lb->capacityFactor = 8.f;
		{
			torch::NoGradGuard ng0;
			lb->routerW.mul_(10.0);
		}
		lb->to(dev);
		auto xl = (torch::randn({ 777, 1024 }, dev) * 0.5).set_requires_grad(true);
		// Random upstream grad: y.sum() (all-ones dOut) makes the B1 reference a
		// heavily sign-cancelled near-zero sum, so relRMS divided fp16 noise by ~0.
		auto upG = torch::randn({ 777, 1024 }, dev);
		auto fnGrads = [&](int force) {
			lb->learnFastForce = force;
			if (xl.grad().defined()) xl.mutable_grad() = torch::Tensor();
			for (auto& pr : lb->parameters())
				if (pr.grad().defined()) pr.mutable_grad() = torch::Tensor();
			auto yl = lb->forward(xl);
			yl.backward(upG);
			std::vector<torch::Tensor> g;
			g.push_back(xl.grad().clone());
			for (auto& pr : { lb->routerW, lb->expertW1, lb->expertB1, lb->expertW2, lb->expertB2 })
				g.push_back(pr.grad().defined() ? pr.grad().clone() : torch::zeros_like(pr));
			return g;
		};
		auto gRef = fnGrads(0);
		auto gFast = fnGrads(1);
		const char* names[] = { "dx", "routerW", "expertW1", "expertB1", "expertW2", "expertB2" };
		bool ok = true;
		for (size_t i = 0; i < gRef.size(); i++) {
			float refN = gRef[i].norm().item<float>();
			// Cancellation-robust denominator: a reference whose entries nearly cancel
			// (bias grads) cannot be the yardstick for fp16 elementwise noise.
			float den = std::max(refN,
				std::max(gFast[i].norm().item<float>() * 0.5f,
					1e-4f * std::sqrt((float)gRef[i].numel())));
			float rel = (gFast[i] - gRef[i]).norm().item<float>() / std::max(den, 1e-6f);
			RG_LOG("  grad parity " << names[i] << " relRMS=" << rel
				<< " (refNorm=" << refN << ")");
			if (!(rel < 3e-2f))
				ok = false;
		}
		if (!ok)
			RG_ERR_CLOSE("MoE CUTLASS LEARN-path grad parity FAILED (see relRMS above)");
		// Autocast coverage: the fleet learn pass runs under AMP, which is exactly
		// how the fp16-logits illegal access escaped the autocast-free gate above.
		{
#if TORCH_VERSION_MAJOR > 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 4)
			at::autocast::set_autocast_dtype(at::kCUDA, torch::kHalf);
			at::autocast::set_autocast_enabled(at::kCUDA, true);
#else
			at::autocast::set_autocast_gpu_dtype(torch::kHalf);
			at::autocast::set_enabled(true);
#endif
			auto gAmp = fnGrads(1);
#if TORCH_VERSION_MAJOR > 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 4)
			at::autocast::set_autocast_enabled(at::kCUDA, false);
#else
			at::autocast::set_enabled(false);
#endif
			at::autocast::clear_cache();
			bool fin = true;
			for (auto& g : gAmp)
				fin = fin && g.isfinite().all().item<bool>();
			RG_LOG("  autocast learn-path grads finite=" << fin);
			if (!fin)
				RG_ERR_CLOSE("MoE CUTLASS LEARN-path under autocast produced non-finite grads");
		}
	}
#endif
	RG_LOG("MoE selftest OK");
	return 0;
}

GGL::MoEBlockImpl::MoEBlockImpl(int64_t dim, int64_t hidden, int64_t numExperts, int64_t topK)
	: dim(dim), hidden(hidden), numExperts(numExperts), topK(topK) {
	reset();
	// Kaiming-ish init scaled like the dense trunk layers; router near-zero so early
	// selection is ~uniform and the balancing bias starts in control. Lives in the
	// CONSTRUCTOR, not reset(): Cloneable::clone calls reset() on every clone and then
	// overwrites the values — a torch::randn init there generated ~1B CPU gaussians per
	// fp16-cache build (~25s/clone, job 4630866). reset()'s contract is registration.
	torch::NoGradGuard ng;
	const double s1 = std::sqrt(2.0 / (double)dim);
	const double s2 = std::sqrt(2.0 / (double)hidden);
	routerW.normal_(0.0, 0.01);
	expertW1.normal_(0.0, s1);
	expertB1.zero_();
	expertW2.normal_(0.0, s2);
	expertB2.zero_();
	sharedW1.normal_(0.0, s1);
	sharedB1.zero_();
	sharedW2.normal_(0.0, s2);
	sharedB2.zero_();
	routerBias.zero_();
	loadAcc.zero_();
}

void GGL::MoEBlockImpl::reset() {
	if (dim <= 0)
		return;
	// Registration ONLY (see ctor note): empty tensors, no expensive init.
	routerW = register_parameter("routerW", torch::empty({ numExperts, dim }));
	expertW1 = register_parameter("expertW1", torch::empty({ numExperts, hidden, dim }));
	expertB1 = register_parameter("expertB1", torch::empty({ numExperts, hidden }));
	expertW2 = register_parameter("expertW2", torch::empty({ numExperts, dim, hidden }));
	expertB2 = register_parameter("expertB2", torch::empty({ numExperts, dim }));
	sharedW1 = register_parameter("sharedW1", torch::empty({ hidden, dim }));
	sharedB1 = register_parameter("sharedB1", torch::empty({ hidden }));
	sharedW2 = register_parameter("sharedW2", torch::empty({ dim, hidden }));
	sharedB2 = register_parameter("sharedB2", torch::empty({ dim }));
	routerBias = register_buffer("routerBias", torch::empty({ numExperts }));
	loadAcc = register_buffer("loadAcc", torch::empty({ numExperts }));
	ln = register_module("ln", torch::nn::LayerNorm(torch::nn::LayerNormOptions({ dim })));
}

torch::Tensor GGL::MoEBlockImpl::forward(torch::Tensor x) {
#ifdef GGL_MOE_KERNELS
	// Fast path: fused routing + exact-M CUTLASS grouped GEMMs (MOE_SPEED.md Stage 1).
	// no-grad fp16 CUDA only — exactly the collect forward; learn keeps autograd baddbmm.
	{
		static const int envOn = [] {
			const char* e = std::getenv("GGL_MOE_CUTLASS");
			return (e && *e && std::string(e) != "0") ? 1 : 0;
		}();
		int on = fastPathForce >= 0 ? fastPathForce : envOn;
		if (on && !torch::GradMode::is_enabled() && x.is_cuda()
			&& x.scalar_type() == torch::kHalf && ggl_moe_kernels_available())
			return ForwardFast(x);
	}
	// Stage 1b learn path: custom autograd routed-FFN (grouped GEMM fwd + dgrad +
	// wgrad). fp32 grad-enabled CUDA only. LN/shared/residual stay in autograd.
	{
		static const int envLearnOn = [] {
			const char* e = std::getenv("GGL_MOE_CUTLASS_LEARN");
			return (e && *e && std::string(e) != "0") ? 1 : 0;
		}();
		int on = learnFastForce >= 0 ? learnFastForce : envLearnOn;
		if (on && torch::GradMode::is_enabled() && x.is_cuda() && x.requires_grad()
			&& x.scalar_type() == torch::kFloat && ggl_moe_kernels_available()) {
			auto xn = ln->forward(x);
			auto routed = GGL::MoERoutedFFNApply(this, xn);
			auto hs = torch::leaky_relu(torch::addmm(sharedB1, xn, sharedW1.t()), 0.01);
			auto ys = torch::addmm(sharedB2, hs, sharedW2.t());
			return x + routed + ys;
		}
	}
#endif
	const int64_t R = x.size(0);
	auto xn = ln->forward(x);

	// Router in fp32 regardless of the block's dtype (V100 fp16 sigmoid/topk stability).
	auto xn32 = xn.to(torch::kFloat32);
	auto aff = torch::sigmoid(torch::matmul(xn32, routerW.to(torch::kFloat32).t())); // [R,E]
	torch::Tensor sel;
	{
		NoGradGuard ng; // selection is non-differentiable anyway; bias is selection-only
		sel = std::get<1>(torch::topk(aff.detach() + routerBias.to(torch::kFloat32).unsqueeze(0),
			topK, /*dim=*/-1)); // [R,k]
		// load tracking via scatter_add on a fresh tensor (boring, static-shaped)
		auto ones = torch::ones({ R * topK }, loadAcc.options());
		loadAcc.add_(torch::zeros({ numExperts }, loadAcc.options())
			.scatter_add(0, sel.flatten().to(torch::kLong), ones));
	}
	auto gateW = torch::gather(aff, 1, sel);                    // [R,k] differentiable
	gateW = gateW / gateW.sum(-1, /*keepdim=*/true).clamp_min(1e-9);

	// Capacity bucketing, STATIC shapes only (no masked_select / bincount / boolean
	// compaction — every tensor here has a shape fixed by (R, E, cap)). Overflowed
	// assignments keep a slot in a dedicated trash row of the buffer and get weight 0.
	const int64_t nAssign = R * topK;
	const int64_t cap = std::max<int64_t>(1,
		(int64_t)std::ceil((double)nAssign / (double)numExperts * (double)capacityFactor));
	torch::Tensor slots, rowIdx, keepF;
	{
		NoGradGuard ng;
		auto flatExp = sel.flatten().to(torch::kLong);           // [R*k]
		auto order = torch::argsort(flatExp, 0, false);
		auto sortedExp = flatExp.index_select(0, order);
		// per-expert rank of each sorted assignment: position - first-position-of-expert.
		// first-position via cummax over a "position where expert changes" mask.
		auto pos = torch::arange(nAssign, flatExp.options());
		auto isFirst = torch::cat({ torch::ones({ 1 }, torch::TensorOptions()
				.dtype(torch::kBool).device(flatExp.device())),
			sortedExp.slice(0, 1, nAssign) != sortedExp.slice(0, 0, nAssign - 1) });
		auto firstPos = std::get<0>(torch::cummax(pos * isFirst.to(pos.dtype()), 0));
		auto rankInExp = pos - firstPos;
		auto keep = rankInExp < cap;                              // [nAssign] bool, static
		// kept -> real slot; dropped -> trash row E*cap (weight zeroed below)
		auto slotSorted = torch::where(keep, sortedExp * cap + rankInExp,
			torch::full_like(sortedExp, numExperts * cap));
		// scatter back to assignment order (inverse of `order`), all static shapes
		slots = torch::empty_like(slotSorted);
		slots = slots.scatter(0, order, slotSorted);
		keepF = torch::zeros({ nAssign }, xn32.options());
		keepF = keepF.scatter(0, order, keep.to(xn32.options().dtype()));
		rowIdx = torch::floor_divide(
			torch::arange(nAssign, flatExp.options()), topK);     // [nAssign], row per assignment
	}

	// Gather every assignment's input row into the padded expert buffer (+1 trash row).
	auto buf = torch::zeros({ numExperts * cap + 1, xn.size(1) }, xn.options());
	buf = buf.index_put({ slots }, xn.index_select(0, rowIdx));   // static [nAssign] writes
	auto b3 = buf.slice(0, 0, numExperts * cap).view({ numExperts, cap, dim });

	auto h = torch::baddbmm(expertB1.unsqueeze(1), b3, expertW1.transpose(1, 2));
	h = torch::leaky_relu(h, 0.01);
	auto y = torch::baddbmm(expertB2.unsqueeze(1), h, expertW2.transpose(1, 2)); // [E,cap,d]

	// Read every assignment's slot back (trash row reads garbage but weight is 0).
	auto yAll = torch::cat({ y.view({ numExperts * cap, dim }),
		torch::zeros({ 1, dim }, y.options()) }, 0)
		.index_select(0, slots);                                  // [nAssign, d]
	auto wAll = (gateW.flatten() * keepF.to(gateW.dtype())).unsqueeze(1).to(yAll.dtype());
	auto out = torch::zeros_like(xn);
	// autocast keeps LN (and thus xn/out) fp32 while the bmm outputs are bf16/fp16 —
	// index_add requires matching dtypes, so cast the contribution to out's type.
	out = out.index_add(0, rowIdx, (yAll * wAll).to(out.scalar_type())); // differentiable

	// Shared expert: dense safety net for capacity drops, always active.
	auto hs = torch::leaky_relu(
		torch::addmm(sharedB1, xn, sharedW1.t()), 0.01);
	auto ys = torch::addmm(sharedB2, hs, sharedW2.t());

	return x + out + ys;
}

#ifdef GGL_MOE_KERNELS
torch::Tensor GGL::MoEBlockImpl::ForwardFast(torch::Tensor x) {
	// One-time activation proof: the env is read lazily inside forward, so without
	// this line nothing in the boot log distinguishes fast-path-on from silently-eager
	// (the exact ambiguity the config-order trap taught us to close).
	static std::once_flag activeLog;
	std::call_once(activeLog, [&] {
		RG_LOG("MoE CUTLASS fast path ACTIVE (E=" << numExperts << ", k=" << topK
			<< ", hidden=" << hidden << ")");
	});
	void* s = (void*)at::cuda::getCurrentCUDAStream().stream();
	const int64_t R = x.size(0);
	const int64_t n = R * topK;
	const int E = (int)numExperts;

	auto xn = ln->forward(x);                              // fp16

	RG_ASSERT(routerBias.scalar_type() == torch::kHalf);   // half-clone buffers only

	// Weight caches in CUTLASS layout. copy_ INTO EXISTING STORAGE on refresh:
	// a CUDA graph captured over this path holds these pointers, and the half
	// clone's params are themselves refreshed in place once per iteration.
	uint64_t ep = g_halfRefreshEpoch.load(std::memory_order_acquire);
	if (!_w1T.defined()) {
		torch::NoGradGuard ng;
		_w1T = expertW1.transpose(1, 2).contiguous();      // [E, d, h]
		_w2T = expertW2.transpose(1, 2).contiguous();      // [E, h, d]
		_routerW32 = routerW.to(torch::kFloat);            // routing is fp32 (parity with eager)
		_wCacheEpoch = ep;
	} else if (ep != _wCacheEpoch) {
		torch::NoGradGuard ng;
		_w1T.copy_(expertW1.transpose(1, 2), true);
		_w2T.copy_(expertW2.transpose(1, 2), true);
		_routerW32.copy_(routerW, true);
		_wCacheEpoch = ep;
	}

	// Per-shape persistent scratch (see MoE.h for why per-shape).
	FastScratch& sc = _scratchByRows[R];
	if (!sc.packed.defined()) {
		auto iopts = torch::TensorOptions().dtype(torch::kInt32).device(x.device());
		auto fopts = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
		sc.counts = torch::empty({ 2 * numExperts }, iopts);
		sc.offsets = torch::empty({ numExperts + 1 }, iopts);
		sc.rowExp = torch::empty({ n }, iopts);
		sc.rowGate = torch::empty({ n }, fopts);
		sc.srcRows = torch::empty({ n }, iopts);
		sc.expertId = torch::empty({ n }, iopts);
		sc.gates = torch::empty({ n }, fopts);
		sc.logits32 = torch::empty({ R, numExperts }, fopts);
		sc.packed = torch::empty({ n, dim }, x.options());
		sc.hid = torch::empty({ n, hidden }, x.options());
		sc.out = torch::empty({ R, dim }, x.options());
	}
	if (!_gemmCtx)
		_gemmCtx = ggl_moe_ctx_create(E);

	// fp32 router logits, written into the persistent buffer (mm_out: no alloc).
	torch::mm_out(sc.logits32, xn.to(torch::kFloat), _routerW32.t());

	ggl_moe_route_plan_f16(
		sc.logits32.data_ptr(), routerBias.data_ptr(), (int)R, E, (int)topK, /*align=*/0,
		sc.counts.data_ptr<int>(), sc.offsets.data_ptr<int>(),
		sc.rowExp.data_ptr<int>(), sc.rowGate.data_ptr<float>(),
		sc.srcRows.data_ptr<int>(), sc.expertId.data_ptr<int>(),
		sc.gates.data_ptr<float>(), s);
	ggl_moe_gather_f16(xn.data_ptr(), sc.srcRows.data_ptr<int>(),
		sc.packed.data_ptr(), (int)n, (int)dim, s);
	ggl_moe_grouped_gemm_f16_dev(_gemmCtx, sc.packed.data_ptr(), _w1T.data_ptr(),
		sc.hid.data_ptr(), sc.offsets.data_ptr<int>(), E, (int)dim, (int)hidden, s);
	ggl_moe_bias_leaky_f16(sc.hid.data_ptr(), expertB1.data_ptr(),
		sc.expertId.data_ptr<int>(), (int)n, (int)hidden, 0.01f, s);
	ggl_moe_grouped_gemm_f16_dev(_gemmCtx, sc.hid.data_ptr(), _w2T.data_ptr(),
		sc.packed.data_ptr(), sc.offsets.data_ptr<int>(), E, (int)hidden, (int)dim, s);
	ggl_moe_zero_f16(sc.out.data_ptr(), (int64_t)R * dim, s);
	ggl_moe_scatter_bias_gate_f16(sc.packed.data_ptr(), expertB2.data_ptr(),
		sc.expertId.data_ptr<int>(), sc.gates.data_ptr<float>(),
		sc.srcRows.data_ptr<int>(), sc.out.data_ptr(), (int)n, (int)dim, s);

	// Load tracking for the aux-free balancing panel/update (counts came free).
	loadAcc.add_(sc.counts.narrow(0, 0, numExperts).to(loadAcc.scalar_type()));

	auto hs = torch::leaky_relu(torch::addmm(sharedB1, xn, sharedW1.t()), 0.01);
	auto ys = torch::addmm(sharedB2, hs, sharedW2.t());
	return x + sc.out + ys;
}
#endif

#ifdef GGL_MOE_KERNELS
// ==================== Stage 1b: custom autograd routed-FFN ====================
// Forward mirrors ForwardFast (exact-M grouped GEMMs, no capacity drops); backward
// is grouped dgrad/wgrad + segment sums + a gate-grad chain. LN, shared expert and
// the residual stay OUTSIDE in ordinary autograd. All GEMM compute fp16 (tensor
// cores); returned grads fp32 so accumulation/AMP-unscale are unchanged.
namespace {

struct MoERoutedFFN : public torch::autograd::Function<MoERoutedFFN> {
	static torch::Tensor forward(
		torch::autograd::AutogradContext* ctx,
		torch::Tensor xn,
		torch::Tensor routerW, torch::Tensor expertW1, torch::Tensor expertB1,
		torch::Tensor expertW2, torch::Tensor expertB2, torch::Tensor routerBias,
		int64_t blkPtr) {

		AutocastOffScope acOff; // see the struct comment: explicit precision only
		auto* blk = reinterpret_cast<GGL::MoEBlockImpl*>(blkPtr);
		void* s = (void*)at::cuda::getCurrentCUDAStream().stream();
		const int64_t R = xn.size(0), d = blk->dim, h = blk->hidden;
		const int E = (int)blk->numExperts, k = (int)blk->topK;
		const int64_t nReal = R * k;
		// Aligned segments: each expert's slot range rounds to 8 (wgrad fp16 tensor-op
		// pointer/K alignment). Buffers sized to the bound; pad slots carry srcRows=-1
		// and contribute exact zeros end to end.
		const int64_t n = ((nReal + 7) & ~7LL) + 8LL * E;
		auto dev = xn.device();
		auto h16 = torch::TensorOptions().dtype(torch::kHalf).device(dev);
		auto f32 = torch::TensorOptions().dtype(torch::kFloat).device(dev);
		auto i32 = torch::TensorOptions().dtype(torch::kInt32).device(dev);

		// fp16 learn-side weight caches, invalidated by the fp32 params' version
		// counters (optimizer steps are in-place).
		uint64_t v1 = expertW1.unsafeGetTensorImpl()->version_counter().current_version();
		uint64_t v2 = expertW2.unsafeGetTensorImpl()->version_counter().current_version();
		if (blk->_lwVer1 != v1 || blk->_lwVer2 != v2 || !blk->_lw1.defined()) {
			torch::NoGradGuard ng;
			blk->_lw1 = expertW1.detach().to(torch::kHalf).contiguous();
			blk->_lw1T = expertW1.detach().transpose(1, 2).to(torch::kHalf).contiguous();
			blk->_lw2 = expertW2.detach().to(torch::kHalf).contiguous();
			blk->_lw2T = expertW2.detach().transpose(1, 2).to(torch::kHalf).contiguous();
			blk->_lb1 = expertB1.detach().to(torch::kHalf).contiguous();
			blk->_lb2 = expertB2.detach().to(torch::kHalf).contiguous();
			blk->_lBias = routerBias.detach().to(torch::kHalf).contiguous();
			blk->_lwVer1 = v1;
			blk->_lwVer2 = v2;
		}
		if (!blk->_gemmCtxLearn)
			blk->_gemmCtxLearn = ggl_moe_ctx_create(E);

		auto logits = torch::mm(xn, routerW.t()).contiguous();          // fp32 [R,E]
		auto counts = torch::empty({ 2LL * E }, i32);
		auto offsets = torch::empty({ (int64_t)E + 1 }, i32);
		auto rowExp = torch::empty({ nReal }, i32);
		auto rowGate = torch::empty({ nReal }, f32);
		auto srcRows = torch::empty({ n }, i32);
		auto expertId = torch::empty({ n }, i32);
		auto gates = torch::empty({ n }, f32);
		ggl_moe_route_plan_f16(logits.data_ptr(), blk->_lBias.data_ptr(),
			(int)R, E, k, /*align=*/1, counts.data_ptr<int>(), offsets.data_ptr<int>(),
			rowExp.data_ptr<int>(), rowGate.data_ptr<float>(),
			srcRows.data_ptr<int>(), expertId.data_ptr<int>(), gates.data_ptr<float>(), s);

		auto xn16 = xn.to(torch::kHalf);
		auto packed = torch::empty({ n, d }, h16);
		auto hid = torch::empty({ n, h }, h16);
		auto y = torch::empty({ n, d }, h16);
		ggl_moe_gather_f16(xn16.data_ptr(), srcRows.data_ptr<int>(),
			packed.data_ptr(), (int)n, (int)d, s);
		ggl_moe_grouped_gemm_f16_dev(blk->_gemmCtxLearn, packed.data_ptr(),
			blk->_lw1T.data_ptr(), hid.data_ptr(), offsets.data_ptr<int>(), E, (int)d, (int)h, s);
		ggl_moe_bias_leaky_f16(hid.data_ptr(), blk->_lb1.data_ptr(),
			expertId.data_ptr<int>(), (int)n, (int)h, 0.01f, s);
		ggl_moe_grouped_gemm_f16_dev(blk->_gemmCtxLearn, hid.data_ptr(),
			blk->_lw2T.data_ptr(), y.data_ptr(), offsets.data_ptr<int>(), E, (int)h, (int)d, s);
		auto out16 = torch::zeros({ R, d }, h16);
		ggl_moe_scatter_bias_gate_f16(y.data_ptr(), blk->_lb2.data_ptr(),
			expertId.data_ptr<int>(), gates.data_ptr<float>(),
			srcRows.data_ptr<int>(), out16.data_ptr(), (int)n, (int)d, s);

		// Learn-side load tracking for the aux-free balancing update.
		{
			torch::NoGradGuard ng;
			blk->loadAcc.add_(counts.narrow(0, 0, E).to(blk->loadAcc.scalar_type()));
		}

		ctx->save_for_backward({ xn, logits, packed, hid, y,
			offsets, srcRows, expertId, gates, routerW });
		ctx->saved_data["blk"] = blkPtr;
		return out16.to(torch::kFloat);
	}

	static torch::autograd::tensor_list backward(
		torch::autograd::AutogradContext* ctx, torch::autograd::tensor_list gradOut) {

		auto saved = ctx->get_saved_variables();
		auto xn = saved[0], logits = saved[1], packed = saved[2], hid = saved[3], y = saved[4];
		auto offsets = saved[5], srcRows = saved[6], expertId = saved[7], gates = saved[8];
		auto routerW = saved[9];
		auto* blk = reinterpret_cast<GGL::MoEBlockImpl*>(ctx->saved_data["blk"].toInt());
		void* s = (void*)at::cuda::getCurrentCUDAStream().stream();
		const int64_t R = xn.size(0), d = blk->dim, h = blk->hidden;
		const int E = (int)blk->numExperts, k = (int)blk->topK;
		// n from the SAVED buffers: the forward uses ALIGNED segments, so the slot
		// count is roundup8(R*k) + 8E, not R*k — re-deriving R*k here made every
		// backward kernel under-cover the layout and indexed gdot out of bounds.
		const int64_t n = saved[6].size(0); // srcRows
		auto dev = xn.device();
		auto h16 = torch::TensorOptions().dtype(torch::kHalf).device(dev);
		auto f32 = torch::TensorOptions().dtype(torch::kFloat).device(dev);

		static const int dbg = [] {
			const char* e = std::getenv("GGL_MOE_LEARN_BISECT");
			return (e && *e) ? std::atoi(e) : 0;
		}();
		if (dbg) RG_LOG("[LEARN-BWD] enter R=" << R << " n=" << n);
		auto dOut16 = gradOut[0].to(torch::kHalf).contiguous();

		// Gate-grad first (needs y before dY overwrites nothing — buffers separate).
		auto gdot = torch::empty({ n }, f32);
		ggl_moe_gate_dot_f16(dOut16.data_ptr(), y.data_ptr(), blk->_lb2.data_ptr(),
			expertId.data_ptr<int>(), srcRows.data_ptr<int>(),
			gdot.data_ptr<float>(), (int)n, (int)d, s);

		// dY = gather(dOut)*gate; dB2 = segsum(dY); dH = dY @ W2; leaky bwd;
		// dB1 = segsum(dHpre); dX = dHpre @ W1; dxn_expert = scatter(dX).
		auto dY = torch::empty({ n, d }, h16);
		ggl_moe_gather_scale_f16(dOut16.data_ptr(), srcRows.data_ptr<int>(),
			gates.data_ptr<float>(), dY.data_ptr(), (int)n, (int)d, s);
		auto dB2 = torch::zeros({ (int64_t)E, d }, f32);
		ggl_moe_segment_sum_f16to32(dY.data_ptr(), expertId.data_ptr<int>(),
			dB2.data_ptr<float>(), (int)n, (int)d, s);
		// zeros, NOT empty: the dgrad GEMM writes only rows inside expert segments;
		// the tail beyond offsets[E] would stay uninitialized garbage and the B1
		// segment-sum (which has no pad guard — pads are zero BY VALUE) swept NaNs
		// into expert 0 (grad-parity run 4631409, refNorm 447 vs fast NaN).
		auto dH = torch::zeros({ n, h }, h16);
		ggl_moe_grouped_gemm_f16_dev(blk->_gemmCtxLearn, dY.data_ptr(),
			blk->_lw2.data_ptr(), dH.data_ptr(), offsets.data_ptr<int>(), E, (int)d, (int)h, s);
		ggl_moe_leaky_bwd_f16(dH.data_ptr(), hid.data_ptr(), (int)n, (int)h, 0.01f, s);
		auto dB1 = torch::zeros({ (int64_t)E, h }, f32);
		ggl_moe_segment_sum_f16to32(dH.data_ptr(), expertId.data_ptr<int>(),
			dB1.data_ptr<float>(), (int)n, (int)h, s);
		auto dX = torch::empty({ n, d }, h16);
		ggl_moe_grouped_gemm_f16_dev(blk->_gemmCtxLearn, dH.data_ptr(),
			blk->_lw1.data_ptr(), dX.data_ptr(), offsets.data_ptr<int>(), E, (int)h, (int)d, s);
		auto dxn16 = torch::zeros({ R, d }, h16);
		ggl_moe_scatter_add_f16(dX.data_ptr(), srcRows.data_ptr<int>(),
			dxn16.data_ptr(), (int)n, (int)d, s);

		// Bisect hook (sanitizer triage): 1 = skip wgrads, 2 = also skip dgrad GEMMs.
		static const int bisect = [] {
			const char* e = std::getenv("GGL_MOE_LEARN_BISECT");
			return (e && *e) ? std::atoi(e) : 0;
		}();
		// wgrads: dW1[e] = dHpre_e^T @ X_e -> [h,d]; dW2[e] = dY_e^T @ Hpost_e -> [d,h].
		auto dW1_16 = torch::empty({ (int64_t)E, h, d }, h16);
		if (bisect < 1)
			ggl_moe_grouped_wgrad_f16_dev(blk->_gemmCtxLearn, dH.data_ptr(), packed.data_ptr(),
				dW1_16.data_ptr(), offsets.data_ptr<int>(), E, (int)h, (int)d, (int)n, s);
		else
			dW1_16.zero_();
		auto dW2_16 = torch::empty({ (int64_t)E, d, h }, h16);
		if (bisect < 1)
			ggl_moe_grouped_wgrad_f16_dev(blk->_gemmCtxLearn, dY.data_ptr(), hid.data_ptr(),
				dW2_16.data_ptr(), offsets.data_ptr<int>(), E, (int)d, (int)h, (int)n, s);
		else
			dW2_16.zero_();

		// Gate -> router-logit chain (tiny [n]/[R,E] torch ops, matches eager math:
		// g_i = a_i / S_row with a = sigmoid of the SELECTED logits).
		if (dbg) { torch::cuda::synchronize(); RG_LOG("[LEARN-BWD] dgrad+wgrad done"); }
		// Pad slots (srcRows = -1) are excluded from the gate/router chain.
		auto realMask = srcRows.ge(0);
		auto realIdx = torch::nonzero(realMask).flatten();
		auto srcL = srcRows.index_select(0, realIdx).to(torch::kLong);
		auto expL = expertId.index_select(0, realIdx).to(torch::kLong);
		auto gdotR = gdot.index_select(0, realIdx);
		auto gatesR = gates.index_select(0, realIdx);
		auto selLogit = logits.index({ srcL, expL });                    // [n] fp32
		auto a = torch::sigmoid(selLogit);
		auto S = torch::zeros({ R }, f32).index_add(0, srcL, a);
		auto Srow = S.index_select(0, srcL).clamp_min(1e-9);
		auto dotRow = torch::zeros({ R }, f32).index_add(0, srcL, gdotR * gatesR);
		auto dA = gdotR / Srow - dotRow.index_select(0, srcL) / Srow;
		auto dLogitSel = dA * a * (1.0 - a);
		auto dLogits = torch::zeros_like(logits);
		dLogits.index_put_({ srcL, expL }, dLogitSel, /*accumulate=*/true);
		if (dbg) { torch::cuda::synchronize(); RG_LOG("[LEARN-BWD] gate chain done"); }
		auto dRouterW = torch::mm(dLogits.t(), xn);                      // [E,d] fp32
		auto dxn = dxn16.to(torch::kFloat) + torch::mm(dLogits, routerW);

		return { dxn, dRouterW,
			dW1_16.to(torch::kFloat), dB1, dW2_16.to(torch::kFloat), dB2,
			torch::Tensor(), torch::Tensor() };
	}
};

} // namespace

torch::Tensor GGL::MoERoutedFFNApply(GGL::MoEBlockImpl* blk, torch::Tensor xn) {
	return MoERoutedFFN::apply(xn, blk->routerW, blk->expertW1, blk->expertB1,
		blk->expertW2, blk->expertB2, blk->routerBias, (int64_t)(intptr_t)blk);
}
#endif

float GGL::MoEBlockImpl::UpdateRouterBias(float gamma) {
	NoGradGuard ng;
	auto load = loadAcc.to(torch::kFloat32);
	float total = load.sum().item<float>();
	if (total < 1.f)
		return 1.f;
	auto frac = load / total;
	auto mean = 1.f / (float)numExperts;
	// DSv3 aux-free rule: overloaded experts' selection bias down, underloaded up.
	routerBias.add_(torch::sign(mean - frac).to(routerBias.dtype()), gamma);
	loadAcc.mul_(0.f); // reset per update window
	auto p = frac.clamp_min(1e-12f);
	float entropy = -(p * p.log()).sum().item<float>() / std::log((float)numExperts);
	return entropy;
}
