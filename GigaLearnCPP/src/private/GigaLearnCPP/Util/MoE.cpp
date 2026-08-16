#include "MoE.h"
#include "Models.h"
#include <torch/cuda.h>

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
	RG_LOG("MoE selftest OK");
	return 0;
}

GGL::MoEBlockImpl::MoEBlockImpl(int64_t dim, int64_t hidden, int64_t numExperts, int64_t topK)
	: dim(dim), hidden(hidden), numExperts(numExperts), topK(topK) {
	reset();
}

void GGL::MoEBlockImpl::reset() {
	if (dim <= 0)
		return;
	// Kaiming-ish init scaled like the dense trunk layers; router near-zero so early
	// selection is ~uniform and the balancing bias starts in control.
	const double s1 = std::sqrt(2.0 / (double)dim);
	const double s2 = std::sqrt(2.0 / (double)hidden);
	routerW = register_parameter("routerW", torch::randn({ numExperts, dim }) * 0.01);
	expertW1 = register_parameter("expertW1", torch::randn({ numExperts, hidden, dim }) * s1);
	expertB1 = register_parameter("expertB1", torch::zeros({ numExperts, hidden }));
	expertW2 = register_parameter("expertW2", torch::randn({ numExperts, dim, hidden }) * s2);
	expertB2 = register_parameter("expertB2", torch::zeros({ numExperts, dim }));
	sharedW1 = register_parameter("sharedW1", torch::randn({ hidden, dim }) * s1);
	sharedB1 = register_parameter("sharedB1", torch::zeros({ hidden }));
	sharedW2 = register_parameter("sharedW2", torch::randn({ dim, hidden }) * s2);
	sharedB2 = register_parameter("sharedB2", torch::zeros({ dim }));
	routerBias = register_buffer("routerBias", torch::zeros({ numExperts }));
	loadAcc = register_buffer("loadAcc", torch::zeros({ numExperts }));
	ln = register_module("ln", torch::nn::LayerNorm(torch::nn::LayerNormOptions({ dim })));
}

torch::Tensor GGL::MoEBlockImpl::forward(torch::Tensor x) {
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
