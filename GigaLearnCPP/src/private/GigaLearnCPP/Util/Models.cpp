#include "Models.h"
#include "MoE.h"
#include "../PPO/CudaGraphPolicy.h"

#include <torch/csrc/api/include/torch/serialize.h>
#include <torch/csrc/api/include/torch/nn/utils/convert_parameters.h>
#include <torch/nn/modules/normalization.h>

#ifdef RG_CUDA_SUPPORT
#include <c10/cuda/CUDAStream.h>
#endif

// See MoE.h: the MoE fast path keys its transposed-weight caches on this counter.
std::atomic<uint64_t> GGL::g_halfRefreshEpoch{ 0 };

GGL::Model::Model(
	const char* modelName,
	ModelConfig config,
	torch::Device device) : 
	modelName(modelName), device(device), seq({}), seqHalf({}), config(config) {

	if (!config.IsValid())
		RG_ERR_CLOSE("Failed to create model \"" << modelName << "\" with invalid config");

	if (config.addResiduals) {
		// The skip adds a block's input to its output, so every hidden layer must share a width.
		for (int size : config.layerSizes)
			if (size != config.layerSizes[0])
				RG_ERR_CLOSE("Model \"" << modelName << "\": addResiduals requires uniform layerSizes, got "
					<< size << " among " << config.layerSizes[0] << "-wide layers");
	}

	// MoE trunk path (see PartialModelConfig::moeBlocks): embed -> MoE blocks -> LN.
	if (config.moeBlocks > 0) {
		const int64_t d = (int64_t)config.layerSizes[0];
		seq->push_back(torch::nn::Linear(config.numInputs, d));
		if (config.addLayerNorm)
			seq->push_back(torch::nn::LayerNorm(torch::nn::LayerNormOptions({ d })));
		AddActivationFunc(seq, config.activationType);
		for (int b = 0; b < config.moeBlocks; b++)
			seq->push_back(GGL::MoEBlock((int64_t)d, (int64_t)config.moeHidden,
				(int64_t)config.moeExperts, (int64_t)config.moeTopK));
		seq->push_back(torch::nn::LayerNorm(torch::nn::LayerNormOptions({ d })));
		if (config.addOutputLayer)
			seq->push_back(torch::nn::Linear(d, config.numOutputs));
		else
			this->config.numOutputs = (int)d;
		register_module("seq", seq);
		seq->to(device);
		optim = MakeOptimizer(config.optimType, this->parameters(), 0);
		return;
	}

	int lastSize = config.numInputs;
	int blockRemaining = 0;   // hidden layers left in the currently-open residual block
	int blockStartModule = -1;
	const int numLayers = (int)config.layerSizes.size();
	for (int i = 0; i < numLayers; i++) {

		// Open a 2-layer residual block at layer i. Never at the stem (i=0, which changes width
		// from numInputs), and only when layer i+1 exists to pair with - a trailing odd layer
		// stays plain, so {W} and {W,W} are unaffected by addResiduals.
		if (config.addResiduals && blockRemaining == 0 && i >= 1 && (i + 1) < numLayers) {
			blockRemaining = 2;
			blockStartModule = (int)seq->size();
		}

		seq->push_back(torch::nn::Linear(lastSize, config.layerSizes[i]));
		if (config.addLayerNorm)
			seq->push_back(torch::nn::LayerNorm(torch::nn::LayerNormOptions({(int64_t)config.layerSizes[i]})));
		lastSize = config.layerSizes[i];

		// Close the block on its SECOND layer, before that layer's activation is pushed:
		// the recorded span ends at the LayerNorm, so Forward adds the skip and the trailing
		// activation then sees the sum (ResNet-v1 ordering).
		if (blockRemaining > 0 && --blockRemaining == 0)
			residualSpans.push_back({ blockStartModule, (int)seq->size() - 1 });

		AddActivationFunc(seq, config.activationType);
	}
	
	if (config.addOutputLayer) {
		seq->push_back(torch::nn::Linear(lastSize, config.numOutputs));
	} else {
		// Write the MEMBER, not the shadowing parameter: the member was already copied in the
		// init list, so assigning `config.numOutputs` here was silently discarded and every
		// no-output-layer model reported numOutputs = 0 to readers of model->config.
		this->config.numOutputs = (int)config.layerSizes.back();
	}

	register_module("seq", seq);
	seq->to(device);
	optim = MakeOptimizer(config.optimType, this->parameters(), 0);
}

torch::Tensor GGL::ForwardSeqModule(const std::shared_ptr<torch::nn::Module>& mod, torch::Tensor x) {
	if (auto lin = std::dynamic_pointer_cast<torch::nn::LinearImpl>(mod))
		return lin->forward(x);
	if (auto moe = std::dynamic_pointer_cast<GGL::MoEBlockImpl>(mod))
		return moe->forward(x);
	if (auto ln = std::dynamic_pointer_cast<torch::nn::LayerNormImpl>(mod))
		return ln->forward(x);
	if (auto act = std::dynamic_pointer_cast<torch::nn::LeakyReLUImpl>(mod))
		return act->forward(x);
	if (auto act = std::dynamic_pointer_cast<torch::nn::ReLUImpl>(mod))
		return act->forward(x);
	if (auto act = std::dynamic_pointer_cast<torch::nn::SigmoidImpl>(mod))
		return act->forward(x);
	if (auto act = std::dynamic_pointer_cast<torch::nn::TanhImpl>(mod))
		return act->forward(x);

	RG_ERR_CLOSE("ForwardSeqModule: unexpected module type in model seq");
	return x;
}

// Walk a flat seq applying the recorded residual spans. Only used when residualSpans is
// non-empty; otherwise callers take Sequential's own fused forward.
static torch::Tensor ForwardResidual(
	torch::nn::Sequential& seq, const std::vector<std::pair<int, int>>& spans, torch::Tensor x) {

	std::vector<torch::Tensor> saved(spans.size());
	for (int i = 0; i < (int)seq->size(); i++) {

		for (int s = 0; s < (int)spans.size(); s++)
			if (spans[s].first == i)
				saved[s] = x;

		x = GGL::ForwardSeqModule(seq->ptr(i), x);

		for (int s = 0; s < (int)spans.size(); s++)
			if (spans[s].second == i && saved[s].defined())
				x = x + saved[s];
	}
	return x;
}

void GGL::Model::RefreshHalfCache() {
		if (_seqHalfOutdated) {
			_seqHalfOutdated = false;
			// GGL_CONSUME_TIMERS: refresh cost breakdown (the 1B MoE read 26s/iter here)
			static const bool refTimers = [] {
				const char* e = std::getenv("GGL_CONSUME_TIMERS");
				return e && *e && std::string(e) != "0";
			}();
			auto tRef0 = std::chrono::steady_clock::now();
			const bool firstBuild = (seqHalf->size() == 0);

			if (seqHalf->size() == 0) {
				// clone(device), NOT clone(): Cloneable's reset() creates fresh CPU tensors,
				// so a device-less clone of a CUDA model round-trips every param through the
				// host — and the CPU-side fp32->fp16 cast of the 1B MoE trunk took 26 SECONDS
				// per build (job 4630865). With the device passed, the whole build is on-GPU.
				for (auto& mod : *seq)
					seqHalf->push_back(mod.clone(device));
				seqHalf->to(RG_HALFPERC_TYPE, true);

				// GGL_FLAT_HALF (default on): re-point every seqHalf param at a view of ONE
				// flat half buffer, so the per-step refresh below is one cat + one casting
				// copy_ instead of a .to() temp + copy_ per param. Bit-identical fp16
				// weights — this is an op-count fix for the ppc64le dispatch tax (measured
				// 0.36s/iter rebuilding the value family at ~0.2-0.5ms per tensor op;
				// x86/desktop never noticed). GGL_FLAT_HALF=0 restores the per-param path.
				static const bool flatHalf = [] {
					const char* e = std::getenv("GGL_FLAT_HALF");
					return !(e && *e && std::string(e) == "0");
				}();
				if (flatHalf) {
					auto halfParams = seqHalf->parameters();
					int64_t total = 0;
					for (auto& p : halfParams)
						total += p.numel();
					if (total > 0) {
						_flatHalfBuf = torch::empty({ total },
							torch::TensorOptions().dtype(RG_HALFPERC_TYPE).device(device));
						int64_t off = 0;
						for (auto& p : halfParams) {
							auto slice = _flatHalfBuf.narrow(0, off, p.numel()).view(p.sizes());
							slice.copy_(p, true);
							p.set_data(slice);
							off += p.numel();
						}
					}
				}
			}
			// Buffers are NOT covered by the flat param store — the MoE router's selection
			// bias lives in a buffer and drifts every learn pass; without this the collect
			// side would route on the birth bias forever.
			{
				auto fromBufs = seq->buffers();
				auto toBufs = seqHalf->buffers();
				for (size_t i = 0; i < fromBufs.size() && i < toBufs.size(); i++)
					toBufs[i].copy_(fromBufs[i], true);
			}
			if (_flatHalfBuf.defined()) {
				// GROUPED gather+cast into the flat half store. A single
				// parameters_to_vector materializes a full fp32 temp — 4.2GB for the 1B
				// MoE trunk, which OOM'd the 32GB V100 (job 4630863). Groups bound the
				// temp to ~tens of MB while keeping op count ~2 per GROUP, not per param.
				auto params = seq->parameters();
				constexpr size_t GROUP = 48;
				int64_t off = 0;
				size_t i = 0;
				while (i < params.size()) {
					size_t j = std::min(i + GROUP, params.size());
					std::vector<torch::Tensor> flat;
					flat.reserve(j - i);
					int64_t n = 0;
					for (size_t k = i; k < j; k++) {
						flat.push_back(params[k].flatten());
						n += params[k].numel();
					}
					_flatHalfBuf.narrow(0, off, n).copy_(torch::cat(flat, 0), true);
					off += n;
					i = j;
				}
			} else {
				auto fromParams = seq->parameters();
				auto toParams = seqHalf->parameters();
				for (int i = 0; i < fromParams.size(); i++) {
					// copy_ casts across dtypes directly; the old .to() temp doubled the
					// op count and allocated every refresh.
					toParams[i].copy_(fromParams[i], true);
				}
			}
			if (refTimers) {
				double ms = std::chrono::duration<double, std::milli>(
					std::chrono::steady_clock::now() - tRef0).count();
				fprintf(stderr, "[HALFREFRESH] model=%s firstBuild=%d nParams=%zu ms=%.1f\n",
					modelName, (int)firstBuild, seq->parameters().size(), ms);
			}
			// The fp16 weights just changed in place; MoE fast-path transposed caches
			// key on this (rebuild once per refresh, not per tick).
			g_halfRefreshEpoch.fetch_add(1, std::memory_order_release);
		}
}

torch::Tensor GGL::Model::Forward(torch::Tensor input, bool halfPrec, bool keepHalf) {

	if (torch::GradMode::is_enabled())
		halfPrec = false;

	if (halfPrec) {

		// (Their branch inlined an older per-param refresh here; RefreshHalfCache is
		// the superset — flat-half buffer, grouped cast, MoE buffer copies, epoch bump.)
		RefreshHalfCache();

		auto halfParams = seqHalf->parameters();
		const auto halfType = halfParams.empty() ? RG_HALFPERC_TYPE : halfParams[0].scalar_type();
		if (input.scalar_type() != halfType)
			input = input.to(halfType);
		auto halfOutput = residualSpans.empty()
			? seqHalf->forward(input)
			: ForwardResidual(seqHalf, residualSpans, input);
		return keepHalf ? halfOutput : halfOutput.to(torch::kFloat);
	} else {
		return residualSpans.empty()
			? seq->forward(input)
			: ForwardResidual(seq, residualSpans, input);
	}
}

GGL::Model::~Model() {
	PolicyCudaGraph::Release(this);
}

// Get sizes of all parameters in a sequence
std::vector<uint64_t> GetSeqSizes(torch::nn::Sequential& seq) {
	std::vector<uint64_t> result = {};

	for (int i = 0; i < seq->size(); i++)
		for (auto param : seq[i]->parameters())
			result.push_back(param.numel());

	return result;
}

void GGL::Model::SetOptimLR(float newLR) {
	SetOptimizerLR(optim, config.optimType, newLR);
}

void GGL::Model::StepOptim() {
	optim->step();
	optim->zero_grad();
	_seqHalfOutdated = true;
}

bool GGL::Model::VerifySavedWeights(std::filesystem::path folder) const {
	// See the header note. Compares every named parameter in the file against the live
	// module. Deliberately exact (equal(), not allclose()): serialization is a byte copy,
	// so any difference at all means the write did not capture the weights we hold.
	try {
		std::filesystem::path path = GetSavePath(folder);
		if (!std::filesystem::exists(path))
			return false;
		RG_NO_GRAD;
		// Load the file exactly the way the real loader will, into a clone of this module,
		// and diff parameter-by-parameter. Reading the archive by flat named_parameters()
		// key does NOT work and silently fails everything: torch::save(Sequential) nests
		// each submodule in its own sub-archive, so "0.weight" does not resolve at the top
		// level. That mistake made a healthy CPU smoke report every save as corrupt, which
		// would have stopped checkpointing entirely on the live run.
		auto clonedImpl = std::dynamic_pointer_cast<torch::nn::SequentialImpl>(seq->clone());
		if (!clonedImpl)
			return false;
		torch::nn::Sequential reloaded(clonedImpl);
		torch::load(reloaded, path.string());

		auto live = seq->parameters();
		auto disk = reloaded->parameters();
		if (live.size() != disk.size() || live.empty())
			return false;
		for (size_t i = 0; i < live.size(); i++) {
			if (live[i].sizes() != disk[i].sizes())
				return false;
			if (!torch::equal(live[i].detach().to(torch::kCPU, torch::kFloat32),
					disk[i].detach().to(torch::kCPU, torch::kFloat32)))
				return false;
		}
		return true;
	} catch (const std::exception& e) {
		RG_LOG("VerifySavedWeights: exception during verify: " << e.what());
		return false;
	} catch (...) {
		return false;
	}
}

void GGL::Model::Save(std::filesystem::path folder, bool saveOptim) {
	std::filesystem::path path = GetSavePath(folder);
	auto streamOut = std::ofstream(path, std::ios::binary);
	torch::save(seq, streamOut);
	// Flush and close BEFORE anyone reads this back (the verifier does, immediately).
	// The stream was previously left to its destructor, so a read-after-write saw a
	// partially-flushed file.
	streamOut.flush();
	streamOut.close();

	if (saveOptim) {
		torch::serialize::OutputArchive optimArchive;
		optim->save(optimArchive);
		optimArchive.save_to(GetOptimSavePath(folder).string());
	}
}

void GGL::Model::Load(std::filesystem::path folder, bool allowNotExist, bool loadOptim) {
	std::filesystem::path path = GetSavePath(folder);

	if (!std::filesystem::exists(path)) {
		if (allowNotExist) {
			RG_LOG("Warning: Model \"" << modelName << "\" does not exist in " << folder << " and will be reset");
			return;
		} else {
			RG_ERR_CLOSE("Model \"" << modelName << "\" does not exist in " << folder);
		}
	}

	auto streamIn = std::ifstream(path, std::ios::binary);
	streamIn >> std::noskipws;

	if (!streamIn.good())
		RG_ERR_CLOSE("Failed to load from " << path << ", file does not exist or can't be accessed");

	auto sizesBefore = GetSeqSizes(seq);

	try {
		torch::load(this->seq, streamIn, device);
	} catch (std::exception& e) {
		RG_ERR_CLOSE(
			"Failed to load model \"" << modelName << ", checkpoint may be corrupt or of different model arch.\n" <<
			"Exception: " << e.what()
		);
	}

	// Torch will happily load in a model of a totally different size, then we will crash when we try to use it
	// So we need to manually check if it is the same size
	auto sizesAfter = GetSeqSizes(seq);
	if (!std::equal(sizesBefore.begin(), sizesBefore.end(), sizesAfter.begin(), sizesAfter.end())) {

		{
			std::stringstream stream;
			stream << "Saved model has different size than current model, cannot load model from " << path << ":\n";

			for (int i = 0; i < 2; i++) {
				stream << " > " << (i ? "Saved model:   [ " : "Current model: [ ");
				for (uint64_t size : (i ? sizesAfter : sizesBefore))
					stream << size << ' ';

				stream << " ]";
				if (i == 0)
					stream << ",\n";
			}

			RG_ERR_CLOSE(stream.str());
		}
	}

	/////////////////////////////

	// GGL_FRESH_OPTIM: skip optimizer-state load on resume. Muon keys its momentum by
	// process-local TensorImpl pointers, so deserialized state can never match the new
	// process's params — the loaded buffers sit ORPHANED next to freshly-allocated ones
	// (~= a full extra weights-worth of GPU memory at MoE scale, the resumed-hop OOM).
	// Momentum resets on resume under this flag: a mild warmup hiccup, not a semantics
	// change (grads/weights are exact).
	static const bool freshOptim = [] {
		const char* e = std::getenv("GGL_FRESH_OPTIM");
		return e && *e && std::string(e) != "0";
	}();
	if (loadOptim && !freshOptim) {
		std::filesystem::path optimPath = GetOptimSavePath(folder);

		if (std::filesystem::exists(optimPath)) {
			std::ifstream testStream = std::ifstream(optimPath, std::istream::ate | std::ios::binary);
			if (testStream.tellg() > 0) {
				torch::serialize::InputArchive optimArchive;
				optimArchive.load_from(optimPath.string(), device);
				optim->load(optimArchive);
			} else {
				RG_LOG("WARNING: Saved optimizer at " << optimPath << " is empty, optimizer will be reset");
			}
		} else {
			RG_LOG("WARNING: No optimizer found at " << optimPath << ", optimizer will be reset");
		}
	}

	// The bf16 inference mirror must be rebuilt from the weights just loaded. StepOptim sets this
	// and so does the collect-snapshot sync — but Load did NOT, so any Model that had already
	// served one half-precision forward kept serving its PRE-LOAD weights forever, silently.
	// Boot resume is unaffected (load precedes the first forward). The live victim is render
	// mode's checkpoint hot-swap, which loads into already-used models: the viewer would keep
	// showing the OLD policy while logging the new checkpoint's timestep.
	_seqHalfOutdated = true;
}

torch::Tensor GGL::Model::CopyParams() const {
	return torch::nn::utils::parameters_to_vector(parameters()).cpu();
}

static GGL::Dist::Session::Stream GGLCurrentCudaStream(const torch::Tensor& t) {
#ifdef RG_CUDA_SUPPORT
	if (t.is_cuda())
		return at::cuda::getCurrentCUDAStream(t.device().index()).stream();
#endif
	(void)t;
	return nullptr;
}

static void GGLCollectGrads(torch::nn::Module& m, const char* name, std::vector<GGL::Dist::Session::GradRef>& refs) {
	// EP (MOE_SPEED.md Stage 2): expert params are OWNER-LOCAL — the owner's
	// backward already produced the complete gradient for its slice and exact
	// zeros elsewhere, so allreducing them would be wrong (it would average the
	// owner's real grad against other ranks' zeros) as well as wasteful. They are
	// reconciled by the post-step owner broadcast instead.
	const bool epOn = GGL::MoEExpertParallelOn();
	const auto& epParams = GGL::MoEExpertParams();
	for (auto& p : m.parameters()) {
		if (!p.requires_grad())
			continue;
		if (epOn) {
			bool isExpert = false;
			for (auto& ep : epParams)
				if (ep.is_same(p)) { isExpert = true; break; }
			if (isExpert)
				continue;
		}
		if (!p.grad().defined())
			p.mutable_grad() = torch::zeros_like(p);
		auto g = p.grad();
		if (!g.is_cuda() || g.scalar_type() != torch::kFloat)
			RG_ERR_CLOSE("AllReduceGrads: expected CUDA float32 grad on " << name);
		if (!g.is_contiguous()) {
			p.mutable_grad() = g.contiguous();
			g = p.grad();
		}
		refs.push_back({ g.data_ptr<float>(), (size_t)g.numel() });
	}
}

void GGL::Model::BroadcastParameters(Dist::Session* dist) {
	if (!dist || !dist->distributed())
		return;
	RG_NO_GRAD;
	for (auto& p : parameters()) {
		if (p.is_cuda() && p.scalar_type() == torch::kFloat) {
			auto t = p.contiguous();
			dist->bcast_device(t.data_ptr<float>(), (size_t)t.numel(), 0, GGLCurrentCudaStream(t));
			if (!p.is_same(t))
				p.copy_(t);
		} else {
			auto cpu = p.contiguous().cpu();
			dist->bcast_host(cpu.data_ptr(), (size_t)cpu.numel() * cpu.element_size(), 0);
			p.copy_(cpu);
		}
	}
	_seqHalfOutdated = true;
}

void GGL::Model::AllReduceGrads(Dist::Session* dist) {
	if (!dist || !dist->distributed())
		return;
	std::vector<Dist::Session::GradRef> refs;
	GGLCollectGrads(*this, modelName, refs);
	if (refs.empty())
		return;
	dist->allreduce_avg_grads(refs, GGLCurrentCudaStream(parameters()[0]));
}

void GGL::ModelSet::BroadcastParameters(Dist::Session* dist) {
	if (!dist || !dist->distributed())
		return;
	for (Model* model : *this)
		model->BroadcastParameters(dist);
}

void GGL::ModelSet::AllReduceGrads(Dist::Session* dist, bool includeExempt) {
	if (!dist || !dist->distributed())
		return;
	std::vector<Dist::Session::GradRef> refs;
	torch::Tensor streamSrc;
	for (Model* model : *this) {
		if (!includeExempt && model->groupStepExempt)
			continue;
		GGLCollectGrads(*model, model->modelName, refs);
		if (!streamSrc.defined()) {
			auto ps = model->parameters();
			if (!ps.empty())
				streamSrc = ps[0];
		}
	}
	if (refs.empty())
		return;
	dist->allreduce_avg_grads(refs, streamSrc.defined() ? GGLCurrentCudaStream(streamSrc) : nullptr);
}

// EP (MOE_SPEED.md Stage 2): after the optimizer step, every owner broadcasts its
// expert slices so all learners hold a complete net for publishing/checkpointing.
// COARSE and at the ALLREDUCE LANE by design: per-param collectives inside the
// optimizer step deadlocked against the collector publish comm (the GGL_MUON_SHARD
// async verdict). Non-owned slices may carry stray momentum-driven updates; the
// broadcast clobbers them, so the owner's value is authoritative either way.
// v2 (task #4) drops replication entirely and gathers from owners at publish time.
void GGL::ModelSet::ReplicateExpertSlices(Dist::Session* dist) {
	if (!dist || !GGL::MoEExpertParallelOn())
		return;
	const int nL = dist->group_world();
	if (nL <= 1)
		return;
	const auto& params = GGL::MoEExpertParams();
	torch::NoGradGuard ng;
	// Deterministic order: param-major, then owner — every rank issues the exact
	// same broadcast sequence.
	for (auto& p : params) {
		if (!p.defined() || p.size(0) <= 0)
			continue;
		const int E = (int)p.size(0);
		const int per = E / nL;
		if (per <= 0)
			continue;
		for (int o = 0; o < nL; o++) {
			const int st = o * per;
			const int cn = (o == nL - 1) ? (E - st) : per;
			auto slice = p.narrow(0, st, cn).contiguous();
			dist->bcast_device_group(slice.data_ptr<float>(), (size_t)slice.numel(), o);
			if (dist->group_rank() != o)
				p.narrow(0, st, cn).copy_(slice);
		}
	}
}

void GGL::ModelSet::StepOptimsSharded(Dist::Session* dist) {
	if (!dist || !dist->distributed()) {
		StepOptims();
		return;
	}
	// group_*, not world/rank: under APPO async routing the collectives run on the
	// learner group, and ownership must partition over the SAME set (see Session.h).
	const int world = dist->group_world(), rank = dist->group_rank();

	// (param, owner) for every NS-eligible 2D param, in the exact order Muon::step will
	// count them. The predicate mirrors Muon::step's and is evaluated PRE-step, while the
	// allreduced (zero-filled) grads are still defined — StepOptim's zero_grad would
	// otherwise change the eligible set between counting and broadcasting.
	std::vector<std::pair<torch::Tensor, int>> shard2D;
	int64_t counter = 0;
	for (Model* model : *this) {
		if (model->groupStepExempt)
			continue;
		Muon* muon = dynamic_cast<Muon*>(model->optim);
		if (!muon) {
			model->StepOptim();
			continue;
		}
		muon->shardRank = rank;
		muon->shardWorld = world;
		muon->shardCounter = counter;
		for (auto& group : muon->param_groups())
			for (auto& param : group.params()) {
				auto g = param.grad();
				// Predicate MUST mirror Muon::step's NS-eligible test (2D matrices and
				// 3D MoE expert stacks) or the shard counters misalign.
				if (g.defined() && ((g.dim() == 2 && g.size(0) > 1 && g.size(1) > 1)
					|| (g.dim() == 3 && g.size(1) > 1 && g.size(2) > 1)))
					shard2D.emplace_back(param, (int)(counter++ % (int64_t)world));
			}
		model->StepOptim();
		// If this fires, the enumeration above no longer mirrors Muon::step's internal
		// count and ownership is misaligned — params would silently stay stale.
		RG_ASSERT(muon->shardCounter == counter);
		muon->shardWorld = 1;
	}

	for (auto& pr : shard2D) {
		torch::Tensor& param = pr.first;
		RG_ASSERT(param.is_cuda() && param.scalar_type() == torch::kFloat && param.is_contiguous());
		dist->bcast_device(param.data_ptr<float>(), (size_t)param.numel(), pr.second,
			GGLCurrentCudaStream(param));
	}
}
