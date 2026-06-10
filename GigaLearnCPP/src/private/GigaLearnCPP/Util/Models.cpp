#include "Models.h"

#include <torch/csrc/api/include/torch/serialize.h>
#include <torch/csrc/api/include/torch/nn/utils/convert_parameters.h>
#include <torch/nn/modules/normalization.h>

GGL::Model::Model(
	const char* modelName,
	ModelConfig config,
	torch::Device device) : 
	modelName(modelName), device(device), seq({}), seqHalf({}), config(config) {

	if (!config.IsValid())
		RG_ERR_CLOSE("Failed to create model \"" << modelName << "\" with invalid config");

	int lastSize = config.numInputs;
	for (int i = 0; i < config.layerSizes.size(); i++) {
		seq->push_back(torch::nn::Linear(lastSize, config.layerSizes[i]));
		if (config.addLayerNorm)
			seq->push_back(ManualLayerNorm((int64_t)config.layerSizes[i]));
		lastSize = config.layerSizes[i];
		AddActivationFunc(seq, config.activationType);
	}
	
	if (config.addOutputLayer) {
		seq->push_back(torch::nn::Linear(lastSize, config.numOutputs));
	} else {
		config.numOutputs = config.layerSizes.back();
	}

	register_module("seq", seq);
	seq->to(device);
	optim = MakeOptimizer(config.optimType, this->parameters(), 0);
}

torch::Tensor GGL::Model::Forward(torch::Tensor input, bool halfPrec) {

	if (torch::GradMode::is_enabled())
		halfPrec = false;

	if (halfPrec) {

		if (_seqHalfOutdated) {
			_seqHalfOutdated = false;

			if (seqHalf->size() == 0) {
				for (auto& mod : *seq)
					seqHalf->push_back(mod.clone());
				seqHalf->to(RG_HALFPERC_TYPE, true);
			} else {
				auto fromParams = seq->parameters();
				auto toParams = seqHalf->parameters();
				for (int i = 0; i < fromParams.size(); i++) {
					auto scaledParams = fromParams[i].to(RG_HALFPERC_TYPE, true);
					toParams[i].copy_(scaledParams, true);
				}
			}
		}
		
		auto halfInput = input.to(RG_HALFPERC_TYPE);
		auto halfOutput = seqHalf->forward(halfInput);
		return halfOutput.to(torch::kFloat);
	} else {
		return seq->forward(input);
	}
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
	// --- Non-finite gradient guard ---
	// clip_grad_norm_ does NOT protect the weights from NaN/Inf gradients: when the total
	// grad norm is non-finite the clip scale becomes 0/Inf or NaN, which launders the bad
	// gradient straight into optim->step(), turning every weight NaN. That corruption only
	// surfaces later, on the next rollout, as torch::multinomial's device-side assert
	// (_assert_async_cuda_kernel) aborting the GPU queue. Standard PPO/AMP practice is to
	// skip any optimizer step whose gradients aren't all finite, so one poisoned minibatch
	// can't kill the whole run. The check is one reduction + one host sync per step.
	torch::Tensor finiteFlag;
	for (auto& p : this->parameters()) {
		if (!p.grad().defined())
			continue;
		torch::Tensor f = torch::isfinite(p.grad()).all();
		finiteFlag = finiteFlag.defined() ? finiteFlag.logical_and(f) : f;
	}
	bool gradsFinite = !finiteFlag.defined() || finiteFlag.item<bool>();

	if (gradsFinite) {
		optim->step();
	} else {
		// Drop the bad gradient (handled by zero_grad below) and leave weights untouched.
		nanGradSkips++;
		if (nanGradSkips == 1 || nanGradSkips % 100 == 0)
			RG_LOG("WARNING: Model::StepOptim() skipped a non-finite gradient (model '" << modelName
				<< "', this model's total skips: " << nanGradSkips << "). The policy/critic is producing NaN/Inf"
				<< " -- inspect reward/advantage magnitudes; weights were left intact.");

		// Attribution dump for the first few skips: which parameters carry non-finite grads?
		// If ALL of them do, one non-finite total norm in clip_grad_norm_ laundered every
		// grad (the overflow could be anywhere in backward); if only SOME do, the producing
		// op is local to those layers (e.g. a backend's fused LayerNorm backward kernel).
		if (nanGradSkips <= 3) {
			std::stringstream dump;
			int badCount = 0, totalCount = 0;
			for (auto& pair : this->named_parameters()) {
				if (!pair.value().grad().defined())
					continue;
				totalCount++;
				if (!torch::isfinite(pair.value().grad()).all().item<bool>()) {
					badCount++;
					dump << " " << pair.key();
				}
			}
			RG_LOG("  > '" << modelName << "' non-finite grads in " << badCount << "/" << totalCount << " params:" << dump.str());
		}
	}

	optim->zero_grad();
	_seqHalfOutdated = true;
}

void GGL::Model::Save(std::filesystem::path folder, bool saveOptim) {
	std::filesystem::path path = GetSavePath(folder);
	auto streamOut = std::ofstream(path, std::ios::binary);
	torch::save(seq, streamOut);

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

	/////////////////////////////

	if (loadOptim) {
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
}

torch::Tensor GGL::Model::CopyParams() const {
	return torch::nn::utils::parameters_to_vector(parameters()).cpu();
}

// ── QuasimetricCritic (Python QCritic port) ─────────────────────────────────

GGL::QuasimetricCritic::QuasimetricCritic(
	const char* _modelName,
	int obs_dim, int _action_dim, int goal_dim,
	const std::vector<int>& hiddenSizes, int _repr_dim,
	ModelActivationType activation, bool addLayerNorm,
	float _tau, float _var_reg, float _infonce_penalty,
	ModelOptimType optimType,
	torch::Device device) :
	Model(), tau(_tau), var_reg(_var_reg), infonce_penalty(_infonce_penalty), action_dim(_action_dim), repr_dim(_repr_dim) {

	this->modelName = _modelName;
	this->device = device;
	// The base Model uses config.optimType in SetOptimLR()/StepOptim(); keep it in sync
	// with the optimizer we actually build below (default-constructed config is ADAM).
	this->config.optimType = optimType;

	// phi and psi share the architecture; only the input dim differs. Output is repr_dim
	// (no activation) and is L2-normalized at scoring time, which is what makes the cosine
	// quasimetric valid -- so both towers must end at the same repr_dim.
	auto buildTower = [&](int inDim) {
		torch::nn::Sequential seq;
		int last = inDim;
		for (int h : hiddenSizes) {
			seq->push_back(torch::nn::Linear(last, h));
			if (addLayerNorm)
				seq->push_back(ManualLayerNorm((int64_t)h));
			AddActivationFunc(seq, activation);
			last = h;
		}
		seq->push_back(torch::nn::Linear(last, repr_dim));
		return seq;
	};

	phi_net = buildTower(obs_dim + _action_dim);
	psi_net = buildTower(goal_dim);

	register_module("phi_net", phi_net);
	register_module("psi_net", psi_net);

	this->to(device);
	optim = MakeOptimizer(optimType, this->parameters(), 0);
}

std::pair<torch::Tensor, torch::Tensor> GGL::QuasimetricCritic::embed(
	torch::Tensor obs, torch::Tensor actions, torch::Tensor goals) {

	auto sa_input = torch::cat({obs, actions.to(torch::kFloat32).to(obs.device())}, -1);
	auto sa_raw = phi_net->forward(sa_input);
	auto g_raw = psi_net->forward(goals.to(torch::kFloat32));

	auto sa_emb = torch::nn::functional::normalize(sa_raw,
		torch::nn::functional::NormalizeFuncOptions().p(2).dim(-1));
	auto g_emb = torch::nn::functional::normalize(g_raw,
		torch::nn::functional::NormalizeFuncOptions().p(2).dim(-1));

	return {sa_emb, g_emb};
}

torch::Tensor GGL::QuasimetricCritic::forward(torch::Tensor obs, torch::Tensor actions, torch::Tensor goals) {
	auto [sa_emb, g_emb] = embed(obs, actions, goals);
	return (sa_emb * g_emb).sum(-1) / tau;
}

torch::Tensor GGL::QuasimetricCritic::score_q(torch::Tensor obs, torch::Tensor actions, torch::Tensor goals) {
	return forward(obs, actions, goals);
}

torch::Tensor GGL::QuasimetricCritic::infonce_loss(
	torch::Tensor obs, torch::Tensor actions, torch::Tensor goals,
	float tau_override,
	torch::Tensor sampleWeights,
	torch::Tensor* outRaw, torch::Tensor* outReg1, torch::Tensor* outReg2) {

	float t = tau_override > 0 ? tau_override : tau;

	auto [sa, g] = embed(obs, actions, goals);

	int64_t B = sa.size(0);
	auto dev = obs.device();

	auto logits = torch::matmul(sa, g.transpose(0, 1)) / t;

	auto arange = torch::arange(B, torch::TensorOptions().dtype(torch::kLong).device(dev));

	auto rowVals = -torch::log_softmax(logits, 1).index({arange, arange});
	auto colVals = -torch::log_softmax(logits, 0).index({arange, arange});

	torch::Tensor loss;
	bool useWeights = false;
	if (sampleWeights.defined() && sampleWeights.size(0) == B) {
		auto w = sampleWeights.to(dev);
		auto wSum = w.sum();
		if (wSum.item<float>() > 1e-8f) {
			loss = (rowVals * w).sum() / wSum + (colVals * w).sum() / wSum;
			useWeights = true;
		}
	}
	if (!useWeights)
		loss = rowVals.mean() + colVals.mean();

	auto reg1 = infonce_penalty * (
		torch::logsumexp(logits, 1, false).square().mean() +
		torch::logsumexp(logits, 0, false).square().mean()
	);

	// The embeddings are L2-normalized, so per-dim std tops out around 1/sqrt(repr_dim);
	// scale by sqrt(repr_dim) so a fully-spread embedding scores ~1 regardless of dim.
	// (Unscaled, 1/std is ~sqrt(repr_dim)=16 at repr_dim 256, making this term a large
	// constant that dominates the loss and fights the contrastive objective.)
	float stdNorm = std::sqrt((float)repr_dim);
	auto reg2 = var_reg * (1.0f / (sa.std(0).mean() * stdNorm + 1e-4f) +
	                       1.0f / (g.std(0).mean() * stdNorm + 1e-4f));

	if (outRaw)  *outRaw = loss.detach();
	if (outReg1) *outReg1 = reg1.detach();
	if (outReg2) *outReg2 = reg2.detach();

	return loss + reg1 + reg2;
}

void GGL::QuasimetricCritic::Save(std::filesystem::path folder, bool saveOptim) {
	auto path = GetSavePath(folder);
	torch::serialize::OutputArchive archive;
	this->save(archive);
	archive.save_to(path.string());

	if (saveOptim) {
		torch::serialize::OutputArchive optimArchive;
		optim->save(optimArchive);
		optimArchive.save_to(GetOptimSavePath(folder).string());
	}
}

void GGL::QuasimetricCritic::Load(std::filesystem::path folder, bool allowNotExist, bool loadOptim) {
	auto path = GetSavePath(folder);
	if (!std::filesystem::exists(path)) {
		if (allowNotExist) {
			RG_LOG("Warning: QuasimetricCritic \"" << modelName << "\" not found in " << folder);
			return;
		}
		RG_ERR_CLOSE("QuasimetricCritic \"" << modelName << "\" not found in " << folder);
	}

	torch::serialize::InputArchive archive;
	archive.load_from(path.string(), device);
	this->load(archive);

	if (loadOptim) {
		auto optimPath = GetOptimSavePath(folder);
		if (std::filesystem::exists(optimPath)) {
			std::ifstream test(optimPath, std::istream::ate | std::ios::binary);
			if (test.tellg() > 0) {
				torch::serialize::InputArchive optArch;
				optArch.load_from(optimPath.string(), device);
				optim->load(optArch);
			}
		}
	}
}

GGL::SORSRewardModel::SORSRewardModel(
	const char* modelName,
	int _obs_dim, int _action_dim,
	PartialModelConfig _config,
	torch::Device device) :
	Model(modelName, [&]() {
		ModelConfig fullConfig = _config;
		fullConfig.numInputs = _obs_dim + _action_dim;
		fullConfig.numOutputs = 1;
		return fullConfig;
	}(), device),
	obs_dim(_obs_dim), action_dim(_action_dim) {
}

torch::Tensor GGL::SORSRewardModel::Forward(torch::Tensor obs, torch::Tensor actions, bool halfPrec) {
	auto input = torch::cat({ obs, actions.to(torch::kFloat32).to(obs.device()) }, -1);
	return Model::Forward(input, halfPrec).flatten();
}
