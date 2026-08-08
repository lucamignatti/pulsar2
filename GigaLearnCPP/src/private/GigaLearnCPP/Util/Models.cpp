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

	if (config.addResiduals) {
		// The skip adds a block's input to its output, so every hidden layer must share a width.
		for (int size : config.layerSizes)
			if (size != config.layerSizes[0])
				RG_ERR_CLOSE("Model \"" << modelName << "\": addResiduals requires uniform layerSizes, got "
					<< size << " among " << config.layerSizes[0] << "-wide layers");
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
		auto halfOutput = residualSpans.empty()
			? seqHalf->forward(halfInput)
			: ForwardResidual(seqHalf, residualSpans, halfInput);
		return halfOutput.to(torch::kFloat);
	} else {
		return residualSpans.empty()
			? seq->forward(input)
			: ForwardResidual(seq, residualSpans, input);
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
