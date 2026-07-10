// Standalone correctness self-test for the PSD/EGGROLL primitive (PolicySlots).
//
// Links against libGigaLearnCPP. Proves the load-bearing properties the plan calls out:
//   [1] sigma=0 => slot-routed forward is BIT-IDENTICAL to the base policy forward
//       (guarantees the DESCEND/baseline path is untouched).
//   [2] sigma>0 => slot-routed forward matches S fully-materialized per-slot clones
//       (independent reference: builds each perturbed weight matrix densely, runs seq->forward).
//   [3] noise is deterministic across regeneration (checkpoint-resume safety).
//   [4] FoldESUpdate applies exactly alpha * sum_s w[s] * eps_s to the base weights.
//
// Build (from build/): see tools/build_psd_selftest.sh. Run: ./psd_selftest
#include <GigaLearnCPP/Framework.h>
#include "../GigaLearnCPP/src/private/GigaLearnCPP/PSD/PolicySlots.h"
#include "../GigaLearnCPP/src/private/GigaLearnCPP/Util/Models.h"

#include <torch/nn/modules/linear.h>
#include <iostream>

using namespace GGL;
using namespace GGL::PSD;

static int g_fails = 0;
static void Check(bool cond, const std::string& name, double err = -1) {
	std::cout << (cond ? "  PASS  " : "  FAIL  ") << name;
	if (err >= 0) std::cout << "   (max abs err " << err << ")";
	std::cout << "\n";
	if (!cond) g_fails++;
}

// Build a policy Model matching the real arch shape (LayerNorm + LeakyReLU + output layer).
static Model* MakePolicy(int obs, int nAct, torch::Device dev) {
	PartialModelConfig pc;
	pc.layerSizes = { 16, 16, 16 };
	pc.activationType = ModelActivationType::LEAKY_RELU;
	pc.addLayerNorm = true;
	pc.addOutputLayer = true;
	ModelConfig mc = pc;
	mc.numInputs = obs;
	mc.numOutputs = nAct;
	return new Model("policy", mc, dev);
}

// Independent reference: clone the policy, add coef_l * A_s B_s^T to every Linear weight,
// then run seq->forward on this slot's rows. Uses the stock forward, not PolicySlots' walk.
static torch::Tensor RefSlotForward(PolicySlots& ps, torch::Tensor xSlot, int slot) {
	RG_NO_GRAD;
	Model* clone = ps.policy->MakeClone();
	int l = 0;
	auto& seq = clone->seq;
	for (size_t i = 0; i < seq->size(); i++) {
		auto lin = std::dynamic_pointer_cast<torch::nn::LinearImpl>(seq->ptr(i));
		if (!lin) continue;
		torch::Tensor A = ps.Astack[l][slot].detach(); // [out, r]
		torch::Tensor B = ps.Bstack[l][slot].detach(); // [in, r]
		torch::Tensor eps = torch::matmul(A, B.transpose(0, 1)) * ps.layers[l].coef; // [out,in]
		lin->weight.add_(eps);
		l++;
	}
	torch::Tensor y = clone->Forward(xSlot, false);
	delete clone;
	return y;
}

int main() {
	torch::manual_seed(1234);
	torch::Device dev(torch::kCPU);
	const int obs = 12, nAct = 7, K = 3, rank = 2, S = 2 * K, rpS = 5;
	const int N = S * rpS;

	Model* policy = MakePolicy(obs, nAct, dev);
	torch::Tensor x = torch::randn({ N, obs }, torch::TensorOptions().dtype(torch::kFloat).device(dev));

	// [1] sigma = 0 -> bit-identical to base forward.
	{
		PolicySlots ps(policy, S, rank, /*sigma=*/0.0f, dev);
		ps.InitFromNoise(/*roundSeed=*/42, /*antithetic=*/true);
		torch::Tensor got = ps.Forward(x);
		torch::Tensor ref = policy->Forward(x, false);
		double err = (got - ref).abs().max().item<double>();
		Check(torch::equal(got, ref) || err == 0.0, "[1] sigma=0 forward is bit-identical to base", err);
	}

	// [2] sigma > 0 -> matches materialized per-slot clones.
	{
		PolicySlots ps(policy, S, rank, /*sigma=*/0.05f, dev);
		ps.InitFromNoise(/*roundSeed=*/7, /*antithetic=*/true);
		torch::Tensor got = ps.Forward(x);
		double maxErr = 0;
		for (int s = 0; s < S; s++) {
			torch::Tensor xSlot = x.slice(0, s * rpS, (s + 1) * rpS);
			torch::Tensor refSlot = RefSlotForward(ps, xSlot, s);
			torch::Tensor gotSlot = got.slice(0, s * rpS, (s + 1) * rpS);
			maxErr = std::max(maxErr, (gotSlot - refSlot).abs().max().item<double>());
		}
		Check(maxErr < 1e-4, "[2] sigma>0 slot routing matches materialized clones", maxErr);
	}

	// [3] antithetic twins are exact opposites in perturbation, and noise regenerates identically.
	{
		PolicySlots a(policy, S, rank, 0.05f, dev), b(policy, S, rank, 0.05f, dev);
		a.InitFromNoise(99, true);
		b.InitFromNoise(99, true);
		double dperm = 0, danti = 0;
		for (int l = 0; l < (int)a.layers.size(); l++) {
			dperm = std::max(dperm, (a.Astack[l] - b.Astack[l]).abs().max().item<double>());
			// slot 2p and 2p+1 share |A| with opposite sign
			for (int p = 0; p < K; p++)
				danti = std::max(danti, (a.Astack[l][2 * p] + a.Astack[l][2 * p + 1]).abs().max().item<double>());
		}
		Check(dperm == 0.0, "[3a] noise regenerates identically from the same key", dperm);
		Check(danti == 0.0, "[3b] antithetic twins are exact opposites", danti);
	}

	// [4] FoldESUpdate applies exactly alpha * sum_s w[s] * eps_s.
	{
		PolicySlots ps(policy, S, rank, 0.05f, dev);
		ps.InitFromNoise(5, true);
		std::vector<float> w(S);
		for (int s = 0; s < S; s++) w[s] = (float)(s - S / 2.0) / S; // arbitrary centered weights
		float alpha = 0.7f;

		// weights before, per Linear
		std::vector<torch::Tensor> before;
		for (size_t i = 0; i < policy->seq->size(); i++)
			if (auto lin = std::dynamic_pointer_cast<torch::nn::LinearImpl>(policy->seq->ptr(i)))
				before.push_back(lin->weight.detach().clone());

		ps.FoldESUpdate(5, true, w, alpha);

		double maxErr = 0;
		int l = 0;
		for (size_t i = 0; i < policy->seq->size(); i++) {
			auto lin = std::dynamic_pointer_cast<torch::nn::LinearImpl>(policy->seq->ptr(i));
			if (!lin) continue;
			torch::Tensor expectedDelta = torch::zeros_like(before[l]);
			for (int s = 0; s < S; s++) {
				int probe = s / 2; float sign = (s % 2 == 1) ? -1.f : 1.f;
				torch::Tensor A0 = GenFactor(5, probe, l, 0, { ps.layers[l].outDim, rank }, dev) * sign;
				torch::Tensor B0 = GenFactor(5, probe, l, 1, { ps.layers[l].inDim, rank }, dev);
				expectedDelta += torch::matmul(A0, B0.transpose(0, 1)) * (w[s] * ps.layers[l].coef);
			}
			expectedDelta *= alpha;
			torch::Tensor actualDelta = lin->weight.detach() - before[l];
			maxErr = std::max(maxErr, (actualDelta - expectedDelta).abs().max().item<double>());
			l++;
		}
		Check(maxErr < 1e-5, "[4] ES fold equals alpha * sum_s w[s] * eps_s", maxErr);
	}

	delete policy;
	std::cout << (g_fails == 0 ? "\nALL PASS\n" : "\nFAILED\n");
	return g_fails == 0 ? 0 : 1;
}
