#pragma once
#include "../FrameworkTorch.h"

#include <torch/types.h>
#include <cstdint>

// Deterministic, counter-based Gaussian noise for the PSD/EGGROLL probe perturbations.
//
// The whole EGGROLL memory win is that a perturbation E = (1/sqrt(r)) A B^T is never
// stored: it is regenerated on demand from a small integer key. We reproduce that here.
// A perturbation factor is fully identified by the tuple
//     (roundSeed, probeIdx, layerIdx, which)         which in {0=A, 1=B}
// hashed into a single 64-bit seed. Given the same key on any process / after any
// checkpoint reload, GenFactor() returns the identical tensor, so:
//   * the ES fold can reconstruct every probe's original epsilon direction, and
//   * a resumed run mid-round regenerates the exact factors it had before the crash.
//
// We deliberately do NOT use the global torch RNG (which the collection loop also draws
// from); a private per-key Generator keeps probe noise independent of everything else.
namespace GGL::PSD {

	// splitmix64 — a fast, well-distributed integer hash. Used to fold the key tuple
	// into a single generator seed. (Not cryptographic; we only need good spread.)
	inline uint64_t SplitMix64(uint64_t x) {
		x += 0x9E3779B97F4A7C15ULL;
		x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
		x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
		return x ^ (x >> 31);
	}

	inline uint64_t NoiseKey(uint64_t roundSeed, int probeIdx, int layerIdx, int which) {
		uint64_t k = SplitMix64(roundSeed);
		k = SplitMix64(k ^ (uint64_t)(uint32_t)probeIdx);
		k = SplitMix64(k ^ ((uint64_t)(uint32_t)layerIdx << 1) ^ (uint64_t)(uint32_t)which);
		return k;
	}

	// Generate a standard-normal tensor of `shape` deterministically from the key.
	// Built on the CPU with a private Generator (reproducible regardless of device RNG
	// state), then moved to `device`. rows/cols are small (<=256) so the CPU draw is cheap.
	inline torch::Tensor GenFactor(
		uint64_t roundSeed, int probeIdx, int layerIdx, int which,
		at::IntArrayRef shape, torch::Device device) {

		uint64_t seed = NoiseKey(roundSeed, probeIdx, layerIdx, which);
		auto gen = at::detail::createCPUGenerator();
		{
			std::lock_guard<std::mutex> lock(gen.mutex());
			gen.set_current_seed(seed);
		}
		torch::Tensor t = torch::randn(shape, gen, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCPU));
		return t.to(device);
	}
}
