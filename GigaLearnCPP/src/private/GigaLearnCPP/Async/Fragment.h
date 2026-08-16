#pragma once
#include "../FrameworkTorch.h"

#include <cstdint>
#include <vector>

namespace GGL {

constexpr uint32_t kFragmentMagic = 0x41535031u;

#pragma pack(push, 1)
struct FragmentHeader {
	uint32_t magic = kFragmentMagic;
	uint32_t version = 1;
	int32_t collector_world = 0;
	int32_t T = 0;
	int32_t obsSize = 0;
	int32_t numActions = 0;
	int32_t nTrunc = 0;
	int32_t flags = 0;
	uint64_t fragment_id = 0;
	uint64_t min_policy_version = 0;
	uint64_t max_policy_version = 0;
	uint64_t env_steps = 0;
	float oppCtx[4] = {};
	int32_t nexto_goals_for = 0;
	int32_t nexto_goals_against = 0;
};
#pragma pack(pop)

struct TrajectoryFragment {
	FragmentHeader hdr;
	torch::Tensor states;       // [T, obs] f32 cpu
	torch::Tensor actions;      // [T] i32
	torch::Tensor logProbs;     // [T] f32
	torch::Tensor rewards;      // [T] f32
	torch::Tensor terminals;    // [T] i8
	torch::Tensor actionMasks;  // [T, A] u8
	torch::Tensor policy_version; // [T] u32
	torch::Tensor nextStates;   // [nTrunc, obs] f32
	torch::Tensor goalRews;
	torch::Tensor carHerGoals;
	torch::Tensor ballHerGoals;
	torch::Tensor ballMovedMask;
	torch::Tensor carStateHerGoals;
	torch::Tensor gatedPos;
	torch::Tensor touched;
	torch::Tensor oppTouched;
	torch::Tensor oppStates;
	torch::Tensor oppActionMasks;
	torch::Tensor practiceMask;

	int64_t rows() const { return hdr.T; }
	std::vector<uint8_t> Pack() const;
	static TrajectoryFragment Unpack(const uint8_t* data, size_t nbytes);
	TrajectoryFragment SlicePrefix(int64_t k) const;
	TrajectoryFragment SliceSuffix(int64_t k) const;
};

}
