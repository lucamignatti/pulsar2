#pragma once
#include "../FrameworkTorch.h"

namespace GGL {
	// Left-right mirror of AdvancedObsPadded observations, as a (permutation, sign)
	// pair over the obs vector: mirrored = obs[perm] * sign. The game is exactly
	// mirror-symmetric across the x=0 plane in the team-canonical frame, so training
	// the VALUE heads on mirrored copies with unchanged targets is free doubled data
	// (measured on the offline testbed: hallucinated-headroom floor 9x down at intact
	// calibration -- research/testbeds/inject2d RESULTS.md batch 10).
	//
	// Element transforms (x -> -x):
	//   world positions / velocities / forward / up : negate x-component
	//   world angular velocity (PSEUDOvector)       : negate y and z
	//   local (forward,right,up) regular vectors    : negate the right component
	//   local (forward,right,up) pseudovectors      : negate forward and up components
	//   prevAction                                  : negate steer, yaw, roll
	//   boost pads                                  : permute to the x-mirrored pad
	//   scalars / flags / presence                  : unchanged
	//
	// Build() constructs the map programmatically from the obs layout and
	// CommonValues::BOOST_LOCATIONS and hard-fails on any structural inconsistency
	// (pad without a unique mirror partner, non-involutive map, size mismatch).
	// A wrong mirror is SILENT training corruption; the checks are not optional.
	namespace ObsMirror {
		struct Map {
			torch::Tensor perm;   // int64 [obsSize]
			torch::Tensor sign;   // float [obsSize]
			int obsSize = 0;
			bool IsValid() const { return obsSize > 0; }
		};

		// maxPlayersPerTeam: the obs builder's padding (3 in the 5.x/6.x runs).
		// obsSize: the runtime obs width, asserted against the layout math.
		Map Build(int maxPlayersPerTeam, int obsSize);

		// rows x obsSize -> rows x obsSize, same device/dtype
		torch::Tensor Apply(const Map& map, const torch::Tensor& obs);
	}
}
