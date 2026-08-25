#pragma once

#include "ObsBuilder.h"

namespace RLGC {
	// V4 obs builder (BonkDaddy V4). 1v1-native. Adds over AdvancedObs:
	//  - continuous car-state timers (flipTime/airTime/airTimeSinceJump/jumpTime),
	//    the flip torque DIRECTION (flipRelTorque), and isFlipping/isJumping/hasFlipped/hasDoubleJumped
	//  - analytic ball-prediction (gravity + RL drag + 0.6 restitution wall/ground/ceiling bounces),
	//    sampled at a few future offsets, in the same inverted-world frame as the live ball.
	// NOTE: separate class from AdvancedObs so V3 (AdvancedObsPadded) is untouched.
	class AdvancedObsV4 : public ObsBuilder {
	public:
		constexpr static float
			POS_COEF = 1 / 5000.f,
			VEL_COEF = 1 / 2300.f,
			ANG_VEL_COEF = 1 / 3.f,
			TIME_COEF = 1 / 1.5f; // car-state timers are O(seconds); ~normalize to ~[0,1]

		// kept for interface parity with AdvancedObs; V4 always trains with correct timers (false)
		bool legacyMirroredTimers;

		AdvancedObsV4(bool legacyMirroredTimers = false) : legacyMirroredTimers(legacyMirroredTimers) {}

		void AddPlayerToObs(FList& obs, const Player& player, bool inv, const PhysState& ball);

		virtual FList BuildObs(const Player& player, const GameState& state) override;
	};
}
