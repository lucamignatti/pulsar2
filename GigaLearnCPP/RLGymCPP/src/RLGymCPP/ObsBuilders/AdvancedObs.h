#pragma once

#include "ObsBuilder.h"

namespace RLGC {
	// NOTE: This not based off of AdvancedObs in Python RLGym, and is specific to GigaLearn
	class AdvancedObs : public ObsBuilder {
	public:

		constexpr static float
			POS_COEF = 1 / 5000.f,
			VEL_COEF = 1 / 2300.f,
			ANG_VEL_COEF = 1 / 3.f;

		virtual void AddPlayerToObs(FList& obs, const Player& player, bool inv, const PhysState& ball);

		virtual FList BuildObs(const Player& player, const GameState& state) override;

		// BuildObs layout before the self player: ball(9) + prevAction(ELEM_AMOUNT) + pads(BOOST_LOCATIONS_AMOUNT).
		// Within AddPlayerToObs the self car contributes pos(3)+fwd(3)+up(3)+vel(3)+angVel(3)+localAngVel(3) = 18
		// floats before its car-local ball pos(3)+vel(3). Keep in sync with BuildObs/AddPlayerToObs.
		int GetCarLocalBallOffset() const override {
			return (int)(9 + Action::ELEM_AMOUNT + CommonValues::BOOST_LOCATIONS_AMOUNT) + 18;
		}
	};
}