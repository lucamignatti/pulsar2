#pragma once

#include "ObsBuilder.h"
#include "../../../RocketSim/src/Sim/BallPredTracker/BallPredTracker.h"
#include <memory>

namespace RLGC {
	// NOTE: This not based off of AdvancedObs in Python RLGym, and is specific to GigaLearn
	class AdvancedObs : public ObsBuilder {
	public:

		constexpr static float
			POS_COEF = 1 / 5000.f,
			VEL_COEF = 1 / 2300.f,
			ANG_VEL_COEF = 1 / 3.f;

		int maxPlayers;
		bool mirrorX;
		bool includeBallPred;
		std::vector<float> ballPredTimes;

		std::unique_ptr<RocketSim::BallPredTracker> ballPredTracker;
		Arena* ballPredArena = nullptr;

		AdvancedObs(
			int maxPlayers = 0,
			bool mirrorX = false,
			bool includeBallPred = false,
			std::vector<float> ballPredTimes = { 0.25f, 0.5f, 1.0f, 1.5f, 2.0f }
		) : maxPlayers(maxPlayers), mirrorX(mirrorX), includeBallPred(includeBallPred), ballPredTimes(ballPredTimes) {}

		virtual void AddPlayerToObs(FList& obs, const Player& player, bool inv, bool xMirror, const PhysState& ball);
		virtual void AddBallPredToObs(FList& obs, const Player& player, const GameState& state, bool inv, bool xMirror);
		void AddPadsToObs(FList& obs, const std::vector<bool>& pads, const std::vector<float>& padTimers, bool xMirror);

		virtual FList BuildObs(const Player& player, const GameState& state) override;

		int GetCarLocalBallOffset() const override {
			// Layout before self player: ball(9) + prevAction(8) + pads(34) + ballPred(9*n if enabled)
			int selfStart = 9 + 8 + CommonValues::BOOST_LOCATIONS_AMOUNT;
			if (includeBallPred)
				selfStart += 9 * (int)ballPredTimes.size();
			// Within self player: pos(3)+fwd(3)+up(3)+vel(3)+angVel(3)+localAngVel(3) = 18 before car-local ball
			return selfStart + 18;
		}
	};
}
