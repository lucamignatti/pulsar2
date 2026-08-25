#include "AdvancedObsV4.h"
#include "../CommonValues.h"
#include <RLGymCPP/Gamestates/StateUtil.h>

void RLGC::AdvancedObsV4::AddPlayerToObs(FList& obs, const Player& player, bool inv, const PhysState& ball) {
	auto phys = InvertPhys(player, inv);

	obs += phys.pos * POS_COEF;
	obs += phys.rotMat.forward;
	obs += phys.rotMat.up;
	obs += phys.vel * VEL_COEF;
	obs += phys.angVel * ANG_VEL_COEF;
	obs += phys.rotMat.Dot(phys.angVel) * ANG_VEL_COEF; // Local ang vel

	// Local ball pos and vel
	obs += phys.rotMat.Dot(ball.pos - phys.pos) * POS_COEF;
	obs += phys.rotMat.Dot(ball.vel - phys.vel) * VEL_COEF;

	obs += player.boost / 100;
	obs += player.isOnGround;
	obs += player.HasFlipOrJump();
	obs += player.isDemoed;
	obs += player.hasJumped; // Allows detecting flip resets

	// === V4 additions: continuous jump/dodge/flip clock + flip direction ===
	// V3's AdvancedObs only had the BOOLEAN flip flags; these expose the precise
	// timing windows the bot needs for dodge / flip-reset / wavedash control.
	obs += player.flipTime * TIME_COEF;
	obs += player.airTime * TIME_COEF;
	obs += player.airTimeSinceJump * TIME_COEF; // dodge window (< DOUBLEJUMP_MAX_DELAY ~1.25s)
	obs += player.jumpTime * TIME_COEF;
	obs += player.flipRelTorque;                // Vec3 — car-relative flip torque direction
	obs += player.isFlipping;
	obs += player.isJumping;
	obs += player.hasFlipped;
	obs += player.hasDoubleJumped;
}

RLGC::FList RLGC::AdvancedObsV4::BuildObs(const Player& player, const GameState& state) {
	FList obs = {};

	bool inv = player.team == Team::ORANGE;

	auto ball = InvertPhys(state.ball, inv);
	auto& pads = state.GetBoostPads(inv);
	auto& padTimers = state.GetBoostPadTimers(legacyMirroredTimers ? !inv : inv);

	// --- Live ball ---
	obs += ball.pos * POS_COEF;
	obs += ball.vel * VEL_COEF;
	obs += ball.angVel * ANG_VEL_COEF;

	// --- Analytic ball prediction (gravity + RL drag + 0.6 restitution bounces) ---
	// Same inverted-world frame as the live ball above. RL has no Magnus, so the in-air
	// arc is exact; bounces are approximate (no contact-spin, no curved corners, goal
	// openings treated as solid wall). Constants are RocketSim's own (RLConst.h):
	//   gravity -650, BALL_DRAG 0.03, BALL_RESTITUTION 0.6.
	{
		auto p = ball.pos;
		auto v = ball.vel;
		constexpr float DT   = 6.f / 120.f;   // 0.05s integration step (6 physics ticks)
		constexpr float DRAG = 0.03f;         // RLConst::BALL_DRAG
		constexpr float REST = 0.6f;          // RLConst::BALL_RESTITUTION
		const float R   = CommonValues::BALL_RADIUS;
		const float GZ  = CommonValues::GRAVITY_Z;
		const float MX  = CommonValues::SIDE_WALL_X - R;
		const float MY  = CommonValues::BACK_WALL_Y - R;
		const float MZH = CommonValues::CEILING_Z - R;
		const float MZL = R;

		const float predTimes[] = { 0.5f, 1.0f, 1.5f };
		constexpr int N_PRED = 3;
		constexpr int STEPS  = 30;            // 1.5s / 0.05s
		int pi = 0;
		float t = 0.f;
		for (int s = 0; s < STEPS; s++) {
			v.z += GZ * DT;
			v = v * (1.f - DRAG * DT);
			p = p + v * DT;
			if (p.z < MZL) { p.z = MZL; v.z = -v.z * REST; }
			if (p.z > MZH) { p.z = MZH; v.z = -v.z * REST; }
			if (p.x < -MX) { p.x = -MX; v.x = -v.x * REST; }
			if (p.x >  MX) { p.x =  MX; v.x = -v.x * REST; }
			if (p.y < -MY) { p.y = -MY; v.y = -v.y * REST; }
			if (p.y >  MY) { p.y =  MY; v.y = -v.y * REST; }
			t += DT;
			while (pi < N_PRED && t >= predTimes[pi] - 1e-4f) {
				obs += p * POS_COEF;
				obs += v * VEL_COEF;
				pi++;
			}
		}
		while (pi < N_PRED) { obs += p * POS_COEF; obs += v * VEL_COEF; pi++; } // safety
	}

	for (int i = 0; i < player.prevAction.ELEM_AMOUNT; i++)
		obs += player.prevAction[i];

	for (int i = 0; i < CommonValues::BOOST_LOCATIONS_AMOUNT; i++) {
		if (pads[i]) {
			obs += 1.f;
		} else {
			obs += 1.f / (1.f + padTimers[i]);
		}
	}

	// --- Self ---
	FList selfBlock = {};
	AddPlayerToObs(selfBlock, player, inv, ball);
	size_t pSize = selfBlock.size();
	obs.insert(obs.end(), selfBlock.begin(), selfBlock.end());

	// --- Opponent (1v1: exactly one; zero-pad if transiently absent) ---
	FList oppBlock = {};
	for (auto& otherPlayer : state.players) {
		if (otherPlayer.carId == player.carId)
			continue;
		if (otherPlayer.team != player.team) {
			AddPlayerToObs(oppBlock, otherPlayer, inv, ball);
			break;
		}
	}
	while (oppBlock.size() < pSize) oppBlock += 0.f;
	obs.insert(obs.end(), oppBlock.begin(), oppBlock.end());

	return obs;
}
