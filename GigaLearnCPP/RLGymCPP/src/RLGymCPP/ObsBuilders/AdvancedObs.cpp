#include "AdvancedObs.h"
#include <RLGymCPP/Gamestates/StateUtil.h>
#include <array>
#include <algorithm>
#include <cfloat>

namespace {
	std::array<int, RLGC::CommonValues::BOOST_LOCATIONS_AMOUNT> MakeMirrorPadMap() {
		std::array<int, RLGC::CommonValues::BOOST_LOCATIONS_AMOUNT> result = {};
		for (int i = 0; i < RLGC::CommonValues::BOOST_LOCATIONS_AMOUNT; i++) {
			Vec target = RLGC::CommonValues::BOOST_LOCATIONS[i];
			target.x *= -1;

			int bestIdx = 0;
			float bestDist = FLT_MAX;
			for (int j = 0; j < RLGC::CommonValues::BOOST_LOCATIONS_AMOUNT; j++) {
				Vec delta = RLGC::CommonValues::BOOST_LOCATIONS[j] - target;
				float dist = delta.Dot(delta);
				if (dist < bestDist) {
					bestDist = dist;
					bestIdx = j;
				}
			}
			result[i] = bestIdx;
		}
		return result;
	}

	const std::array<int, RLGC::CommonValues::BOOST_LOCATIONS_AMOUNT> MIRROR_PAD_MAP = MakeMirrorPadMap();
}

void RLGC::AdvancedObs::AddPlayerToObs(FList& obs, const Player& player, bool inv, bool xMirror, const PhysState& ball) {
	auto phys = MirrorPhysX(InvertPhys(player, inv), xMirror);

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
}

void RLGC::AdvancedObs::AddBallPredToObs(FList& obs, const Player& player, const GameState& state, bool inv, bool xMirror) {
	auto selfPhys = MirrorPhysX(InvertPhys(player, inv), xMirror);

	if (!state.lastArena) {
		for (int i = 0; i < ballPredTimes.size(); i++) {
			obs += Vec() * POS_COEF;
			obs += Vec() * VEL_COEF;
			obs += Vec() * POS_COEF;
		}
		return;
	}

	if (!ballPredTracker || ballPredArena != state.lastArena) {
		ballPredArena = state.lastArena;
		ballPredTracker = std::make_unique<RocketSim::BallPredTracker>(state.lastArena, 256);
	} else {
		ballPredTracker->UpdatePredFromArena(state.lastArena);
	}

	for (float predTime : ballPredTimes) {
		auto predBall = MirrorPhysX(InvertPhys(ballPredTracker->GetBallStateForTime(predTime), inv), xMirror);
		obs += predBall.pos * POS_COEF;
		obs += predBall.vel * VEL_COEF;
		obs += (predBall.pos - selfPhys.pos) * POS_COEF;
	}
}

void RLGC::AdvancedObs::AddPadsToObs(FList& obs, const std::vector<bool>& pads, const std::vector<float>& padTimers, bool xMirror) {
	for (int i = 0; i < CommonValues::BOOST_LOCATIONS_AMOUNT; i++) {
		int padIdx = xMirror ? MIRROR_PAD_MAP[i] : i;
		if (pads[padIdx]) {
			obs += 1.f;
		} else {
			obs += 1.f / (1.f + padTimers[padIdx]);
		}
	}
}

RLGC::FList RLGC::AdvancedObs::BuildObs(const Player& player, const GameState& state) {
	FList obs;
	BuildObsInto(obs, player, state);
	return obs;
}

void RLGC::AdvancedObs::BuildObsInto(FList& obs, const Player& player, const GameState& state) {
	obs.clear();

	bool inv = player.team == Team::ORANGE;
	bool xMirror = mirrorX && ShouldMirrorXForPlayer(player);

	auto ball = MirrorPhysX(InvertPhys(state.ball, inv), xMirror);
	auto& pads = state.GetBoostPads(inv);
	auto& padTimers = state.GetBoostPadTimers(inv);

	constexpr int PLAYER_OBS_SIZE_HINT = 29;
	int paddedPlayerCount = maxPlayers > 0 ? (maxPlayers * 2) : (int)state.players.size();
	size_t baseObsSize = 9 + player.prevAction.ELEM_AMOUNT + CommonValues::BOOST_LOCATIONS_AMOUNT + extraObs.size();
	if (includeBallPred)
		baseObsSize += 9 * ballPredTimes.size();
	obs.reserve(baseObsSize + (size_t)paddedPlayerCount * PLAYER_OBS_SIZE_HINT);

	obs += ball.pos * POS_COEF;
	obs += ball.vel * VEL_COEF;
	obs += ball.angVel * ANG_VEL_COEF;

	Action prevAction = MirrorActionX(player.prevAction, xMirror);
	for (int i = 0; i < prevAction.ELEM_AMOUNT; i++)
		obs += prevAction[i];

	AddPadsToObs(obs, pads, padTimers, xMirror);

	if (includeBallPred)
		AddBallPredToObs(obs, player, state, inv, xMirror);

	size_t selfObsStart = obs.size();
	AddPlayerToObs(obs, player, inv, xMirror, ball);
	int playerObsSize = (int)(obs.size() - selfObsStart);

	std::vector<const Player*> teammates;
	std::vector<const Player*> opponents;
	teammates.reserve(maxPlayers > 0 ? maxPlayers - 1 : state.players.size());
	opponents.reserve(maxPlayers > 0 ? maxPlayers : state.players.size());

	for (auto& otherPlayer : state.players) {
		if (otherPlayer.carId == player.carId)
			continue;

		((otherPlayer.team == player.team) ? teammates : opponents).push_back(&otherPlayer);
	}

	if (maxPlayers > 0) {
		if (teammates.size() > maxPlayers - 1)
			RG_ERR_CLOSE("AdvancedObs: Too many teammates for Obs, maximum is " << (maxPlayers - 1));

		if (opponents.size() > maxPlayers)
			RG_ERR_CLOSE("AdvancedObs: Too many opponents for Obs, maximum is " << maxPlayers);

		while (teammates.size() < maxPlayers - 1)
			teammates.push_back(nullptr);
		while (opponents.size() < maxPlayers)
			opponents.push_back(nullptr);
	}

	std::shuffle(teammates.begin(), teammates.end(), ::Math::GetRandEngine());
	std::shuffle(opponents.begin(), opponents.end(), ::Math::GetRandEngine());

	for (const Player* teammate : teammates) {
		if (teammate)
			AddPlayerToObs(obs, *teammate, inv, xMirror, ball);
		else
			obs.insert(obs.end(), playerObsSize, 0.0f);
	}
	for (const Player* opponent : opponents) {
		if (opponent)
			AddPlayerToObs(obs, *opponent, inv, xMirror, ball);
		else
			obs.insert(obs.end(), playerObsSize, 0.0f);
	}

	// User-supplied constants (e.g. game mode flags), always last so fixed offsets above hold
	obs += extraObs;
}
