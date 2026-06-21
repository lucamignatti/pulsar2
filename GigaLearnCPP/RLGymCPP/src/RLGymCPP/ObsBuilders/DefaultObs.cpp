#include "DefaultObs.h"
#include "../Gamestates/StateUtil.h"

void RLGC::DefaultObs::AddPlayerToObs(FList& obs, const Player& player, bool inv) {
	auto phys = InvertPhys(player, inv);

	obs += phys.pos * posCoef;
	obs += phys.rotMat.forward;
	obs += phys.rotMat.up;
	obs += phys.vel * velCoef;
	obs += phys.angVel * angVelCoef;

	obs += player.boost / 100;
	obs += player.isOnGround;
	obs += player.HasFlipOrJump();
	obs += player.isDemoed;
}

RLGC::FList RLGC::DefaultObs::BuildObs(const Player& player, const GameState& state) {
	FList result = {};
	if (_obsSizeHint) result.reserve(_obsSizeHint);

	bool inv = player.team == Team::ORANGE;

	auto ball = InvertPhys(state.ball, inv);
	auto& pads = state.GetBoostPads(inv);

	result += ball.pos * posCoef;
	result += ball.vel * velCoef;
	result += ball.angVel * angVelCoef;

	for (int i = 0; i < player.prevAction.ELEM_AMOUNT; i++)
		result += player.prevAction[i];

	for (int i = 0; i < CommonValues::BOOST_LOCATIONS_AMOUNT; i++)
		result += (float)pads[i];

	AddPlayerToObs(result, player, inv);
	FList teammates = {}, opponents = {};
	teammates.reserve(28 * state.players.size());
	opponents.reserve(28 * state.players.size());

	for (auto& otherPlayer : state.players) {
		if (otherPlayer.carId == player.carId)
			continue;

		AddPlayerToObs(
			(otherPlayer.team == player.team) ? teammates : opponents,
			otherPlayer,
			inv
		);
	}

	result += teammates;
	result += opponents;
	_obsSizeHint = result.size();
	return result;
}