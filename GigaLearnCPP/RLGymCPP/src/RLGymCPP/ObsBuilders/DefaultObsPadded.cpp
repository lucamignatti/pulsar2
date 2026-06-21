#include "DefaultObsPadded.h"
#include "../Gamestates/StateUtil.h"

RLGC::FList RLGC::DefaultObsPadded::BuildObs(const Player& player, const GameState& state) {
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

	FList selfObs = {};
	AddPlayerToObs(selfObs, player, inv);
	result += selfObs;
	int playerObsSize = selfObs.size();

	std::vector<FList> teammates = {}, opponents = {};

	for (auto& otherPlayer : state.players) {
		if (otherPlayer.carId == player.carId)
			continue;

		FList playerObs = {};
		AddPlayerToObs(
			playerObs,
			otherPlayer,
			inv
		);
		((otherPlayer.team == player.team) ? teammates : opponents).push_back(playerObs);
	}

	if (teammates.size() > maxPlayers - 1)
		RG_ERR_CLOSE("DefaultObsPadded: Too many teammates for Obs, maximum is " << (maxPlayers - 1));
	
	if (opponents.size() > maxPlayers)
		RG_ERR_CLOSE("DefaultObsPadded: Too many opponents for Obs, maximum is " << maxPlayers);

	for (int i = 0; i < 2; i++) {
		auto& playerList = i ? teammates : opponents;
		int targetCount = i ? maxPlayers - 1 : maxPlayers;

		while (playerList.size() < targetCount) {
			FList pad = FList(playerObsSize);
			playerList.push_back(pad);
		}
	}

	// Shuffle both lists
	std::shuffle(teammates.begin(), teammates.end(), ::Math::GetRandEngine());
	std::shuffle(opponents.begin(), opponents.end(), ::Math::GetRandEngine());

	for (auto& teammate : teammates)
		result += teammate;
	for (auto& opponent : opponents)
		result += opponent;

	_obsSizeHint = result.size();
	return result;
}