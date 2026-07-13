#include "AdvancedObsPadded.h"
#include <RLGymCPP/Gamestates/StateUtil.h>

RLGC::FList RLGC::AdvancedObsPadded::BuildObs(const Player& player, const GameState& state) {
	int numTeammateSlots = maxPlayersPerTeam - 1;
	int numOpponentSlots = maxPlayersPerTeam;

	FList obs = {};
	constexpr int PLAYER_OBS_ELEMS = 29; // Reserve hint only; pad rows are sized from the real self block
	obs.reserve(9 + player.prevAction.ELEM_AMOUNT + CommonValues::BOOST_LOCATIONS_AMOUNT
		+ PLAYER_OBS_ELEMS * (1 + numTeammateSlots + numOpponentSlots)
		+ numTeammateSlots + numOpponentSlots);

	bool inv = player.team == Team::ORANGE;

	auto ball = InvertPhys(state.ball, inv);
	auto& pads = state.GetBoostPads(inv);
	auto& padTimers = state.GetBoostPadTimers(inv);

	obs += ball.pos * POS_COEF;
	obs += ball.vel * VEL_COEF;
	obs += ball.angVel * ANG_VEL_COEF;

	for (int i = 0; i < player.prevAction.ELEM_AMOUNT; i++)
		obs += player.prevAction[i];

	for (int i = 0; i < CommonValues::BOOST_LOCATIONS_AMOUNT; i++) {
		// A clever trick that blends the boost pads using their timers
		if (pads[i]) {
			obs += 1.f; // Pad is already available
		} else {
			obs += 1.f / (1.f + padTimers[i]); // Approaches 1 as the pad becomes available
		}
	}

	FList selfObs = {};
	AddPlayerToObs(selfObs, player, inv, ball);
	obs += selfObs;
	int playerObsSize = selfObs.size();

	std::vector<FList> teammates = {}, opponents = {};

	for (auto& otherPlayer : state.players) {
		if (otherPlayer.carId == player.carId)
			continue;

		FList playerObs = {};
		AddPlayerToObs(playerObs, otherPlayer, inv, ball);
		((otherPlayer.team == player.team) ? teammates : opponents).push_back(playerObs);
	}

	if ((int)teammates.size() > numTeammateSlots)
		RG_ERR_CLOSE("AdvancedObsPadded: Too many teammates for obs, maximum is " << numTeammateSlots);

	if ((int)opponents.size() > numOpponentSlots)
		RG_ERR_CLOSE("AdvancedObsPadded: Too many opponents for obs, maximum is " << numOpponentSlots);

	// Assign real players to shuffled slots; pads are all-zero rows. The presence flags are
	// paired with the SAME slot assignment and appended after all player blocks, so the
	// per-player stride stays uniform and existing offset-based readers are unaffected.
	FList presenceFlags = {};
	presenceFlags.reserve(numTeammateSlots + numOpponentSlots);

	for (int group = 0; group < 2; group++) {
		auto& playerList = group ? opponents : teammates;
		int slotCount = group ? numOpponentSlots : numTeammateSlots;

		std::vector<int> slotOrder(slotCount);
		for (int i = 0; i < slotCount; i++)
			slotOrder[i] = i;
		std::shuffle(slotOrder.begin(), slotOrder.end(), ::Math::GetRandEngine());

		std::vector<const FList*> slots(slotCount, nullptr);
		for (int i = 0; i < (int)playerList.size(); i++)
			slots[slotOrder[i]] = &playerList[i];

		for (int i = 0; i < slotCount; i++) {
			if (slots[i]) {
				obs += *slots[i];
				presenceFlags += 1.f;
			} else {
				obs += FList(playerObsSize);
				presenceFlags += 0.f;
			}
		}
	}

	obs += presenceFlags;
	return obs;
}
