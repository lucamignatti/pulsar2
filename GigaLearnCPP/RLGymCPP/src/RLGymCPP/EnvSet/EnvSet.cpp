#include "EnvSet.h"
#include  "../Rewards/ZeroSumReward.h"

template<bool RLGC::PlayerEventState::* DATA_VAR>
void IncPlayerCounter(Car* car, void* userInfoPtr) {
	if (!car)
		return;

	auto userInfo = (RLGC::EnvSet::CallbackUserInfo*)userInfoPtr;

	auto& gs = userInfo->envSet->state.gameStates[userInfo->arenaIdx];
	for (auto& player : gs.players)
		if (player.carId == car->id)
			(player.eventState.*DATA_VAR) = true;
}

void _ShotEventCallback(Arena* arena, Car* shooter, Car* passer, void* userInfo) {
	IncPlayerCounter<&RLGC::PlayerEventState::shot>(shooter, userInfo);
	IncPlayerCounter<&RLGC::PlayerEventState::shotPass>(passer, userInfo);
}

void _GoalEventCallback(Arena* arena, Car* scorer, Car* passer, void* userInfo) {
	IncPlayerCounter<&RLGC::PlayerEventState::goal>(scorer, userInfo);
	IncPlayerCounter<&RLGC::PlayerEventState::assist>(passer, userInfo);
}

void _SaveEventCallback(Arena* arena, Car* saver, void* userInfo) {
	IncPlayerCounter<&RLGC::PlayerEventState::save>(saver, userInfo);
}

void _BumpCallback(Arena* arena, Car* bumper, Car* victim, bool isDemo, void* userInfo) {
	if (bumper->team == victim->team)
		return;

	IncPlayerCounter<&RLGC::PlayerEventState::bump>(bumper, userInfo);
	IncPlayerCounter<&RLGC::PlayerEventState::bumped>(victim, userInfo);

	if (isDemo) {
		IncPlayerCounter<&RLGC::PlayerEventState::demo>(bumper, userInfo);
		IncPlayerCounter<&RLGC::PlayerEventState::demoed>(victim, userInfo);
	}
}

/////////////////////////////

RLGC::EnvSet::EnvSet(const EnvSetConfig& config) : config(config) {

	RG_ASSERT(config.tickSkip > 0);
	RG_ASSERT(config.actionDelay >= 0 && config.actionDelay <= config.tickSkip);

	std::mutex appendMutex = {};
	auto fnCreateArenas = [&](int idx) {
		auto createResult = config.envCreateFn(idx);
		auto arena = createResult.arena;

		appendMutex.lock();
		{
			arenas.push_back(arena);

			auto userInfo = new CallbackUserInfo();
			userInfo->arena = arena;
			userInfo->arenaIdx = idx;
			userInfo->envSet = this;
			eventCallbackInfos.push_back(userInfo);
			arena->SetCarBumpCallback(_BumpCallback, userInfo);

			if (arena->gameMode != GameMode::HEATSEEKER) {
				GameEventTracker* tracker = new GameEventTracker({});
				eventTrackers.push_back(tracker);

				tracker->SetShotCallback(_ShotEventCallback, userInfo);
				tracker->SetGoalCallback(_GoalEventCallback, userInfo);
				tracker->SetSaveCallback(_SaveEventCallback, userInfo);
			} else {
				eventTrackers.push_back(NULL);
				eventCallbackInfos.push_back(NULL);
			}

			userInfos.push_back(createResult.userInfo);

			rewards.push_back(createResult.rewards);
			terminalConditions.push_back(createResult.terminalConditions);
			obsBuilders.push_back(createResult.obsBuilder);
			actionParsers.push_back(createResult.actionParser);
			stateSetters.push_back(createResult.stateSetter);
		}
		appendMutex.unlock();
	};
	g_ThreadPool.StartBatchedJobs(fnCreateArenas, config.numArenas, false);

	state.Resize(arenas);
	
	// Determine obs size and action amount, initialize arrays accordingly
	{
		stateSetters[0]->ResetArena(arenas[0]);
		GameState testState = GameState(arenas[0]);
		testState.userInfo = userInfos[0];
		obsBuilders[0]->Reset(testState);
		obsSize = obsBuilders[0]->BuildObs(testState.players[0], testState).size();
		state.obs = DimList2<float>(state.numPlayers, obsSize);

		state.actionMasks = DimList2<uint8_t>(state.numPlayers, actionParsers[0]->GetActionAmount());
	}

	// Reset all arenas initially
	g_ThreadPool.StartBatchedJobs(
		std::bind(&RLGC::EnvSet::ResetArena, this, std::placeholders::_1),
		config.numArenas, false
	);
	
}

void RLGC::EnvSet::StepFirstHalf(bool async) {

	auto fnStepArena = [&](int arenaIdx) {
		Arena* arena = arenas[arenaIdx];
		auto& gs = state.gameStates[arenaIdx];

		{
			// Set previous gamestates
			state.prevGameStates[arenaIdx] = gs;
		}

		gs.ResetBeforeStep();

		// Step arena with old actions
		arena->Step(config.actionDelay);
	};

	g_ThreadPool.StartBatchedJobs(fnStepArena, arenas.size(), async);
}

void RLGC::EnvSet::StepSecondHalf(const IList& actionIndices, bool async) {

	auto fnStepArenas = [&](int arenaIdx) {

		Arena* arena = arenas[arenaIdx];
		auto& gs = state.gameStates[arenaIdx];
		int playerStartIdx = state.arenaPlayerStartIdx[arenaIdx];
			
		// Parse and set actions
		auto actions = std::vector<Action>(gs.players.size());
		auto carItr = arena->_cars.begin();
		for (int i = 0; i < gs.players.size(); i++, carItr++) {
			auto& player = gs.players[i];
			Car* car = *carItr;
			Action action = actionParsers[arenaIdx]->ParseAction(actionIndices[playerStartIdx + i], player, gs);
			car->controls = (CarControls)action;
			actions[i] = action;
		}

		// Step arena with new actions we got from observing the last state
		// Update the gamestate after
		{
			arena->Step(config.tickSkip - config.actionDelay);

			if (eventTrackers[arenaIdx])
				eventTrackers[arenaIdx]->Update(arena);

			GameState* gsPrev = &state.prevGameStates[arenaIdx];
			if (gsPrev->IsEmpty())
				gsPrev = NULL;

			gs.UpdateFromArena(arena, actions, gsPrev);
		}

		// Update terminal
		uint8_t terminalType = TerminalType::NOT_TERMINAL;
		{
			for (auto cond : terminalConditions[arenaIdx]) {
				if (cond->IsTerminal(gs)) {
					bool isTrunc = cond->IsTruncation();
					uint8_t curTerminalType = isTrunc ? TerminalType::TRUNCATED : TerminalType::NORMAL;
					if (terminalType == TerminalType::NOT_TERMINAL) {
						terminalType = curTerminalType;
					} else {
						// We already know this state is terminal
						// However, if we only know it is a truncated terminal, we should let normal terminals take priority
						// (Normal terminals are better information than truncations)
						if (curTerminalType == TerminalType::NORMAL)
							terminalType = curTerminalType;
					}

					// NOTE: We can't break since terminal conditions are guaranteed to be called once per step
				}
			}
			state.terminals[arenaIdx] = terminalType;
		}
		
		// Pre-step rewards
		{
			for (auto& weighted : rewards[arenaIdx])
				weighted.reward->PreStep(gs);
		}

		// Update rewards
		{
			FList allRewards = FList(gs.players.size(), 0);
			for (int rewardIdx = 0; rewardIdx < rewards[arenaIdx].size(); rewardIdx++) {
				auto& weightedReward = rewards[arenaIdx][rewardIdx];
				FList output = weightedReward.reward->GetAllRewards(gs, terminalType);
				for (int i = 0; i < gs.players.size(); i++)
					allRewards[i] += output[i] * weightedReward.weight;

				// Save the reward
				if (config.saveRewards) {
					int playerSampleIndex;
					if (config.shuffleRewardSampling) {
						playerSampleIndex = Math::RandInt(0, output.size());
					} else {
						// Find player with the lowest id
						playerSampleIndex = 0;
						int lowestID = gs.players[0].carId;
						for (int i = 1; i < gs.players.size(); i++) {
							auto id = gs.players[i].carId;
							if (id < lowestID) {
								lowestID = id;
								playerSampleIndex = i;
							}
						}
					}
					// We will only take the reward from a random player
					float rewardToSave = output[playerSampleIndex];
						
					// If zero-sum, use the inner reward
					if (ZeroSumReward* zeroSum = dynamic_cast<ZeroSumReward*>(weightedReward.reward))
						rewardToSave = zeroSum->_lastRewards[playerSampleIndex];

					// If needed, initialize last rewards
					if (state.lastRewards[arenaIdx].empty())
						state.lastRewards[arenaIdx].resize(rewards[arenaIdx].size());

					state.lastRewards[arenaIdx][rewardIdx] = rewardToSave;
				}
			}

			for (int i = 0; i < gs.players.size(); i++)
				state.rewards[playerStartIdx + i] = allRewards[i];
		}

		// Update observations
		{
			for (int i = 0; i < gs.players.size(); i++)
				state.obs.Set(playerStartIdx + i, obsBuilders[arenaIdx]->BuildObs(gs.players[i], gs));
		}

		// Update action masks
		{
			for (int i = 0; i < gs.players.size(); i++)
				state.actionMasks.Set(playerStartIdx + i, actionParsers[arenaIdx]->GetActionMask(gs.players[i], gs));
		}
	};

	g_ThreadPool.StartBatchedJobs(fnStepArenas, arenas.size(), async);
}

void RLGC::EnvSet::ResetArena(int index) {
	stateSetters[index]->ResetArena(arenas[index]);
	GameState newState = GameState(arenas[index]);
	state.gameStates[index] = newState;

	newState.userInfo = userInfos[index];

	// Update event tracker
	if (eventTrackers[index])
		eventTrackers[index]->ResetPersistentInfo();

	// Reset all the other stuff
	obsBuilders[index]->Reset(newState);
	for (auto& cond : terminalConditions[index])
		cond->Reset(newState);
	for (auto& weightedReward : rewards[index])
		weightedReward.reward->Reset(newState);

	int playerStartIdx = state.arenaPlayerStartIdx[index];
	for (int i = 0; i < newState.players.size(); i++) {

		// Update obs
		auto obs = obsBuilders[index]->BuildObs(newState.players[i], newState);
		state.obs.Set(playerStartIdx + i, obs);

		// Update action mask
		auto actionMask = actionParsers[index]->GetActionMask(newState.players[i], newState);
		state.actionMasks.Set(playerStartIdx + i, actionMask);
	}

	// Remove previous state
	state.prevGameStates[index].MakeEmpty();
}

void RLGC::EnvSet::Reset() {
	for (int i = 0; i < arenas.size(); i++)
		if (state.terminals[i])
			g_ThreadPool.StartJobAsync(std::bind(&EnvSet::ResetArena, this, std::placeholders::_1), i);
	std::fill(state.terminals.begin(), state.terminals.end(), 0);
	g_ThreadPool.WaitUntilDone();
}