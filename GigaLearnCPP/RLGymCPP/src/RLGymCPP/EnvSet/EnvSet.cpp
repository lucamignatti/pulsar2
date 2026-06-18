#include "EnvSet.h"
#include  "../Rewards/ZeroSumReward.h"
#include "../Util/CrashDebug.h"

#include <atomic>

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
	RG_CRASH_LOG(
		"EnvSet ctor begin numArenas=" << config.numArenas <<
		" tickSkip=" << config.tickSkip <<
		" actionDelay=" << config.actionDelay
	);

	RG_ASSERT(config.tickSkip > 0);
	RG_ASSERT(config.actionDelay >= 0 && config.actionDelay <= config.tickSkip);

	std::mutex appendMutex = {};
	auto fnCreateArenas = [&](int idx) {
		RG_CRASH_LOG("EnvSet create arena job begin idx=" << idx);
		auto createResult = config.envCreateFn(idx);
		auto arena = createResult.arena;
		RG_CRASH_LOG("EnvSet create arena job after envCreate idx=" << idx << " arena=" << arena << " cars=" << arena->_cars.size());

		{
			std::lock_guard<std::mutex> lk(appendMutex);

			arenas.push_back(arena);

			auto userInfo = new CallbackUserInfo();
			userInfo->arena = arena;
			userInfo->arenaIdx = (int)arenas.size() - 1;
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
			}

			userInfos.push_back(createResult.userInfo);

			rewards.push_back(createResult.rewards);
			terminalConditions.push_back(createResult.terminalConditions);
			obsBuilders.push_back(createResult.obsBuilder);
			actionParsers.push_back(createResult.actionParser);
			stateSetters.push_back(createResult.stateSetter);
			RG_CRASH_LOG(
				"EnvSet appended arena idx=" << idx <<
				" storedIdx=" << ((int)arenas.size() - 1) <<
				" arena=" << arena <<
				" rewards=" << createResult.rewards.size() <<
				" terminals=" << createResult.terminalConditions.size()
			);
		}
	};
	g_ThreadPool.StartBatchedJobs(fnCreateArenas, config.numArenas, false);
	RG_CRASH_LOG("EnvSet arenas created count=" << arenas.size());

	state.Resize(arenas);
	RG_CRASH_LOG("EnvSet state resized numPlayers=" << state.numPlayers);
	
	// Determine obs size and action amount, initialize arrays accordingly
	{
		RG_CRASH_LOG("EnvSet test reset begin arena=0");
		stateSetters[0]->ResetArena(arenas[0]);
		GameState testState = GameState(arenas[0]);
		testState.userInfo = userInfos[0];
		obsBuilders[0]->Reset(testState);
		obsSize = obsBuilders[0]->BuildObs(testState.players[0], testState).size();
		state.obs = DimList2<float>(state.numPlayers, obsSize);

		state.actionMasks = DimList2<uint8_t>(state.numPlayers, actionParsers[0]->GetActionAmount());
		RG_CRASH_LOG("EnvSet obs/action initialized obsSize=" << obsSize << " actionAmount=" << actionParsers[0]->GetActionAmount());
	}

	// Reset all arenas initially
	RG_CRASH_LOG("EnvSet initial reset all begin");
	g_ThreadPool.StartBatchedJobs(
		std::bind(&RLGC::EnvSet::ResetArena, this, std::placeholders::_1),
		config.numArenas, false
	);
	RG_CRASH_LOG("EnvSet ctor end");
	
}

void RLGC::EnvSet::StepFirstHalf(bool async) {
	static std::atomic<uint64_t> nextCallID = 0;
	uint64_t callID = ++nextCallID;
	RG_CRASH_LOG("EnvSet::StepFirstHalf schedule call=" << callID << " async=" << async << " arenas=" << arenas.size());

	auto fnStepArena = [&](int arenaIdx) {
		Arena* arena = arenas[arenaIdx];
		auto& gs = state.gameStates[arenaIdx];
		RG_CRASH_LOG(
			"StepFirstHalf begin call=" << callID <<
			" arenaIdx=" << arenaIdx <<
			" arena=" << arena <<
			" tick=" << arena->tickCount <<
			" players=" << gs.players.size() <<
			" ball=(" << gs.ball.pos.x << "," << gs.ball.pos.y << "," << gs.ball.pos.z << ")"
		);

		{
			// Set previous gamestates
			RG_CRASH_LOG("StepFirstHalf copy prev begin call=" << callID << " arenaIdx=" << arenaIdx);
			state.prevGameStates[arenaIdx] = gs;
			RG_CRASH_LOG("StepFirstHalf copy prev end call=" << callID << " arenaIdx=" << arenaIdx << " prevPlayers=" << state.prevGameStates[arenaIdx].players.size());
		}

		gs.ResetBeforeStep();
		RG_CRASH_LOG("StepFirstHalf reset before step done call=" << callID << " arenaIdx=" << arenaIdx);

		// Step arena with old actions
		RG_CRASH_LOG("StepFirstHalf arena->Step begin call=" << callID << " arenaIdx=" << arenaIdx << " ticks=" << config.actionDelay);
		arena->Step(config.actionDelay);
		RG_CRASH_LOG("StepFirstHalf arena->Step end call=" << callID << " arenaIdx=" << arenaIdx << " tick=" << arena->tickCount);
	};

	g_ThreadPool.StartBatchedJobs(fnStepArena, arenas.size(), async);
	RG_CRASH_LOG("EnvSet::StepFirstHalf dispatched call=" << callID << " async=" << async);
}

void RLGC::EnvSet::StepSecondHalf(const IList& actionIndices, bool async) {
	static std::atomic<uint64_t> nextCallID = 0;
	uint64_t callID = ++nextCallID;
	RG_CRASH_LOG("EnvSet::StepSecondHalf schedule call=" << callID << " async=" << async << " arenas=" << arenas.size() << " actions=" << actionIndices.size());

	auto fnStepArenas = [&](int arenaIdx) {

		Arena* arena = arenas[arenaIdx];
		auto& gs = state.gameStates[arenaIdx];
		int playerStartIdx = state.arenaPlayerStartIdx[arenaIdx];
		RG_CRASH_LOG(
			"StepSecondHalf begin call=" << callID <<
			" arenaIdx=" << arenaIdx <<
			" arena=" << arena <<
			" tick=" << arena->tickCount <<
			" players=" << gs.players.size() <<
			" playerStartIdx=" << playerStartIdx
		);
			
		// Parse and set actions
		auto actions = std::vector<Action>(gs.players.size());
		auto carItr = arena->_cars.begin();
		for (int i = 0; i < gs.players.size(); i++, carItr++) {
			auto& player = gs.players[i];
			Car* car = *carItr;
			RG_CRASH_LOG(
				"StepSecondHalf parse action call=" << callID <<
				" arenaIdx=" << arenaIdx <<
				" localPlayer=" << i <<
				" flatPlayer=" << (playerStartIdx + i) <<
				" car=" << car <<
				" actionIdx=" << actionIndices[playerStartIdx + i]
			);
			Action action = actionParsers[arenaIdx]->ParseAction(actionIndices[playerStartIdx + i], player, gs);
			car->controls = (CarControls)action;
			actions[i] = action;
		}

		// Step arena with new actions we got from observing the last state
		// Update the gamestate after
		{
			RG_CRASH_LOG("StepSecondHalf arena->Step begin call=" << callID << " arenaIdx=" << arenaIdx << " ticks=" << (config.tickSkip - config.actionDelay));
			arena->Step(config.tickSkip - config.actionDelay);
			RG_CRASH_LOG("StepSecondHalf arena->Step end call=" << callID << " arenaIdx=" << arenaIdx << " tick=" << arena->tickCount);

			RG_CRASH_LOG("StepSecondHalf event tracker begin call=" << callID << " arenaIdx=" << arenaIdx);
			if (eventTrackers[arenaIdx])
				eventTrackers[arenaIdx]->Update(arena);
			RG_CRASH_LOG("StepSecondHalf event tracker end call=" << callID << " arenaIdx=" << arenaIdx);

			GameState* gsPrev = &state.prevGameStates[arenaIdx];
			if (gsPrev->IsEmpty())
				gsPrev = NULL;

			RG_CRASH_LOG("StepSecondHalf UpdateFromArena begin call=" << callID << " arenaIdx=" << arenaIdx << " hasPrev=" << (gsPrev != NULL));
			gs.UpdateFromArena(arena, actions, gsPrev);
			RG_CRASH_LOG(
				"StepSecondHalf UpdateFromArena end call=" << callID <<
				" arenaIdx=" << arenaIdx <<
				" players=" << gs.players.size() <<
				" ball=(" << gs.ball.pos.x << "," << gs.ball.pos.y << "," << gs.ball.pos.z << ")" <<
				" vel=(" << gs.ball.vel.x << "," << gs.ball.vel.y << "," << gs.ball.vel.z << ")"
			);
		}

		// Update terminal
		uint8_t terminalType = TerminalType::NOT_TERMINAL;
		{
			RG_CRASH_LOG("StepSecondHalf terminals begin call=" << callID << " arenaIdx=" << arenaIdx);
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
			RG_CRASH_LOG("StepSecondHalf terminals end call=" << callID << " arenaIdx=" << arenaIdx << " terminalType=" << (int)terminalType);
		}
		
		// Pre-step rewards
		{
			RG_CRASH_LOG("StepSecondHalf rewards PreStep begin call=" << callID << " arenaIdx=" << arenaIdx << " count=" << rewards[arenaIdx].size());
			for (auto& weighted : rewards[arenaIdx])
				weighted.reward->PreStep(gs);
			RG_CRASH_LOG("StepSecondHalf rewards PreStep end call=" << callID << " arenaIdx=" << arenaIdx);
		}

		// Update rewards
		{
			RG_CRASH_LOG("StepSecondHalf rewards compute begin call=" << callID << " arenaIdx=" << arenaIdx);
			FList allRewards = FList(gs.players.size(), 0);
			for (int rewardIdx = 0; rewardIdx < rewards[arenaIdx].size(); rewardIdx++) {
				auto& weightedReward = rewards[arenaIdx][rewardIdx];
				RG_CRASH_LOG("StepSecondHalf reward begin call=" << callID << " arenaIdx=" << arenaIdx << " rewardIdx=" << rewardIdx << " name=" << weightedReward.reward->GetName());
				FList output = weightedReward.reward->GetAllRewards(gs, terminalType);
				RG_CRASH_LOG("StepSecondHalf reward end call=" << callID << " arenaIdx=" << arenaIdx << " rewardIdx=" << rewardIdx << " outputSize=" << output.size());
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
			RG_CRASH_LOG("StepSecondHalf rewards compute end call=" << callID << " arenaIdx=" << arenaIdx);
		}

		// Update observations
		{
			RG_CRASH_LOG("StepSecondHalf obs begin call=" << callID << " arenaIdx=" << arenaIdx);
			for (int i = 0; i < gs.players.size(); i++)
				state.obs.Set(playerStartIdx + i, obsBuilders[arenaIdx]->BuildObs(gs.players[i], gs));
			RG_CRASH_LOG("StepSecondHalf obs end call=" << callID << " arenaIdx=" << arenaIdx);
		}

		// Update action masks
		{
			RG_CRASH_LOG("StepSecondHalf masks begin call=" << callID << " arenaIdx=" << arenaIdx);
			for (int i = 0; i < gs.players.size(); i++)
				state.actionMasks.Set(playerStartIdx + i, actionParsers[arenaIdx]->GetActionMask(gs.players[i], gs));
			RG_CRASH_LOG("StepSecondHalf masks end call=" << callID << " arenaIdx=" << arenaIdx);
		}
		RG_CRASH_LOG("StepSecondHalf end call=" << callID << " arenaIdx=" << arenaIdx);
	};

	g_ThreadPool.StartBatchedJobs(fnStepArenas, arenas.size(), async);
	RG_CRASH_LOG("EnvSet::StepSecondHalf dispatched call=" << callID << " async=" << async);
}

void RLGC::EnvSet::ResetArena(int index) {
	RG_CRASH_LOG("EnvSet::ResetArena begin index=" << index << " arena=" << arenas[index] << " tick=" << arenas[index]->tickCount);
	stateSetters[index]->ResetArena(arenas[index]);
	RG_CRASH_LOG("EnvSet::ResetArena after setter index=" << index << " tick=" << arenas[index]->tickCount);
	GameState newState = GameState(arenas[index]);
	RG_CRASH_LOG("EnvSet::ResetArena after GameState index=" << index << " players=" << newState.players.size());
	newState.userInfo = userInfos[index];
	state.gameStates[index] = newState;
	RG_CRASH_LOG("EnvSet::ResetArena assigned state index=" << index);

	// Update event tracker
	if (eventTrackers[index])
		eventTrackers[index]->ResetPersistentInfo();

	// Reset all the other stuff
	RG_CRASH_LOG("EnvSet::ResetArena obs/reward resets begin index=" << index);
	obsBuilders[index]->Reset(newState);
	for (auto& cond : terminalConditions[index])
		cond->Reset(newState);
	for (auto& weightedReward : rewards[index])
		weightedReward.reward->Reset(newState);
	RG_CRASH_LOG("EnvSet::ResetArena obs/reward resets end index=" << index);

	int playerStartIdx = state.arenaPlayerStartIdx[index];
	for (int i = 0; i < newState.players.size(); i++) {
		RG_CRASH_LOG("EnvSet::ResetArena build player begin index=" << index << " localPlayer=" << i << " flatPlayer=" << (playerStartIdx + i));

		// Update obs
		auto obs = obsBuilders[index]->BuildObs(newState.players[i], newState);
		state.obs.Set(playerStartIdx + i, obs);

		// Update action mask
		auto actionMask = actionParsers[index]->GetActionMask(newState.players[i], newState);
		state.actionMasks.Set(playerStartIdx + i, actionMask);
		RG_CRASH_LOG("EnvSet::ResetArena build player end index=" << index << " localPlayer=" << i);
	}

	// Remove previous state
	state.prevGameStates[index].MakeEmpty();
	RG_CRASH_LOG("EnvSet::ResetArena end index=" << index);
}

void RLGC::EnvSet::Reset() {
	RG_CRASH_LOG("EnvSet::Reset begin arenas=" << arenas.size());
	for (int i = 0; i < arenas.size(); i++)
		if (state.terminals[i]) {
			RG_CRASH_LOG("EnvSet::Reset schedule arena=" << i << " terminal=" << (int)state.terminals[i]);
			g_ThreadPool.StartJobAsync(std::bind(&EnvSet::ResetArena, this, std::placeholders::_1), i);
		}
	std::fill(state.terminals.begin(), state.terminals.end(), 0);
	g_ThreadPool.WaitUntilDone();
	RG_CRASH_LOG("EnvSet::Reset end");
}
