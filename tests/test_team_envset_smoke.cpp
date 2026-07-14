#include "TestFramework.h"

#include <RLGymCPP/EnvSet/EnvSet.h>
#include <RLGymCPP/Rewards/CommonRewards.h>
#include <RLGymCPP/Rewards/ZeroSumReward.h>
#include <RLGymCPP/TerminalConditions/NoTouchCondition.h>
#include <RLGymCPP/TerminalConditions/GoalScoreCondition.h>
#include <RLGymCPP/ObsBuilders/AdvancedObsPadded.h>
#include <RLGymCPP/StateSetters/KickoffState.h>
#include <RLGymCPP/StateSetters/RandomState.h>
#include <RLGymCPP/StateSetters/BallNearCarState.h>
#include <RLGymCPP/StateSetters/AirDrillState.h>
#include <RLGymCPP/StateSetters/CombinedState.h>
#include <RLGymCPP/ActionParsers/DefaultAction.h>

#include <cmath>
#include <filesystem>

using namespace RLGC;

// Integration smoke for the 4.0 team-play config: a 1v1 + 2v2 + 3v3 EnvSet with the
// production reward stack (TEAM_SPIRIT 0.3) and the team-aware state setters, stepped
// with random valid actions through several resets. Asserts the whole-stack zero-sum
// invariant (every component is zero-sum or antisymmetric across equal-size teams, so
// per-arena reward sums to ~0 every step) and that nothing NaNs or crashes.

static constexpr float SMOKE_GAMMA = 0.9985f;
static constexpr float SMOKE_TS = 0.3f;

static void EnsureRocketSimSmoke() {
	static bool inited = false;
	if (inited)
		return;
	for (const char* path : { "collision_meshes", "../collision_meshes", "build/collision_meshes" }) {
		if (std::filesystem::exists(std::filesystem::path(path) / "soccar")) {
			RocketSim::Init(path);
			inited = true;
			return;
		}
	}
	// RocketSim::Init is idempotent-guarded upstream; if another test file already
	// initialized it, creating an arena works without a second Init
	inited = true;
}

// Mirrors ExampleMain's BuildRewards (FRONTIER-9, team-play spirit)
static std::vector<WeightedReward> SmokeRewards() {
	return {
		{ new BallToGoalPotentialReward(SMOKE_GAMMA), 75.f },
		{ new ZeroSumReward(new TouchAccelReward(), SMOKE_TS), 10.f },
		{ new ZeroSumReward(new DemoReward(), SMOKE_TS), 37.5f },
		{ new ZeroSumReward(new BallProximityPotentialReward(SMOKE_GAMMA), SMOKE_TS), 4.f },
		{ new ZeroSumReward(new GuardedPickupBoostReward(), SMOKE_TS), 6.f },
		{ new ZeroSumReward(new AerialTouchReward(), SMOKE_TS), 25.f },
		{ new ZeroSumReward(new AirInterceptPotentialReward(SMOKE_GAMMA), SMOKE_TS), 10.f },
		{ new ZeroSumReward(new OpposedSaveReward(), SMOKE_TS), 25.f },
		{ new GoalReward(), 150 }
	};
}

TEST(TeamEnvSet_smoke_zero_sum_and_finite) {
	EnsureRocketSimSmoke();

	// Arena 0 = 1v1, arena 1 = 2v2, arena 2 = 3v3 (the PHASE B mix in miniature)
	EnvSetConfig cfg = {};
	cfg.numArenas = 3;
	cfg.tickSkip = 4;
	cfg.actionDelay = 3;
	cfg.saveRewards = false;
	cfg.envCreateFn = [](int index) {
		int playersPerTeam = index + 1;
		auto arena = Arena::Create(GameMode::SOCCAR);
		for (int i = 0; i < playersPerTeam; i++) {
			arena->AddCar(Team::BLUE);
			arena->AddCar(Team::ORANGE);
		}
		EnvCreateResult result = {};
		result.arena = arena;
		result.actionParser = new DefaultAction();
		result.obsBuilder = new AdvancedObsPadded(3);
		result.stateSetter = new CombinedState({
			{ new BallNearCarState(600, 900), 0.35f },
			{ new AirDrillState(), 0.20f },
			{ new KickoffState(), 0.15f },
			{ new RandomState(true, true, false), 0.30f },
		});
		result.terminalConditions = { new NoTouchCondition(10), new GoalScoreCondition() };
		result.rewards = SmokeRewards();
		return result;
	};

	EnvSet envSet(cfg);
	CHECK_EQ(envSet.state.numPlayers, 2 + 4 + 6);
	CHECK_EQ(envSet.obsSize, 230); // padded obs: 51 header + 6x29 slots + 5 presence flags

	int resetsSeen = 0;
	for (int step = 0; step < 600; step++) {
		envSet.StepFirstHalf(false);

		// Random valid action per player
		IList actions(envSet.state.numPlayers, 0);
		for (int p = 0; p < envSet.state.numPlayers; p++) {
			if (!envSet.state.actionMasks.Defined()) // first step: masks not built yet
				continue;
			std::vector<int> valid;
			for (int a = 0; a < envSet.numActions; a++)
				if (envSet.state.actionMasks.At(p, a))
					valid.push_back(a);
			if (!valid.empty())
				actions[p] = valid[RocketSim::Math::RandInt(0, (int)valid.size())];
		}
		envSet.StepSecondHalf(actions, false);

		for (int arenaIdx = 0; arenaIdx < cfg.numArenas; arenaIdx++) {
			int start = envSet.state.arenaPlayerStartIdx[arenaIdx];
			int count = (int)envSet.state.gameStates[arenaIdx].players.size();
			float sum = 0;
			for (int p = start; p < start + count; p++) {
				float r = envSet.state.rewards[p];
				CHECK(std::isfinite(r));
				sum += r;
			}
			// Whole-stack invariant: zero-sum per arena at every step (equal team sizes)
			CHECK_NEAR(sum, 0.f, 1e-2f);
			if (envSet.state.terminals[arenaIdx])
				resetsSeen++;
		}
	}
	envSet.Sync(); // let any pending async arena resets finish before teardown

	// 600 steps = 20s of game time per arena; NoTouch(10) under random actions
	// guarantees at least one reset happened, so the setters ran mid-episode-stream
	CHECK(resetsSeen >= 1);
}
