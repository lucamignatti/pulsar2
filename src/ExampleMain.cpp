#include <GigaLearnCPP/Learner.h>

#include <RLGymCPP/Rewards/CommonRewards.h>
#include <RLGymCPP/Rewards/ZeroSumReward.h>
#include <RLGymCPP/TerminalConditions/NoTouchCondition.h>
#include <RLGymCPP/TerminalConditions/GoalScoreCondition.h>
#include <RLGymCPP/ObsBuilders/DefaultObs.h>
#include <RLGymCPP/ObsBuilders/AdvancedObs.h>
#include <RLGymCPP/StateSetters/KickoffState.h>
#include <RLGymCPP/ActionParsers/DefaultAction.h>

using namespace GGL; // GigaLearn
using namespace RLGC; // RLGymCPP

// Create the RLGymCPP environment for each of our games
EnvCreateResult EnvCreateFunc(int index) {
	constexpr float TEAM_SPIRIT = 0.6f;
	constexpr float OPPONENT_PUNISH = 1.f;
	auto teamMixed = [](Reward* reward) {
		return new ZeroSumReward(reward, TEAM_SPIRIT, OPPONENT_PUNISH);
	};

	std::vector<WeightedReward> rewards = {

		// Ball-goal shaping
		{ teamMixed(new BallGoalDistanceReward()), 2.5f },

		// Goals
		{ teamMixed(new TeamGoalReward()), 12.5f },
		{ teamMixed(new GoalSpeedBonusReward()), 1.25f },
		{ teamMixed(new ConcedeDistanceReward()), 1.25f },

		// Touches
		{ teamMixed(new TouchHeightReward()), 1.f },
		{ teamMixed(new FlipResetReward()), 5.f },

		// Boost
		{ teamMixed(new BoostGainReward()), 0.7f },
		{ teamMixed(new BoostLoseReward()), 0.4f },

		// Demos
		{ teamMixed(new DemoReward()), 2.5f },
		{ teamMixed(new DemoedPenalty()), 2.5f }
	};

	std::vector<TerminalCondition*> terminalConditions = {
		new NoTouchCondition(10),
		new GoalScoreCondition()
	};

	// Make the arena
	int playersPerTeam = 3;
	auto arena = Arena::Create(GameMode::SOCCAR);
	for (int i = 0; i < playersPerTeam; i++) {
		arena->AddCar(Team::BLUE);
		arena->AddCar(Team::ORANGE);
	}

	EnvCreateResult result = {};
	result.actionParser = new DefaultAction();
	result.obsBuilder = new AdvancedObs();
	result.stateSetter = new KickoffState();
	result.terminalConditions = terminalConditions;
	result.rewards = rewards;

	result.arena = arena;

	return result;
}

void StepCallback(Learner* learner, const std::vector<GameState>& states, Report& report) {
	// To prevent expensive metrics from eating at performance, we will only run them on 1/4th of steps
	// This doesn't really matter unless you have expensive metrics (which this example doesn't)
	bool doExpensiveMetrics = (rand() % 4) == 0;

	// Add our metrics
	for (auto& state : states) {
		if (doExpensiveMetrics) {
			for (auto& player : state.players) {
				report.AddAvg("Player/In Air Ratio", !player.isOnGround);
				report.AddAvg("Player/Ball Touch Ratio", player.ballTouchedStep);
				report.AddAvg("Player/Demoed Ratio", player.isDemoed);

				report.AddAvg("Player/Speed", player.vel.Length());
				Vec dirToBall = (state.ball.pos - player.pos).Normalized();
				report.AddAvg("Player/Speed Towards Ball", RS_MAX(0, player.vel.Dot(dirToBall)));

				report.AddAvg("Player/Boost", player.boost);

				if (player.ballTouchedStep)
					report.AddAvg("Player/Touch Height", state.ball.pos.z);
			}
		}

		if (state.goalScored)
			report.AddAvg("Game/Goal Speed", state.ball.vel.Length());
	}
}

int main(int argc, char* argv[]) {
	// Initialize RocketSim with collision meshes
	// Change this path to point to your meshes!
	RocketSim::Init("collision_meshes");

	// Make configuration for the learner
	LearnerConfig cfg = {};

	cfg.deviceType = LearnerDeviceType::AUTO;

	cfg.tickSkip = 8;
	cfg.actionDelay = cfg.tickSkip - 1; // Normal value in other RLGym frameworks

	// 3v3 has 3x as many cars per arena as 1v1, so keep total simulated cars near the old setup.
	cfg.numGames = 1700;

	// Leave this empty to use a random seed each run
	// The random seed can have a strong effect on the outcome of a run
	cfg.randomSeed = 67;

	int tsPerItr = 150'000;
	cfg.ppo.tsPerItr = tsPerItr;
	cfg.ppo.batchSize = tsPerItr;
	cfg.ppo.miniBatchSize = 50'000; // 16 GB VRAM target

	// Using 2 epochs seems pretty optimal when comparing time training to skill
	// Perhaps 1 or 3 is better for you, test and find out!
	cfg.ppo.epochs = 1;

	// This scales differently than "ent_coef" in other frameworks
	// This is the scale for normalized entropy, which means you won't have to change it if you add more actions
	cfg.ppo.entropyScale = 0.035f;

	// Rate of reward decay
	// Starting low tends to work out
	cfg.ppo.gaeGamma = 0.99;

	// Good learning rate to start
	cfg.ppo.policyLR = 1.5e-4;
	cfg.ppo.criticLR = 1.5e-4;

	cfg.ppo.sharedHead.layerSizes = { 256, 256 };
	cfg.ppo.policy.layerSizes = { 256, 256, 256 };
	cfg.ppo.critic.layerSizes = { 256, 256, 256 };

	cfg.ppo.contrastiveGoal.enabled = true;
	cfg.ppo.contrastiveGoal.lambdaStart = 0.f;
	cfg.ppo.contrastiveGoal.lambda = 0.65f;
	cfg.ppo.contrastiveGoal.lambdaAnnealSteps = 200'000'000;
	cfg.ppo.contrastiveGoal.criticLR = 3e-4f;
	cfg.ppo.contrastiveGoal.criticEpochs = 1;
	cfg.ppo.contrastiveGoal.criticMiniBatchSize = 256; // GCRL InfoNCE logits scale quadratically with this
	cfg.ppo.contrastiveGoal.policyScoreBatchSize = 4096;
	cfg.ppo.contrastiveGoal.targetSpeed = 1500.f;
	cfg.ppo.contrastiveGoal.targetSpeedJitter = 0.f;

	auto optim = ModelOptimType::MUON;
	cfg.ppo.policy.optimType = optim;
	cfg.ppo.critic.optimType = optim;
	cfg.ppo.sharedHead.optimType = optim;

	auto activation = ModelActivationType::LEAKY_RELU;
	cfg.ppo.policy.activationType = activation;
	cfg.ppo.critic.activationType = activation;
	cfg.ppo.sharedHead.activationType = activation;

	bool addLayerNorm = false;
	cfg.ppo.policy.addLayerNorm = addLayerNorm;
	cfg.ppo.critic.addLayerNorm = addLayerNorm;
	cfg.ppo.sharedHead.addLayerNorm = addLayerNorm;

	cfg.sendMetrics = true; // Send metrics
	cfg.renderMode = false; // Don't render
	cfg.ppo.deterministic = cfg.renderMode;

	cfg.skillTracker.enabled = true;
	cfg.skillTracker.numArenas = 16;
	cfg.skillTracker.simTime = 45;
	cfg.skillTracker.maxSimTime = 240;
	cfg.skillTracker.updateInterval = 16;
	cfg.skillTracker.ratingInc = 5;
	cfg.skillTracker.initialRating = 0;
	cfg.skillTracker.deterministic = false;

	// Make the learner with the environment creation function and the config we just made
	Learner* learner = new Learner(EnvCreateFunc, cfg, StepCallback);

	// Start learning!
	learner->Start();

	return EXIT_SUCCESS;
}
