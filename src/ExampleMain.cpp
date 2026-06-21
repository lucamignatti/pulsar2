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

		// DENSE SHAPING IS NOW THE GCRL POTENTIAL HEADS, not rewards. Removed (they would
		// double-shape on top of the potentials):
		//   PlayerBallDistanceReward -> car head (egocentric ball -> contact)
		//   BallGoalDistanceReward   -> goal head (HER ball -> scoring-mouth range)
		//   ConcedeDistanceReward    -> defense head (opponents' scoring reachability)
		// What remains: SPARSE task/event rewards (the objective + discrete achievements that a
		// smooth reachability potential can't represent), plus boost (no GCRL head yet).

		// Goals (sparse task)
		{ teamMixed(new TeamGoalReward()), 12.5f },
		{ teamMixed(new GoalSpeedBonusReward()), 1.25f },

		// Touches / aerial events
		{ teamMixed(new TouchHeightReward()), 1.f },
		{ teamMixed(new FlipResetReward()), 5.f },

		// Boost (dense, but no GCRL head yet -- the one remaining dense reward)
		{ teamMixed(new BoostGainReward()), 0.7f },
		{ teamMixed(new BoostLoseReward()), 0.4f },

		// Demos (events)
		{ teamMixed(new DemoReward()), 2.5f },
		{ teamMixed(new DemoedPenalty()), 2.5f }
	};

	std::vector<TerminalCondition*> terminalConditions = {
		new NoTouchCondition(10),
		new GoalScoreCondition()
	};

	// Make the arena
	int playersPerTeam = 3; // 3v3 SOCCAR (real training)
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

	// 3v3 SOCCAR: 1700 games * 6 cars ~= 10,200 simulated cars (the original setup).
	cfg.numGames = 1700;

	// Leave this empty to use a random seed each run
	// The random seed can have a strong effect on the outcome of a run
	cfg.randomSeed = 67;

	int tsPerItr = 150'000;
	cfg.ppo.tsPerItr = tsPerItr;
	cfg.ppo.batchSize = tsPerItr;
	cfg.ppo.miniBatchSize = 50'000; // 16 GB VRAM target

	// 2 epochs (the value this comment has always recommended). epochs=1 gave the
	// policy only a single gradient step per iteration's data AND made Mean KL a
	// meaningless ~0 (ratio==1 by construction, so reported KL was just float
	// noise), hiding whether the policy was actually moving.
	cfg.ppo.epochs = 2;

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

	// GCRL as the UNIFIED POTENTIAL-BASED SHAPING framework. Each head's reachability potential Phi(s)
	// is injected as gamma*Phi(s')-Phi(s) into the reward stream (policy-invariant, farming-proof),
	// instead of the magnitude-blend advantage. Heads: CAR (egocentric ball -> contact), GOAL (HER ball
	// -> soft-max over the scoring-mouth range), DEFENSE (-soft-max over opponents' goal-reachability).
	// Watch GCRL/Shaping {Car,Goal,Defense} AbsMean and GCRL/Potential * Mean.
	//   A/B baseline = the magnitude-blend advantage: set usePotentialShaping=false (then gcrlLambda*
	//   apply and the advantage path runs instead). useSharedBase folds goal+car onto one phi base.
	cfg.ppo.contrastiveGoal.enabled = true;
	cfg.ppo.contrastiveGoal.useCarCritic = true;
	cfg.ppo.contrastiveGoal.usePotentialShaping = true;  // POTENTIAL framework (false -> advantage A/B baseline)
	cfg.ppo.contrastiveGoal.potentialDefense = true;     // defense head (opponent reachability); false = offense only
	cfg.ppo.contrastiveGoal.useSharedBase = false;       // true -> one shared phi base across the heads
	// The potentials are now the PRIMARY dense signal (the dense shaping rewards are gone), so the
	// scale is bumped from the 0.3 "augment" value. THIS IS THE CRITICAL COLD-START KNOB: the car
	// head's contact potential must bootstrap ball approach in place of PlayerBallDistanceReward.
	// If touch ratio doesn't climb in the first ~30M ts, raise this (toward 2-3).
	cfg.ppo.contrastiveGoal.potentialShapingScale = 1.0f;
	cfg.ppo.contrastiveGoal.gcrlLambda = 0.3f;                  // (advantage-mode only)
	cfg.ppo.contrastiveGoal.gcrlLambdaWarmupSteps = 30'000'000; // (advantage-mode only)
	cfg.ppo.contrastiveGoal.carHerMaxOffset = 20;               // car critic: short, near-term controllability window
	cfg.ppo.contrastiveGoal.criticLR = 3e-4f;
	cfg.ppo.contrastiveGoal.criticEpochs = 1;
	cfg.ppo.contrastiveGoal.criticMiniBatchSize = 256; // GCRL InfoNCE logits scale quadratically with this
	cfg.ppo.contrastiveGoal.policyScoreBatchSize = 4096;

	// SimBa RSNorm (running observation normalization), default-off. When enabled it
	// standardizes obs as the first op of the actor & critic (one shared normalizer),
	// updated once per rollout and frozen during the epochs, persisted with weights.
	cfg.ppo.rsNorm.enabled = true;

	auto optim = ModelOptimType::MUON;
	cfg.ppo.policy.optimType = optim;
	cfg.ppo.critic.optimType = optim;
	cfg.ppo.sharedHead.optimType = optim;

	auto activation = ModelActivationType::LEAKY_RELU;
	cfg.ppo.policy.activationType = activation;
	cfg.ppo.critic.activationType = activation;
	cfg.ppo.sharedHead.activationType = activation;

	bool addLayerNorm = true; // ROCm-safe ManualLayerNorm (see Models.h)
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
