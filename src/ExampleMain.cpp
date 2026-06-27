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
	// TRIAD-NATIVE reward stack (the income-unfreeze keystone, fork 9).
	// opponentScale=0.0 (and teamSpirit=0.0): in 1v1 ZeroSumReward becomes an IDENTITY
	// pass-through (own*(1-0)+avgTeam*0-avgOpp*0 = own). The OLD op=1.0 made the dense
	// stream anti-symmetric across the two cars so symmetric income cancelled to ~0 (the
	// broken-run Average Step Reward~0 signature). At op=0 every dense/event term is pure
	// positive income (L1). For >1v1, teamSpirit=0 = no team sharing, op=0 = no opp subtraction.
	auto teamMixed = [](Reward* reward) {
		return new ZeroSumReward(reward, /*teamSpirit*/0.0f, /*opponentScale*/0.0f);
	};

	// Non-telescoping table: TERMINAL (lands once) + IMPULSE (event-gated one-shot) + the
	// single LEVEL term. The telescoping potential-DELTAS (BallGoalDistance/PlayerBallDistance/
	// Align) and the farm-bait terms (AngVel/TouchGrass/TouchHeight/LowTouchAccel/FlipReset/
	// Boost*/Demo*) are DELETED. Dense income sized <30% of one GoalReward by construction
	// (~10.9%): the only sustained LEVEL term is VelocityPlayerToBall 0.5 (bounded [-1,1]).
	std::vector<WeightedReward> rewards = {
		// Terminal scoring (GoalReward self-zero-sums via concedeScale -1: scorer +275 / conceder -275,
		// NOT routed through opponentScale). Dominant + non-telescoping.
		{ new GoalReward(/*concedeScale*/-1.f), 275.f },
		{ teamMixed(new GoalSpeedBonusReward()), 15.f },     // scorer-only, fires on goal: goal_speed/BALL_MAX_SPEED
		// Defensive/offensive event impulses (event-gated, one-shot, pro-scoring)
		{ teamMixed(new SaveReward()), 20.f },               // PlayerEventState::save
		{ teamMixed(new ShotReward()), 35.f },               // PlayerEventState::shot
		// Dense GOALWARD-STRIKE gradient (xddib2kd fix): rewards the goalward ball-speed INCREASE the
		// agent's touch produces -> breaks the approach-then-idle peak (the CONTROL critic's ball-hold
		// pull). The graded striking signal the event-only touch rewards below could not provide.
		{ teamMixed(new GoalwardImpactReward()), 50.f },
		// Touch impulses
		{ teamMixed(new StrongTouchReward(20.f, 130.f)), 60.f }, // |dBallVel| in [20,130]kph -> committed strike
		{ teamMixed(new TouchBallReward()), 30.f },          // any contact (touch-volume engine)
		{ teamMixed(new AerialTouchReward()), 12.f },        // genuine aerial touch (the reward-stream aerial mechanism)
		// The ONLY sustained LEVEL term: absolute approach velocity, UNWRAPPED (cannot telescope-hug-farm).
		{ new VelocityPlayerToBallReward(), 0.5f },
	};

	std::vector<TerminalCondition*> terminalConditions = {
		new NoTouchCondition(10),
		new GoalScoreCondition()
	};

	// Make the arena
	int playersPerTeam = 1; // 1v1 SOCCAR
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

	// 1v1 SOCCAR: 5120 games * 2 cars ~= 10,240 simulated cars (hold car count constant vs 3v3's 1700*6).
	cfg.numGames = 5120;

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
	cfg.ppo.entropyScale = 0.035f; // initial/fallback scale; the adaptive controller takes over

	// Entropy PIN at 0.70. Capped controllers (old 0.10 ceiling) could not hold entropy --
	// it collapsed to ~0 once the reward/GCRL gradient overpowered the bonus (aooxff9n
	// 0.022, 5gjwloq2 0.19, z533fbde 0.004, the last freezing the policy). The controller
	// is now an uncapped multiplicative Lagrangian: it applies whatever bonus is needed to
	// hold entropy here. Paired with the blended-advantage renorm (bounds the GCRL car
	// critic, which otherwise explodes and fights the pin). Lower the target later once it
	// learns; 0.70 keeps it exploratory so it never freezes.
	cfg.ppo.adaptiveEntropy = true;
	cfg.ppo.targetEntropy = 0.70f;
	cfg.ppo.maxEntropyScale = 5.0f;        // high sanity bound; the pin is uncapped in practice
	cfg.ppo.entropyScaleAdjustRate = 0.2f; // log-space (multiplicative) gain

	// Rate of reward decay
	// 0.995 (~200-step horizon) so the terminal goal propagates back across the
	// ~1000-step episodes to the buildup; 0.99 (~100 steps) was far too short and
	// the goal reward never reached the positioning/strike that earned it.
	cfg.ppo.gaeGamma = 0.995;

	// Good learning rate to start
	cfg.ppo.policyLR = 1.5e-4;
	cfg.ppo.criticLR = 1.5e-4;

	cfg.ppo.sharedHead.layerSizes = { 256, 256 };
	cfg.ppo.policy.layerSizes = { 256, 256, 256 };
	cfg.ppo.critic.layerSizes = { 256, 256, 256 };

	// GCRL magnitude-blend with two isolated critics (goal + car). Per critic the
	// policy-gradient contribution is (taken - baseline)/spread -- NOT unit-renormalized --
	// so a critic that can't yet discriminate the action self-attenuates to ~0 (no noise
	// injection, no gate). The CAR critic (egocentric ball, short HER window) carries early
	// action-attributable controllability signal; the GOAL critic (HER achieved ball, not
	// the synthetic net) firms up later. gcrlLambda ramps in over a short warmup, then holds.
	// Watch GCRL/Car Separation (should climb first) vs GCRL/Goal Separation.
	cfg.ppo.contrastiveGoal.enabled = true;
	cfg.ppo.contrastiveGoal.useCarCritic = true;
	cfg.ppo.contrastiveGoal.gcrlLambda = 0.3f;                  // GCRL-vs-reward blend weight (held after warmup)
	cfg.ppo.contrastiveGoal.gcrlLambdaWarmupSteps = 30'000'000; // short bootstrap ramp, then hold
	cfg.ppo.contrastiveGoal.carHerMaxOffset = 20;               // car critic: short, near-term controllability window
	cfg.ppo.contrastiveGoal.criticLR = 3e-4f;
	cfg.ppo.contrastiveGoal.criticEpochs = 1;
	cfg.ppo.contrastiveGoal.criticMiniBatchSize = 256; // GCRL InfoNCE logits scale quadratically with this
	cfg.ppo.contrastiveGoal.policyScoreBatchSize = 4096;
	// TRIAD-NATIVE: GOALSHORT = world-ball REACH, HER window CAPPED short (long horizons re-enter the
	// 0.055 state-dominated, action-invalid regime); drop the off-policy goalward HER bias.
	cfg.ppo.contrastiveGoal.herMaxOffset = 15;
	cfg.ppo.contrastiveGoal.herGoalwardBias = 0.f;
	// xddib2kd fix: lower GCRL's target share of the policy gradient (was 1:1) so the egocentric
	// CONTROL critic's continuous ball-HOLD pull stops dominating the strike action and the reward
	// (now with the goalward-strike impulse) gets more relative weight. CONTROL still leads cold-start
	// approach (it climbed to edge ~1.0). If approach-then-idle persists, lower further toward 0.3.
	cfg.ppo.contrastiveGoal.gcrlRatioTarget = 0.5f;
	// tau 0.05, VICReg, masked-random K16 baseline, the always-on variance-weight + ratio-pinned lambda
	// controller + RenormToStd are config defaults (PPOLearnerConfig.h). ANTI critic + TD-contrastive are
	// flagged off by default (useAntiCritic / useTDContrastive) pending their dedicated builds + validation.

	// TRIAD-NATIVE self-play (fork 10): 0.15 vs a rolling frozen-checkpoint pool (no ES; BC-seeded once the
	// offline warmstart lands -- until then version 0 is the initial policy). Breaks mutual-idle equilibria.
	cfg.trainAgainstOldVersions = true;
	cfg.trainAgainstOldChance = 0.15f;
	cfg.maxOldVersions = 20;
	cfg.tsPerVersion = 25'000'000;

	// SimBa RSNorm (running observation normalization), default-off. When enabled it
	// standardizes obs as the first op of the actor & critic (one shared normalizer),
	// updated once per rollout and frozen during the epochs, persisted with weights.
	cfg.ppo.rsNorm.enabled = false;

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
