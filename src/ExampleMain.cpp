#include <GigaLearnCPP/Learner.h>

#include <RLGymCPP/Rewards/CommonRewards.h>
#include <RLGymCPP/Rewards/ZeroSumReward.h>
#include <RLGymCPP/TerminalConditions/NoTouchCondition.h>
#include <RLGymCPP/TerminalConditions/GoalScoreCondition.h>
#include <RLGymCPP/ObsBuilders/DefaultObs.h>
#include <RLGymCPP/ObsBuilders/AdvancedObs.h>
#include <RLGymCPP/StateSetters/KickoffState.h>
#include <RLGymCPP/StateSetters/RandomState.h>
#include <RLGymCPP/StateSetters/BallNearCarState.h>
#include <RLGymCPP/StateSetters/CombinedState.h>
#include <RLGymCPP/ActionParsers/DefaultAction.h>

using namespace GGL; // GigaLearn
using namespace RLGC; // RLGymCPP

// Create the RLGymCPP environment for each of our games
EnvCreateResult EnvCreateFunc(int index) {
	// SURGICAL-7: potentials + impulse-scaled touch height (Nexto ratios x15, Goal=150).
	// Design rules:
	// (1) every continuous term is an EXACT potential (gamma*Phi(s') - Phi(s)) -> all
	//     movement cycles, whack-and-chase loops, and truncation harvests telescope to 0
	//     by construction, not by weight-tuning;
	// (2) every touch term pays for IMPULSE (delta ball-vel), never contact-time -> no
	//     dribble/wall-pin/ceiling-carry annuity can exist;
	// (3) events (touch, demo, goal) are ungated and zero-sum; nothing pays per-step for
	//     existing (no Air, no SaveBoost) -> ambient do-nothing income is exactly 0.
	// The gamma inside the potential classes must match cfg.ppo.gaeGamma (0.99).
	std::vector<WeightedReward> rewards = {

		// Player->ball proximity potential (Nexto liu_dist, dist_w=0.5 x15). Replaces
		// VelocityPlayerToBallReward: the potential charges the full Phi drop when the
		// ball is whacked away, so the 7.5/sec chase annuity nets ~0 per cycle. Its 3D
		// distance also pays climbing toward an overhead ball and refunds whiffs.
		{ new BallProximityPotentialReward(), 7.5f },

		// Ball->goal potential (Nexto state_quality). Replaces ZeroSum(VelocityBallToGoal):
		// antisymmetric between teams, so it is ALREADY zero-sum — no wrapper (the wrapper
		// was a silent 2x). exp() concentrates credit at the goal mouth: a corner spray
		// pockets ~3, not ~60, and rolls back for a full refund. ~+25 integrated midfield->net.
		{ new BallToGoalPotentialReward(), 75.f },

		// Touch quality: unchanged proven bootstrap (Nexto touch_accel). Pays only for
		// adding ball speed, ~10 total 0->110kph, zero-sum so touches can't be co-farmed.
		{ new ZeroSumReward(new TouchAccelReward(), 0), 10.f },

		// Touch HEIGHT (Nexto touch_height x15), impulse-scaled: carries pay ~0, a real
		// strike pays the full height credit, airborne strikes pay double. The only term
		// where a high touch is worth more than a low one — the aerial gradient.
		// GATED: the impulse factor already kills contact-time carries; the reachability
		// gate additionally mutes the aerial juggle self-rally (a contested pop at height
		// pays 10-15/sec) in states that don't matter. Beta anneals from 0 and the 0.2
		// floor keep cold aerial learning intact (mult ~0.78 at the live run's beta~0.44).
		{ new ZeroSumReward(new TouchHeightReward(), 0), 15.f, true },

		// Boost pickup, halved (big pad from empty = 4 = 2.7% of a goal). SaveBoost removed:
		// per-step income for holding a full tank taxed spending boost on aerials.
		// GATED (as in the proven old stack): don't pay for a boost-collection circuit in
		// states where we can't win the ball or score.
		{ new PickupBoostReward(), 4.f, true },

		// Demo halved so the zero-sum pair swing is 75 = goal/2. Bump removed entirely:
		// its 0.25s re-fire push-grind paid up to 40/sec; Nexto had no bump term.
		{ new ZeroSumReward(new DemoReward(), 0.5f), 37.5f },

		// The objective. All dense income per scoring possession sums to ~35-40 (~25% of
		// a goal), none of it collectible without moving the ball toward scoring.
		{ new GoalReward(), 150 }
	};

	std::vector<TerminalCondition*> terminalConditions = {
		new NoTouchCondition(10),
		new GoalScoreCondition()
	};

	// Make the arena
	int playersPerTeam = 1;
	auto arena = Arena::Create(GameMode::SOCCAR);
	for (int i = 0; i < playersPerTeam; i++) {
		arena->AddCar(Team::BLUE);
		arena->AddCar(Team::ORANGE);
	}

	EnvCreateResult result = {};
	result.actionParser = new DefaultAction();
	result.obsBuilder = new AdvancedObs();
	// Effective near-ball share is 0.55 (0.35 ground + 0.20 aerial drill, cars <= 1200
	// from the ball), preserving the anti-freeze touch bootstrap margin while finally
	// posing aerial and defensive states.
	result.stateSetter = new CombinedState({
		// Ground touch bootstrap — unchanged behavior, the proven anti-freeze state
		{ new BallNearCarState(600, 900), 0.35f },
		// Aerial drill: ball hangs at 500-1500uu drifting down, BOTH cars on a 300-1200
		// ring with >=40 boost, facing it. Symmetric, so there is no lucky-car windfall:
		// the race to meet the falling ball IS the zero-sum challenge. From 1000uu the
		// ball takes ~1.6-2.2s to land — a real window for jump/double-jump/boost-climb.
		{ new BallNearCarState(300, 1200, 500, 1500, 400, 40), 0.20f },
		{ new KickoffState(), 0.15f },
		// Sole source of chaotic/defensive/air-recovery states (bounds widened to reach
		// corners and goal lines)
		{ new RandomState(true, true, false), 0.30f },
	});
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

				// Tripwires for the reward stack: aerials emerging / demo farming
				report.AddAvg("Player/Aerial Touch Ratio",
					player.ballTouchedStep && !player.isOnGround && state.ball.pos.z > 400);
				report.AddAvg("Player/Demo Rate", (float)player.eventState.demo);
			}
		}

		if (state.goalScored)
			report.AddAvg("Game/Goal Speed", state.ball.vel.Length());
	}
}

int main(int argc, char* argv[]) {
	// Initialize RocketSim with collision meshes (run from the repo/build dir;
	// provision them with tools/get_collision_meshes.sh if missing)
	RocketSim::Init("collision_meshes");

	// Make configuration for the learner
	LearnerConfig cfg = {};

	cfg.deviceType = LearnerDeviceType::GPU_CUDA;

	cfg.tickSkip = 8;
	cfg.actionDelay = cfg.tickSkip - 1; // Normal value in other RLGym frameworks

	// Play around with this to see what the optimal is for your machine, more games will consume more RAM
	// 1024 sized for a 7900X (24 threads) + 5080: at 256 the per-step policy forward was a
	// 512-row batch — pure launch overhead on this GPU. 2048 rows/step keeps it fed.
	cfg.numGames = 1024;

	// Leave this empty to use a random seed each run
	// The random seed can have a strong effect on the outcome of a run
	cfg.randomSeed = 123;

	int tsPerItr = 200'000;
	cfg.ppo.tsPerItr = tsPerItr;
	cfg.ppo.batchSize = tsPerItr;
	cfg.ppo.miniBatchSize = 200'000; // Lower this if too much VRAM is being allocated

	// BF16 inference for collection + GAE value preds (Blackwell tensor cores). The
	// reachability paths are unaffected: rho/gate evals request fp32 explicitly and
	// grad-enabled forwards (InfoNCE training) always run fp32.
	cfg.ppo.useHalfPrecision = true;

	// Using 2 epochs seems pretty optimal when comparing time training to skill
	// Perhaps 1 or 3 is better for you, test and find out!
	cfg.ppo.epochs = 2;

	// This scales differently than "ent_coef" in other frameworks
	// This is the scale for normalized entropy, which means you won't have to change it if you add more actions
	cfg.ppo.entropyScale = 0.035f;

	// Reachability (aux InfoNCE heads on the shared trunk + reward gate).
	// Experiment arms: A = both off (pure baseline), B = enabled only (aux representation
	// effect), C = both on (the full gate). Arm C for the SURGICAL-7 stack: the gate and
	// the potentials are COMPLEMENTARY on DISJOINT terms. Potentials stay ungated (positive-
	// part gating breaks their telescoping); the gate takes the residual farmable surface —
	// TouchHeight (aerial juggle rally) + PickupBoost (boost circuit) — the two rewards
	// marked gated above. Beta anneals in on measured head validity, floored at 0.2.
	cfg.ppo.reachability.enabled = true;
	cfg.ppo.reachability.gateEnabled = true;

	// Wide clip, NOT 0: cold return-sigma under this near-sparse stack is ~2-4, so the
	// default clip of 10 compressed the first goals 2-5x right at goal onset — but 0
	// would let a first goal land as an unclipped 40+ sigma value-target spike under the
	// lifetime Welford sigma. 50 releases the full 150 once sigma >= 3 and bounds the tail.
	cfg.ppo.rewardClipRange = 50;

	// Rate of reward decay
	// Starting low tends to work out
	cfg.ppo.gaeGamma = 0.99;

	// Good learning rate to start
	cfg.ppo.policyLR = 1.5e-4;
	cfg.ppo.criticLR = 1.5e-4;

	cfg.ppo.sharedHead.layerSizes = { 256, 256 };
	cfg.ppo.policy.layerSizes = { 256, 256, 256 };
	cfg.ppo.critic.layerSizes = { 256, 256, 256 };
	cfg.ppo.reachability.phi.layerSizes = { 256, 256 };
	cfg.ppo.reachability.psi.layerSizes = { 256, 256 };
	cfg.ppo.reachability.lr = 3e-4f;

	// Muon's RMS-matched scaling makes Adam-tuned LRs transfer as-is.
	// The reachability heads deliberately stay on Adam: contrastive InfoNCE embeddings
	// train poorly under orthogonalized updates (see ReachabilityConfig).
	auto optim = ModelOptimType::MUON;
	cfg.ppo.policy.optimType = optim;
	cfg.ppo.critic.optimType = optim;
	cfg.ppo.sharedHead.optimType = optim;

	auto activation = ModelActivationType::LEAKY_RELU;
	cfg.ppo.policy.activationType = activation;
	cfg.ppo.critic.activationType = activation;
	cfg.ppo.sharedHead.activationType = activation;
	cfg.ppo.reachability.phi.activationType = activation;
	cfg.ppo.reachability.psi.activationType = activation;

	bool addLayerNorm = true;
	cfg.ppo.policy.addLayerNorm = addLayerNorm;
	cfg.ppo.critic.addLayerNorm = addLayerNorm;
	cfg.ppo.sharedHead.addLayerNorm = addLayerNorm;
	cfg.ppo.reachability.phi.addLayerNorm = addLayerNorm;
	cfg.ppo.reachability.psi.addLayerNorm = addLayerNorm;

	// Skill rating: Elo-style eval matches against saved policy versions, logged to
	// wandb as Rating/<mode> (e.g. Rating/1v1). Also turns on savePolicyVersions.
	cfg.skillTracker.enabled = true;

	cfg.sendMetrics = true; // Send metrics
	cfg.renderMode = false; // Don't render

	// Make the learner with the environment creation function and the config we just made
	Learner* learner = new Learner(EnvCreateFunc, cfg, StepCallback);

	// Start learning!
	learner->Start();

	return EXIT_SUCCESS;
}
