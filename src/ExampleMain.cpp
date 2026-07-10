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

// 2.6: a faithful revert to the last GOOD state of run 9uz761ua's lineage, on the current
// (fast) codebase.
//
// The forensics (wandb 9uz761ua + git times): the run climbed hard, 0 -> ~1004 Rating/1v1 in
// its first ~10B steps, then at wandb step ~54.4k / ~10.9B timesteps (2026-07-04 18:08 UTC) it
// was resumed from an older checkpoint onto a rebuild that had just ENABLED the proposer + drill
// machinery (commits db1dd87 "Add drill bank and proposer module" -> bc27a81 "Enable Stage 2 and
// Stage 3 proposer settings"). From that point Rating improvement decelerated ~10x - it only
// crept 1004 -> ~1190 over the *next 19B* steps. The last clean commit before that regression is
// cf993b7 "Enable reachability gating for farmable rewards": pure PBRS + reachability GATING,
// no proposer, no drill bank, no HRL. This file reproduces cf993b7's exact reward + training
// config, but on the current codebase so we keep the post-regression SPEED commits (3d344fe/
// 2211cce ~2x throughput, plus the reachability chunk-size tuning) - "current code, config
// reverted", as requested.
//
// What is deliberately NOT here vs. the current (HEAD) config: the goal proposer, the drill bank
// / DrillSetter, the car-proposer head, and the recent uncommitted edit that turned on the
// self-play LEAGUE (trainAgainstOldVersions) and halved tsPerItr to 100k. None of those were in
// the proven-good era.

// SURGICAL-7 reward stack (Nexto-ratio potentials + impulse-scaled touch height), verbatim from
// cf993b7 - the reward that got the bot to ~1004 Elo before the proposer was bolted on. NOT
// touched: the regression was the proposer, not the reward, so re-establishing this as a clean
// baseline means changing it as little as possible. (ShotReward/SaveReward from 2.5 are a
// separate, unproven experiment to layer on top AFTER this baseline re-validates, not part of it.)
std::vector<WeightedReward> BuildRewards() {
	return {
		// Player->ball proximity potential (Nexto liu_dist, dist_w=0.5 x15). Exact PBRS, so the
		// chase annuity telescopes to ~0 per cycle; its 3D distance also pays climbing toward an
		// overhead ball and refunds whiffs.
		{ new BallProximityPotentialReward(), 7.5f },

		// Ball->goal potential (Nexto state_quality). Antisymmetric between teams, so ALREADY
		// zero-sum - no ZeroSum wrapper (that would silently 2x it). exp() concentrates credit at
		// the goal mouth and refunds rolled-back balls in full.
		{ new BallToGoalPotentialReward(), 75.f },

		// Touch quality (Nexto touch_accel): pays only for adding ball speed, ~10 total 0->110kph,
		// zero-sum so touches can't be co-farmed.
		{ new ZeroSumReward(new TouchAccelReward(), 0), 10.f },

		// Touch HEIGHT (Nexto touch_height x15), impulse-scaled: carries pay ~0, a real strike
		// pays full height credit, airborne strikes pay double - the aerial gradient.
		// GATED: the reachability gate mutes the aerial-juggle self-rally in states that don't
		// matter. Beta anneals from 0, 0.2 floor keeps cold aerial learning intact.
		{ new ZeroSumReward(new TouchHeightReward(), 0), 15.f, true },

		// Boost pickup, halved (big pad from empty = 4 = 2.7% of a goal).
		// GATED: don't pay for a boost-collection circuit in states where we can't win the ball
		// or score - the residual farmable surface the gate exists for.
		{ new PickupBoostReward(), 4.f, true },

		// Demo halved so the zero-sum pair swing is 75 = goal/2. No bump term (Nexto had none).
		{ new ZeroSumReward(new DemoReward(), 0.5f), 37.5f },

		// The objective. All dense income per scoring possession sums to ~35-40 (~25% of a goal),
		// none of it collectible without moving the ball toward scoring.
		{ new GoalReward(), 150 }
	};
}

// Create the RLGymCPP environment for each of our games
EnvCreateResult EnvCreateFunc(int index) {
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
	// The proven cf993b7 reset mix - effective near-ball share 0.55 (0.35 ground + 0.20 aerial
	// drill), no drill-replay slice (that came with the drill bank in the regression).
	result.stateSetter = new CombinedState({
		// Ground touch bootstrap - the proven anti-freeze state
		{ new BallNearCarState(600, 900), 0.35f },
		// Aerial drill: ball hangs at 500-1500uu drifting down, BOTH cars on a 300-1200 ring
		// with >=40 boost, facing it. Symmetric, so the race to the falling ball IS the zero-sum
		// challenge.
		{ new BallNearCarState(300, 1200, 500, 1500, 400, 40), 0.20f },
		{ new KickoffState(), 0.15f },
		// Sole source of chaotic/defensive/air-recovery states (bounds widened to corners/goal lines)
		{ new RandomState(true, true, false), 0.30f },
	});
	result.terminalConditions = terminalConditions;
	result.rewards = BuildRewards();

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
	// Keep stdout live when it isn't a terminal (kept from the current codebase - a logging fix,
	// not part of the regression). Under tools/run_trainer.sh stdout is a log file, so glibc
	// block-buffers; unitbuf flushes after every insertion so --follow behaves like a terminal.
	std::cout << std::unitbuf;

	// Initialize RocketSim with collision meshes (run from the repo/build dir;
	// provision them with tools/get_collision_meshes.sh if missing)
	RocketSim::Init("collision_meshes");

	// Make configuration for the learner
	LearnerConfig cfg = {};

	cfg.deviceType = LearnerDeviceType::GPU_CUDA;

	cfg.tickSkip = 8;
	cfg.actionDelay = cfg.tickSkip - 1; // Normal value in other RLGym frameworks

	cfg.numGames = 1024;

	// Leave this empty to use a random seed each run
	cfg.randomSeed = 123;

	// 200k, the proven-good-era value (the recent uncommitted 100k edit was NOT in the good run).
	int tsPerItr = 200'000;
	cfg.ppo.tsPerItr = tsPerItr;
	cfg.ppo.batchSize = tsPerItr;
	cfg.ppo.miniBatchSize = 200'000; // Lower this if too much VRAM is being allocated

	// BF16 inference for collection + GAE value preds. rho/gate evals request fp32 explicitly and
	// grad-enabled forwards (InfoNCE training) always run fp32, so the gate is unaffected.
	cfg.ppo.useHalfPrecision = true;

	cfg.ppo.epochs = 2;
	cfg.ppo.entropyScale = 0.035f;

	// Reachability: aux InfoNCE heads on the shared trunk + reward GATE on the two farmable terms
	// (TouchHeight, PickupBoost). This is the ONLY "smart" component - and with the proposer gone,
	// the heads have exactly one consumer: the gate. This is the arm that was live in the good era.
	cfg.ppo.reachability.enabled = true;
	cfg.ppo.reachability.gateEnabled = true;

	// Explicitly OFF - the regression. ProposerConfig defaults enabled=true upstream, so this
	// override is what actually keeps the proposer / drill bank / car-proposer / HRL machinery
	// out of this build (no drill bank is ever constructed or attached, either).
	cfg.ppo.proposer.enabled = false;

	// Wide clip (cold return-sigma is ~2-4 under this near-sparse stack; default 10 compressed the
	// first goals). 50 releases the full 150 once sigma >= 3 and bounds the tail.
	cfg.ppo.rewardClipRange = 50;

	cfg.ppo.gaeGamma = 0.99;

	cfg.ppo.policyLR = 1.5e-4;
	cfg.ppo.criticLR = 1.5e-4;

	cfg.ppo.sharedHead.layerSizes = { 256, 256 };
	cfg.ppo.policy.layerSizes = { 256, 256, 256 };
	cfg.ppo.critic.layerSizes = { 256, 256, 256 };
	cfg.ppo.reachability.phi.layerSizes = { 256, 256 };
	cfg.ppo.reachability.psi.layerSizes = { 256, 256 };
	cfg.ppo.reachability.lr = 3e-4f;

	// Speed knob kept from the post-good-era "speed 2" commit (2211cce): larger rho-read chunks
	// fill the GPU better. This only changes CHUNKING of the gate's rho reads, never their values,
	// so it's a pure throughput win with zero behavioral effect on the gate.
	cfg.ppo.reachability.scoreChunkSize = 16384;

	// Muon for the dense nets (RMS-matched, Adam LRs transfer). Reachability heads stay Adam:
	// contrastive InfoNCE embeddings train poorly under orthogonalized updates.
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

	// Skill rating: Elo-style eval matches vs saved versions (logged as Rating/1v1). Also turns on
	// savePolicyVersions. This was ON in the good era. NOTE: this only EVALUATES against old
	// versions - it does NOT train against them. The separate trainAgainstOldVersions (self-play
	// league) is deliberately left OFF: it was not part of the proven-good config (it was a recent
	// uncommitted addition). Flip it on later as its own experiment if desired.
	cfg.skillTracker.enabled = true;

	// Distinct wandb run name so this shows up as its own line, not resuming 9uz761ua.
	cfg.metricsRunName = "2.6-reachgate-pbrs";

	cfg.sendMetrics = true; // Send metrics
	cfg.renderMode = false; // Don't render

	// ---------------------------------------------------------------------------------------------
	// Basin-Racing (PSD) + QD league. Both ADDITIVE and OFF by default: with these two flags false
	// the trainer is byte-for-byte the proven 2.6 baseline. Enable to layer the outer loop on top.
	//
	// PSD alternates ordinary DESCEND (== the baseline above) with EGGROLL PROBE rounds: K antithetic
	// low-rank perturbations of the policy each get a short factor-only finetune, are scored on
	// held-out arenas, and the fitness-weighted sum of the ORIGINAL directions is folded into the
	// base weights. warmupUntilPlateau keeps pure DESCEND until Rating/1v1 stalls, so the K-cost is
	// only paid where PPO alone plateaus (resuming the live 2.6 checkpoint trips this immediately).
	// Recommended first live run: enable PSD only; add the league once probe rounds look healthy.
	cfg.psd.enabled = false;              // <- flip to true to turn on Basin-Racing
	cfg.psd.warmupUntilPlateau = true;
	cfg.psd.K = 16;
	cfg.psd.rank = 4;
	cfg.psd.sigma = 0.02f;
	cfg.psd.GProbe = 50;
	cfg.psd.GExploit = 2500;
	cfg.psd.fitnessMode = 1;              // 1 = end-of-window slope (handoff §4.1 fix), 0 = level

	cfg.league.enabled = false;           // <- flip to true to turn on the QD opponent league
	cfg.league.gridAxes = { "in_air_ratio", "field_y", "boost_economy" };
	cfg.league.binsPerAxis = 4;
	cfg.league.exploiterSlots = 2;

	// Make the learner with the environment creation function and the config we just made
	Learner* learner = new Learner(EnvCreateFunc, cfg, StepCallback);

	// Start learning!
	learner->Start();

	return EXIT_SUCCESS;
}
