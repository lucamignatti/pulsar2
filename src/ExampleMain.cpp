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
		// Dense GOALWARD-STRIKE gradient (xddib2kd fix; weight bumped 50->100 per the wucwxpfx
		// diagnosis: touch-PRESENCE rewards were ~70% of income and finishing was only ~6% ->
		// possession/dribble drift, goal speed decaying). Rewards the goalward ball-speed INCREASE the
		// agent's touch produces -> breaks the approach-then-idle peak (the CONTROL critic's ball-hold
		// pull) AND pays for HARD goalward contact over mere contact.
		{ teamMixed(new GoalwardImpactReward()), 100.f },
		// Touch impulses — TRIMMED (StrongTouch 60->40, TouchBall 30->15) so finishing outweighs
		// presence; a small TouchBall floor is kept for cold-start "go touch the ball" bootstrap.
		{ teamMixed(new StrongTouchReward(20.f, 130.f)), 40.f }, // |dBallVel| in [20,130]kph -> committed strike
		{ teamMixed(new TouchBallReward()), 15.f },          // any contact (touch-volume engine, trimmed floor)
		{ teamMixed(new AerialTouchReward()), 12.f },        // genuine aerial touch (the reward-stream aerial mechanism)
		// The ONLY sustained LEVEL term: absolute approach velocity, UNWRAPPED (cannot telescope-hug-farm).
		// 0.5 -> 6.0: this is now the COLD-START APPROACH BOOTSTRAP. It used to be small because the egocentric
		// -ball CAR critic drove approach ("CONTROL leads cold-start approach, edge ~1.0"); making the car critic
		// BALL-AGNOSTIC removed that, and 0.5 alone is too weak (instantaneous momentum, ~97% value-absorbed) to
		// bootstrap a cold policy -> run bsx969sr froze at touch ratio ~0.0003 / rating ~46. 6.0 is the proven
		// bootstrap value (g7jf6cwc/ryp4gxwv reached 560); VPB->0 as the car reaches the ball, so it can't be
		// stall-farmed. The positioning push past the chase plateau now comes from the goal potential + anchors.
		{ new VelocityPlayerToBallReward(), 6.0f },
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

	// Overlap collection with the PPO update (IMPALA-style depth-1 pipeline): collect rollout N+1 on a
	// background thread (frozen actor clone + RSNorm snapshot) while the main thread runs Learn on rollout N.
	// Hides the shorter of collect/consume behind the longer; PPO's clipped ratio absorbs the 1-iter staleness.
	cfg.overlapCollection = true;

	cfg.tickSkip = 8;
	cfg.actionDelay = cfg.tickSkip - 1; // Normal value in other RLGym frameworks

	// 1v1 SOCCAR. Raised 5120->8192 to spend the idle CPU headroom (the 7900X sat ~20% even at peak) on more
	// parallel collection. LEARNING-AFFECTING (unlike the chunk knobs): more arenas = more decorrelated
	// samples, and with tsPerItr scaled below the per-update batch grows (fewer optimizer steps per million
	// timesteps). Judge on rating-per-WALLCLOCK, not SPS alone; revert to 5120/150k if sample-efficiency drops.
	cfg.numGames = 8192;

	// Leave this empty to use a random seed each run
	// The random seed can have a strong effect on the outcome of a run
	cfg.randomSeed = 67;

	// Scaled 150k->240k with numGames so per-env segment length holds (~240k/8192 ~= 29 steps/env) while the
	// per-update batch grows to amortize the overhead-bound consumption over more samples. LEARNING-AFFECTING.
	int tsPerItr = 240'000;
	cfg.ppo.tsPerItr = tsPerItr;
	cfg.ppo.batchSize = tsPerItr;
	// Pure VRAM chunking, NOT a learning knob: grads accumulate across minibatches (each loss pre-scaled by
	// batchSizeRatio) and the optimizer steps once per batch, so any miniBatchSize gives the IDENTICAL
	// gradient -- bigger just means fewer/larger GPU kernels + fewer per-minibatch syncs. MUST divide
	// batchSize (=tsPerItr), so it's capped there; = batchSize gives the largest chunk. (Overbatching extends
	// the actual rollout to ~330k, which then runs as 240k + ~90k.) To use more VRAM per kernel, raise
	// tsPerItr (raises this cap too); watch nvidia-smi for the 16 GB ceiling.
	cfg.ppo.miniBatchSize = 240'000;

	// Blackwell/RTX 5080 perf levers (all enabled). useHalfPrecision: BF16 inference (collection +
	// GAE value pred). useAMP: BF16 autocast on the PPO update (matmuls -> tensor cores; softmax/exp/
	// log/layer_norm/losses auto-promote to fp32; master weights fp32, no GradScaler). pinBatchMemory:
	// pinned host batches so the non_blocking H2D copies are real async DMA. Watch Mean KL / Clip
	// Fraction / touch / non-finite counts on the first run since BF16 shifts numerics.
	cfg.ppo.useHalfPrecision = true;
	cfg.ppo.useAMP = true;
	// pinBatchMemory left OFF: it pinned the whole rollout (~GB) in page-locked host RAM every iteration,
	// which the kernel can't compact -> kcompactd soft-lockups on the training box. Marginal benefit
	// without prefetch anyway. Revisit only with a small reused minibatch-sized pinned staging buffer.
	cfg.ppo.pinBatchMemory = false;

	// 2 epochs (the value this comment has always recommended). epochs=1 gave the
	// policy only a single gradient step per iteration's data AND made Mean KL a
	// meaningless ~0 (ratio==1 by construction, so reported KL was just float
	// noise), hiding whether the policy was actually moving.
	cfg.ppo.epochs = 3; // Run1: extract KL headroom (KL ~0.005 vs clip 0.2). Judge on rating-per-WALLCLOCK (SPS drops ~14k->~9-10k); KL>0.02 sustained = revert.

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
	cfg.ppo.criticLR = 3e-4; // Run1: value head only (Critic Loss drifting 0.006->0.07 w/ reward scale). Trunk stays at policyLR via min(policyLR,criticLR) in SetLearningRates.

	cfg.ppo.sharedHead.layerSizes = { 256, 256 };
	cfg.ppo.policy.layerSizes = { 256, 256, 256 };
	cfg.ppo.critic.layerSizes = { 256, 256, 256 };

	// GCRL magnitude-blend, per-critic policy-gradient contribution is (taken - baseline)/spread -- NOT
	// unit-renormalized -- so a critic that can't yet discriminate the action self-attenuates to ~0 (no
	// noise injection, no gate). The CAR critic is now BALL-AGNOSTIC: its goal is the car's OWN future
	// kinematic + mechanic state (velocity + orientation + angular velocity + air-control flags), so it learns
	// car CONTROL -- speedflips, wavedashes, aerial control, recoveries -- with no reason to drive at the ball (the old
	// egocentric-ball goal made it a ball magnet). Approach is left to VelocityPlayerToBall + the reward
	// stack; positioning to the goal-potential critic. Watch GCRL/Car Separation + GCRL/Car Edge Mean.
	cfg.ppo.contrastiveGoal.enabled = true;
	cfg.ppo.contrastiveGoal.useCarCritic = true;
	cfg.ppo.contrastiveGoal.carGoalInputSize = 15; // car self-state goal: vel(3)+fwd(3)+up(3)+angVel(3)+airflags(3: onGround,hasFlipOrJump,isFlipping)
	// Keep the goal critic's ACTION-EDGE coupling OFF -- it is action-INERT far-field (edge ~0.04) and its
	// batch-normed ~0 edge reinflates to ~30% unit-std NOISE. But re-introduce the goal critic as a POSITIONING
	// signal via its POTENTIAL: useGoalPotential consumes Phi(s) = action-marginalized reachability of the net
	// as a telescoping shaping term gamma*Phi(s')-Phi(s). This is the dense positioning gradient the egocentric
	// CAR critic structurally can't provide (and the mirror-chase plateau needs), without the noise OR a
	// farmable raw positioning reward. The critic is trained for Phi (useGoalPotential), just not edge-coupled.
	cfg.ppo.contrastiveGoal.useGoalCritic = false;
	cfg.ppo.contrastiveGoal.useGoalPotential = true;
	// 0.3 -> 0.1: at 0.3 the potential was ~26% of the policy-gradient magnitude, but the goal critic is
	// near-chance (categorical accuracy ~0.15) for a COLD bot -- the ball barely moves, so its reachability
	// manifold is degenerate -> that 26% was mostly NOISE (the documented freeze mechanism, cf 0hd1s2ne).
	// Positioning shaping is premature before the bot can touch the ball; 0.1 bounds the cold-phase noise.
	// It self-heals as VPB bootstraps ball contact (the goal critic then gets real data); re-raise once
	// GCRL Categorical Accuracy climbs and touch ratio is off the floor.
	cfg.ppo.contrastiveGoal.gcrlGoalPotentialScale = 0.1f;
	cfg.ppo.contrastiveGoal.gcrlLambda = 0.3f;                  // GCRL-vs-reward blend weight (held after warmup)
	cfg.ppo.contrastiveGoal.gcrlLambdaWarmupSteps = 30'000'000; // short bootstrap ramp, then hold
	cfg.ppo.contrastiveGoal.carHerMaxOffset = 30;               // car critic: ~2s self-state window (covers flips/wavedashes/short aerials)
	cfg.ppo.contrastiveGoal.criticLR = 3e-4f;
	cfg.ppo.contrastiveGoal.criticEpochs = 1;
	// THE consumption lever: the InfoNCE critic train was 2.17s = 81% of "PPO Learn" at bs=256 (~960 serial
	// minibatches over the rollout). The effective InfoNCE minibatch is min(criticMiniBatchSize,
	// infoSubSample) (ContrastiveGoalLearner.cpp:128); infoSubSample=0 below DISABLES that cap, so this is
	// now the SOLE lever (it was silently clamped to 512 -- the old "1024/2048" values were no-ops).
	// FIRST REAL large-B test: total InfoNCE matmul work scales ~linearly with B (= numRows * B * reprDim),
	// so 512->2048 is ~4x the dominant [B,B] matmul compute but ~4x fewer serial launches -- net effect
	// UNKNOWN (the audit says this path is only PARTIALLY launch-bound). Watch Consumption/GCRL Train Time:
	// if it balloons we're compute-bound (dial back, or cut the TD K-sample multiplier); if it holds/drops,
	// push to 4096. NOTE: GCRL/Car Categorical Accuracy MECHANICALLY falls as B grows (1-of-B is a harder
	// pick), so a lower number at 2048 is NOT a regression -- judge the critic by the policy/score signal.
	cfg.ppo.contrastiveGoal.criticMiniBatchSize = 2048;
	// Disable the infoSubSample clamp so criticMiniBatchSize is the true effective minibatch.
	cfg.ppo.contrastiveGoal.infoSubSample = 0;
	// GCRL scoring chunk size. PURE chunking (numerically identical): bigger = fewer/larger kernels in the
	// ~17-pass counterfactual-baseline scoring loop. 4096->32768 collapses ~50 chunks/pass to ~6. Tiny
	// activations (psi/phi are small) so it fits easily; push higher if Consumption/GCRL Score Time is still high.
	cfg.ppo.contrastiveGoal.policyScoreBatchSize = 32768;
	// Goal critic HER window. The old short cap (15) existed because the action-EDGE went action-invalid at
	// long horizons -- but we now consume the goal critic as a VALUE/potential (Phi), not the edge, so a
	// LONGER horizon is correct: it makes Phi a genuine "can I score from here soon" reachability over a
	// full play (~4s at tickSkip 8), which is what teaches positioning. Keep the goalward bias off.
	cfg.ppo.contrastiveGoal.herMaxOffset = 60;
	cfg.ppo.contrastiveGoal.herGoalwardBias = 0.f;
	// xddib2kd fix: lower GCRL's target share of the policy gradient (was 1:1) so the egocentric
	// CONTROL critic's continuous ball-HOLD pull stops dominating the strike action and the reward
	// (now with the goalward-strike impulse) gets more relative weight. CONTROL still leads cold-start
	// approach (it climbed to edge ~1.0). If approach-then-idle persists, lower further toward 0.3.
	cfg.ppo.contrastiveGoal.gcrlRatioTarget = 0.5f;
	// tau 0.05, VICReg, masked-random K16 baseline, the always-on variance-weight + ratio-pinned lambda
	// controller + RenormToStd are config defaults (PPOLearnerConfig.h). ANTI critic still flagged off.
	// FORK2 TD-contrastive on GOALSHORT: TESTED in run 2fkzih10 and REVERTED. It FAILED its falsification
	// gate (Goal Edge stayed ~0.044, never near the 0.15 target, over 71M ts at r=1) and DEGRADED the Goal
	// head (Categorical Accuracy 0.20->0.10 while MC climbs to 0.88). Root cause is STRUCTURAL: the MC
	// baseline pins Goal Edge ~0.03 to 984M ts (successor-feature asymptote) and the pi-soft bootstrap value
	// is itself action-invariant (SoftValue EntropyFrac 0.999), so TD has no per-action signal to inject.
	// The far-field world-ball critic is the WRONG geometry; the only action-discriminating critic is the
	// egocentric CAR critic (edge ~0.78). Retuning the ramp will NOT help. Code stays (flag-gated) for a
	// future geometry-changed revival, not a re-enable on this critic.
	cfg.ppo.contrastiveGoal.useTDContrastive = false;
	// FORK2 Part C: GOALSHORT HER goal = 0.5 achieved-future + 0.5 net-directed scoring goal (synthetic
	// rows are excluded from the TD bootstrap). Populates real near-net states + the scoring-goal manifold.
	cfg.ppo.contrastiveGoal.scoringGoalMixFrac = 0.5f;

	// TRIAD-NATIVE self-play (fork 10): vs a rolling frozen-checkpoint pool (no ES; BC-seeded once the
	// offline warmstart lands -- until then version 0 is the initial policy). Breaks mutual-idle equilibria.
	// 0.15->0.35: the mirror-chase plateau (run 04zisffz ~rating 300) needs more non-self exposure to make
	// chasing stop being near-optimal; pool DIVERSITY is the bigger lever, hence the anchor set below.
	cfg.trainAgainstOldVersions = true;
	cfg.trainAgainstOldChance = 0.35f;
	cfg.maxOldVersions = 20;
	cfg.tsPerVersion = 25'000'000;
	// "Gold anchor" opponents: keep a few strong, temporally-spaced past selves around permanently so the
	// pool isn't just 20 near-identical recent chasers. Half of old-version games face an anchor.
	cfg.maxAnchorVersions = 3;
	cfg.anchorSelectChance = 0.5f;
	cfg.anchorPromoteMargin = 25.0f;
	cfg.anchorMinTsSpacing = 100'000'000;

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
