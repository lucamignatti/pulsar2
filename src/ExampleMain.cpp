#include <GigaLearnCPP/Learner.h>

#include <RLGymCPP/Rewards/CommonRewards.h>
#include <RLGymCPP/Rewards/ZeroSumReward.h>
#include <RLGymCPP/TerminalConditions/NoTouchCondition.h>
#include <RLGymCPP/TerminalConditions/GoalScoreCondition.h>
#include <RLGymCPP/ObsBuilders/DefaultObs.h>
#include <RLGymCPP/ObsBuilders/AdvancedObs.h>
#include <RLGymCPP/StateSetters/KickoffState.h>
#include <RLGymCPP/StateSetters/CombinedState.h>
#include <RLGymCPP/StateSetters/RandomState.h>
#include <RLGymCPP/ActionParsers/DefaultAction.h>

using namespace GGL; // GigaLearn
using namespace RLGC; // RLGymCPP

struct EnvResetInfo {
	bool isKickoffReset = true;
};

class ResetModeStateSetter : public StateSetter {
public:
	StateSetter* child;
	EnvResetInfo* resetInfo;
	bool isKickoffReset;

	ResetModeStateSetter(StateSetter* child, EnvResetInfo* resetInfo, bool isKickoffReset) :
		child(child), resetInfo(resetInfo), isKickoffReset(isKickoffReset) {
	}

	virtual void ResetArena(Arena* arena) override {
		resetInfo->isKickoffReset = isKickoffReset;
		child->ResetArena(arena);
	}
};

class KickoffOnlyReward : public Reward {
public:
	Reward* child;

	KickoffOnlyReward(Reward* child) : child(child) {}

	virtual void Reset(const GameState& initialState) override {
		child->Reset(initialState);
	}

	virtual void PreStep(const GameState& state) override {
		EnvResetInfo* resetInfo = (EnvResetInfo*)state.userInfo;
		if (resetInfo && !resetInfo->isKickoffReset)
			return;

		child->PreStep(state);
	}

	virtual std::vector<float> GetAllRewards(const GameState& state, bool isFinal) override {
		EnvResetInfo* resetInfo = (EnvResetInfo*)state.userInfo;
		if (resetInfo && !resetInfo->isKickoffReset)
			return std::vector<float>(state.players.size(), 0);

		return child->GetAllRewards(state, isFinal);
	}
};

// Create the RLGymCPP environment for each of our games
// Cycles through four modes every 4 arenas so the fleet is evenly split:
//   index % 4 == 0 → 1v1 SOCCAR
//   index % 4 == 1 → 2v2 SOCCAR
//   index % 4 == 2 → 3v3 SOCCAR
//   index % 4 == 3 → Heatseeker 2v2
// AdvancedObs(maxPlayers=3) pads every obs to the same fixed size regardless of
// how many cars are actually present, so all modes share one policy network.
EnvCreateResult EnvCreateFunc(int index) {
	EnvResetInfo* resetInfo = new EnvResetInfo();

	int modeIdx = index % 4;
	int playersPerTeam;
	GameMode gameMode;
	switch (modeIdx) {
		case 0:  playersPerTeam = 1; gameMode = GameMode::SOCCAR;     break; // 1v1
		case 1:  playersPerTeam = 2; gameMode = GameMode::SOCCAR;     break; // 2v2
		case 2:  playersPerTeam = 3; gameMode = GameMode::SOCCAR;     break; // 3v3
		default: playersPerTeam = 2; gameMode = GameMode::HEATSEEKER; break; // Heatseeker 2v2
	}

	// Fraction of individual event rewards (shots, saves, strong touches) shared with the
	// team average, so enabling plays (passes, leave-its) earn credit too. opponentScale 0
	// keeps these team-distributed WITHOUT making them zero-sum. In 1v1 the team average
	// equals the player's own reward, so this is mathematically a no-op there.
	constexpr float TEAM_SPIRIT = 0.4f;

	// Chase incentive scaled down in team modes: paying every player to drive at the ball
	// is the classic cause of double/triple commits in 2v2/3v3.
	float chaseWeight;
	switch (playersPerTeam) {
		case 1:  chaseWeight = 6.0f; break;
		case 2:  chaseWeight = 3.0f; break;
		default: chaseWeight = 2.0f; break;
	}

	std::vector<WeightedReward> rewards = {

		// Ball-goal
		{ new ZeroSumReward(new VelocityBallToGoalReward(), 1), 5.0f },

		// Ungated aerial bootstrap; completed aerial touches/followups remain GCRL-gated.
		{ new AerialCommitReward(), 6.0f },

		// Boost
		{ new PickupBoostReward(), 10.f },
		{ new SaveBoostReward(), 0.2f },

		// Game events — shot/save callbacks are NULL in Heatseeker (no event tracker),
		// so ShotReward and SaveReward naturally return 0 there.
		{ new ZeroSumReward(new ShotReward(), TEAM_SPIRIT, 0.0f), 15.f },
		{ new ZeroSumReward(new ShotOnFrameReward(), TEAM_SPIRIT, 0.0f), 35.f },
		{ new ZeroSumReward(new SaveReward(), TEAM_SPIRIT, 0.0f), 20.f },
		{ new ZeroSumReward(new KickoffOnlyReward(new KickoffTouchReward(3.0f)), 0.0f), 5.0f },
		{ new GoalReward(), 275 },

		// Getting demoed costs the team a defender for 3 seconds
		{ new DemoedPenalty(), 5.f },

		// time penalty
		{ new TimePenalty(-0.075f), 1.0f }
	};

	std::vector<WeightedReward> gcrlGatedRewards = {

		// Ground/contact rewards are filtered by GCRL terminal progress.
		// (Weight cut from 90: the sigmoid gate revert roughly doubled this reward's
		// expected value vs the signed-tanh runs.)
		{ new ZeroSumReward(new StrongTouchReward(20, 100), TEAM_SPIRIT, 0.0f), 60.f }
	};

	std::vector<WeightedReward> curriculumRewards = {

		// Temporary chase incentive; GCRL gate suppresses steps with below-average progress toward goal.
		{ new VelocityPlayerToBallReward(), chaseWeight }
	};

	std::vector<WeightedReward> aerialGCRLGatedRewards = {

		// Aerial rewards use a slower gate anneal so bootstrap signals are not filtered out early.
		// NOTE: No unconditional AirReward here -- per-step "be airborne" income taught the bots
		// to hover (80%+ air time, ground-level touch heights). The rewards below all require
		// the ball to actually be up and/or productive contact.
		{ new HeightWeightedAerialApproachReward(), 1.5f },
		{ new UsefulAirTouchReward(), 25.0f },
		{ new SecondTouchReward(), 12.0f },
		{ new FlipResetFollowupReward(), 8.0f },
		{ new UsefulFlickReward(), 8.0f }
	};

	std::vector<WeightedReward> aerialCurriculumRewards = {

		// Temporary capped bootstrap: go make progress toward high balls while airborne.
		{ new ExponentialAerialBallProgressReward(), 1.0f }
	};

	std::vector<TerminalCondition*> terminalConditions = {
		new NoTouchCondition(10),
		new GoalScoreCondition()
	};

	auto arena = Arena::Create(gameMode);
	for (int i = 0; i < playersPerTeam; i++) {
		arena->AddCar(Team::BLUE);
		arena->AddCar(Team::ORANGE);
	}

	EnvCreateResult result = {};
	result.actionParser = new DefaultAction();
	// maxPlayers=3 → fixed obs size for all modes; extraObs tells the policy which mode
	// it's in (a heatseeker kickoff is otherwise indistinguishable from a soccar one)
	result.obsBuilder = new AdvancedObs(
		3, true, true, { 0.25f, 0.5f, 1.0f, 1.5f, 2.0f },
		{ gameMode == GameMode::HEATSEEKER ? 1.0f : 0.0f, playersPerTeam / 3.0f });
	result.stateSetter = new CombinedState({
		{ new ResetModeStateSetter(new KickoffState(), resetInfo, true), 0.80f },
		{ new ResetModeStateSetter(new RandomState(true, true, false), resetInfo, false), 0.20f }
	});
	result.terminalConditions = terminalConditions;
	result.rewards = rewards;
	result.gcrlGatedRewards = gcrlGatedRewards;
	result.curriculumRewards = curriculumRewards;
	result.aerialGCRLGatedRewards = aerialGCRLGatedRewards;
	result.aerialCurriculumRewards = aerialCurriculumRewards;
	result.userInfo = resetInfo;

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

				bool highBall = state.ball.pos.z >= 650;
				report.AddAvg("Aerial/High Ball Ratio", highBall);
				report.AddAvg("Aerial/Airborne On High Ball", highBall && !player.isOnGround);
				report.AddAvg("Aerial/Jump Action Ratio", player.prevAction.jump);
				report.AddAvg("Aerial/Air Boost Ratio", !player.isOnGround && player.prevAction.boost);
				report.AddAvg("Aerial/High Air Touch Ratio", player.ballTouchedStep && !player.isOnGround && state.ball.pos.z >= 500);

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
	// Let GigaLearnCPP select the best available device automatically.
	cfg.deviceType = LearnerDeviceType::AUTO;


	cfg.tickSkip = 4;
	cfg.actionDelay = cfg.tickSkip - 2;

	// Play around with this to see what the optimal is for your machine, more games will consume more RAM
	cfg.numGames = 5120;

	// Leave this empty to use a random seed each run
	// The random seed can have a strong effect on the outcome of a run
	cfg.randomSeed = 67;
	cfg.savePolicyVersions = true;
	cfg.trainAgainstOldVersions = true;
	cfg.trainAgainstOldChance = 0.25f;
	cfg.tsPerVersion = 25'000'000;
	cfg.maxOldVersions = 32;

	int tsPerItr = 150'000;
	cfg.ppo.tsPerItr = tsPerItr;
	cfg.ppo.batchSize = tsPerItr;
	cfg.ppo.miniBatchSize = 37'500; // Lower this if too much VRAM is being allocated
	cfg.ppo.overbatching = true;
	// 4 optimizer steps per batch (one per minibatch) instead of one accumulated step.
	// If Mean KL Divergence sits above ~0.01, lower policyLR or flip this off.
	cfg.ppo.stepPerMiniBatch = true;

	// Using 2 epochs seems pretty optimal when comparing time training to skill
	// Perhaps 1 or 3 is better for you, test and find out!
	cfg.ppo.epochs = 2;

	// This scales differently than "ent_coef" in other frameworks
	// This is the scale for normalized entropy, which means you won't have to change it if you add more actions
	cfg.ppo.entropyScale = 0.035f;
	cfg.ppo.adaptiveEntropy = true;
	// 0.70 was unreachable: the controller pinned at maxEntropyScale for entire runs and the
	// entropy term dwarfed the PPO surrogate. 0.60 is still exploratory for ~40 actions.
	cfg.ppo.targetEntropy = 0.60f;
	cfg.ppo.adaptiveEntropyLR = 5e-3f;
	cfg.ppo.minEntropyScale = 0.0f;
	cfg.ppo.maxEntropyScale = 0.10f;

	// Rate of reward decay
	// At tickSkip 4 (30 steps/sec) 0.995 gives a ~6.6s credit half-life -- the same time
	// horizon 0.99 gives at tickSkip 8. (0.99 here was only ~3.3s, too short to propagate
	// goal credit back through buildup play.)
	cfg.ppo.gaeGamma = 0.995;

	// Good learning rate to start
	cfg.ppo.policyLR = 4e-4;
	cfg.ppo.criticLR = 2e-4;
	cfg.ppo.gcrlLR = 2e-4;
	cfg.ppo.sorsLR = 2e-4f;

	// faster but broken on rocm
	cfg.ppo.useHalfPrecision = false;

	// Three quasimetric critics learn positioning/anticipation contrastively (InfoNCE)
	// over hindsight-relabeled future ball goals, and blend their advantage into the
	// policy gradient on top of the reward-driven GAE advantage. The dense rewards above
	// teach mechanics; GCRL teaches where to be.
	cfg.ppo.useGCRL = true;
	cfg.ppo.gcrlAdvScale = 0.75f;   // Target GCRL advantage weight after annealing
	cfg.ppo.gcrlAdvScaleAnnealStart = -1; // Start ramping from the current checkpoint/load point
	cfg.ppo.gcrlAdvScaleAnnealSteps = 100'000'000;
	// The anti critic now scores own-goal danger (it queries the own-goal target instead of
	// duplicating the goal critic), so this weighs real defensive signal, not twin-network
	// noise. Start moderate; raise if defense lags.
	cfg.ppo.gcrlAntiScale = 0.4f;
	cfg.ppo.gcrlCarScale = 0.5f;     // car-positioning critic weight in the GCRL advantage
	cfg.ppo.gcrlBaselineSamples = 4; // counterfactual action baseline (K shuffled-action forwards)
	cfg.ppo.gcrlHorizon = 128;       // max HER goal offset in steps (upper bound; ~4.3s at tickSkip 4)
	cfg.ppo.gcrlMinHorizon = 32;     // min HER goal offset in steps (lower bound; ~1.05s at tickSkip 4)
	cfg.ppo.gcrlUseVariableHER = true; // sample goal offset uniformly in [minH, H]; else fixed H
	cfg.ppo.gcrlTau = 0.05f;         // embedding temperature; 0.02 scaled logits 50x and was the likely NaN source
	cfg.ppo.gcrlReprDim = 256;       // phi/psi embedding dimension (the metric-space capacity)
	cfg.ppo.gcrlInfoNCECoef = 0.75f;  // weight of the InfoNCE loss in the combined objective
	cfg.ppo.gcrlInfoNCEPenalty = 0.01f; // logsumexp penalty inside InfoNCE
	cfg.ppo.gcrlVarReg = 0.3f;       // embedding variance regularization (anti-collapse)
	cfg.ppo.gcrlInfoSubSample = 256; // contrastive sub-batch size
	cfg.ppo.useGCRLRewardGate = true;
	cfg.ppo.gcrlRewardGateInfluence = 1.0f;
	cfg.ppo.gcrlRewardGateAnnealStart = -1;
	cfg.ppo.gcrlRewardGateAnnealSteps = 100'000'000;
	cfg.ppo.gcrlRewardGateSharpness = 1.0f;
	cfg.ppo.gcrlRewardGateAntiScale = 0.85f;
	cfg.ppo.gcrlRewardGateTargetVel = 1200.0f;
	cfg.ppo.gcrlRewardGateLookahead = 32;
	cfg.ppo.gcrlAerialRewardGateInfluence = 1.0f;
	cfg.ppo.gcrlAerialRewardGateStartInfluence = 0.2f;
	cfg.ppo.gcrlAerialRewardGateAnnealStart = -1;
	cfg.ppo.gcrlAerialRewardGateAnnealSteps = 1'000'000'000;
	cfg.ppo.curriculumRewardScale = 1.0f;
	cfg.ppo.curriculumRewardAnnealStart = -1;
	cfg.ppo.curriculumRewardAnnealSteps = 800'000'000;
	cfg.ppo.aerialCurriculumRewardScale = 1.0f;
	cfg.ppo.aerialCurriculumRewardAnnealStart = -1;
	cfg.ppo.aerialCurriculumRewardAnnealSteps = 800'000'000;

	cfg.ppo.useSORS = false; // DISABLED
	cfg.ppo.sorsRewardScale = 0.10f;
	cfg.ppo.sorsRewardScaleAnnealStart = -1; // Train SORS immediately, but delay reward influence
	cfg.ppo.sorsRewardScaleAnnealSteps = 100'000'000;
	cfg.ppo.sorsRewardClipRange = 1.0f;
	cfg.ppo.sorsWarmupIters = 8;
	cfg.ppo.sorsTrainPairs = 4096;
	cfg.ppo.sorsMaxReplayWindows = 20000;
	cfg.ppo.sorsMinLabelDelta = 0.25f;
	cfg.ppo.sorsWindowBefore = 45;
	cfg.ppo.sorsWindowAfter = 30;
	cfg.ppo.sorsLabels = {
		{ new AirTouchSORSLabel(500), 1.0f },
		{ new FlipResetSORSLabel(500), 1.0f },
		{ new PostResetTouchSORSLabel(500), 1.0f },
		{ new WavedashSORSLabel(), 1.0f },

		{ new UsefulBallDeltaSORSLabel(30, 300, 300), 0.5f },
		{ new PossessionOrFirstToBallSORSLabel(45), 0.5f },
		{ new GoodRecoverySORSLabel(30, 0.5f), 0.3f },
		{ new ShotCreatedSORSLabel(60), 1.0f },
		{ new GoalForSORSLabel(90), 2.0f },
		{ new GoalAgainstSORSLabel(90), -2.0f },
		{ new LostPossessionSORSLabel(45), -0.7f },
		{ new BadRecoverySORSLabel(30, 0.0f), -0.7f }
	};

	cfg.ppo.sharedHead.layerSizes = {};
	cfg.ppo.policy.layerSizes = { 512, 512, 256 };
	cfg.ppo.critic.layerSizes = { 768, 512, 256 };
	cfg.ppo.gcrlCritic.layerSizes = { 768, 512, 256 };
	cfg.ppo.sorsReward.layerSizes = { 256, 256, 128 };

	cfg.skillTracker.enabled = true;
    cfg.skillTracker.numArenas = 24;        // eval arenas, keep near CPU thread count
    cfg.skillTracker.simTime = 45;          // seconds simulated per eval run
    cfg.skillTracker.maxSimTime = 240;      // continuation cap if too few goals happen
    cfg.skillTracker.updateInterval = 16;   // run Elo every N training iterations
    cfg.skillTracker.ratingInc = 5;         // Elo K-ish scale per goal
    cfg.skillTracker.initialRating = 0;
    cfg.skillTracker.deterministic = true;  // optional: rate greedy policy instead of sampled policy


	// ── Evolution Strategies (EGGROLL) ──
	// Periodically perturbs a large population of policy variants, plays full games against the
	// current policy, and nudges the live weights toward what actually wins (goal differential,
	// reward breaks ties). A gradient-free booster that helps PPO escape local minima.
	// Like the skill tracker, it only runs when renderMode = false.
	cfg.evolutionStrategy.enabled = false;
	cfg.evolutionStrategy.populationSize = 8192;   // members per ES step (split into populationSize/numGames full-game rollouts)
	cfg.evolutionStrategy.lowRankRank = 4;         // EGGROLL low-rank perturbation rank
	cfg.evolutionStrategy.sigma = 0.02f;           // perturbation scale (exploration radius in weight space)
	cfg.evolutionStrategy.learningRate = 0.01f;    // ES step size; watch ES/UpdateNorm, raise if too gentle
	cfg.evolutionStrategy.weightDecay = 0.005f;
	cfg.evolutionStrategy.gameSimTime = 60.0f;     // sim-seconds per full game (longer = better scoring signal, slower)
	cfg.evolutionStrategy.maxSimTime = 90.0f;      // hard cap per game
	cfg.evolutionStrategy.updateInterval = 25;    // run ES every N training iters; main dial for ES wall-time cost (watch ES/StepTime)
	cfg.evolutionStrategy.antithetic = true;       // evaluate +/- pairs (variance reduction)
	cfg.evolutionStrategy.rankNormalize = true;    // centered-rank fitness shaping
	// cfg.evolutionStrategy.scope = EvolutionStrategyConfig::Scope::POLICY_ONLY; // default; only perturb the policy head


	auto optim = ModelOptimType::MUON;
	cfg.ppo.policy.optimType = optim;
	cfg.ppo.critic.optimType = optim;
	cfg.ppo.sharedHead.optimType = optim;
	cfg.ppo.gcrlCritic.optimType = optim;
	cfg.ppo.sorsReward.optimType = optim;

	auto activation = ModelActivationType::LEAKY_RELU;
	cfg.ppo.policy.activationType = activation;
	cfg.ppo.critic.activationType = activation;
	cfg.ppo.sharedHead.activationType = activation;
	cfg.ppo.gcrlCritic.activationType = activation;
	cfg.ppo.sorsReward.activationType = activation;

	bool addLayerNorm = true;
	cfg.ppo.policy.addLayerNorm = addLayerNorm;
	cfg.ppo.critic.addLayerNorm = addLayerNorm;
	cfg.ppo.sharedHead.addLayerNorm = addLayerNorm;
	// Optional: the embedding is L2-normalized at scoring time, so trunk LayerNorm is less
	// important here than for policy/critic. Off by default; flip to addLayerNorm to match.
	cfg.ppo.gcrlCritic.addLayerNorm = false;
	cfg.ppo.sorsReward.addLayerNorm = addLayerNorm;

	cfg.sendMetrics = true; // Send metrics
	cfg.renderMode = false; // Don't render
	cfg.ppo.deterministic = cfg.renderMode;

	// Make the learner with the environment creation function and the config we just made
	Learner* learner = new Learner(EnvCreateFunc, cfg, StepCallback);

	// Start learning!
	learner->Start();

	return EXIT_SUCCESS;
}
