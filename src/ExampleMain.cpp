#include <GigaLearnCPP/Learner.h>

#include <RLGymCPP/Rewards/CommonRewards.h>
#include <RLGymCPP/TerminalConditions/NoTouchCondition.h>
#include <RLGymCPP/TerminalConditions/GoalScoreCondition.h>
#include <RLGymCPP/ObsBuilders/AdvancedObs.h>
#include <RLGymCPP/StateSetters/KickoffState.h>
#include <RLGymCPP/StateSetters/CombinedState.h>
#include <RLGymCPP/StateSetters/RandomState.h>
#include <RLGymCPP/ActionParsers/DefaultAction.h>

#include <cstdlib>
#include <memory>

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

	virtual ~ResetModeStateSetter() override {
		delete child;
	}

	virtual void ResetArena(Arena* arena) override {
		resetInfo->isKickoffReset = isKickoffReset;
		child->ResetArena(arena);
	}
};

// Create the RLGymCPP environment for each of our games
// PHASE 1 — single mode: 1v1 SOCCAR everywhere. Every multi-mode-from-scratch run
// (rwkkyfej 25/25/25/25, tkpk0780 5/8-1v1-heavy) failed to learn ball touches by 2.5-3.4B
// steps; the only run that ever played ball (g7jf6cwc) was pure 1v1. Team modes split the
// gradient across conflicting behaviors and heatseeker pumps zero-agency goals into the
// returns/GCRL critics (its returns-STD ran ~320 vs soccar's ~115). Reintroduce 2v2/3v3/
// heatseeker from a checkpoint AFTER the touch EMA is comfortably past its gate.
// AdvancedObs(maxPlayers=3) pads the obs as if up to 3v3 were present and keeps the
// mode/team-size extraObs inputs, so the obs layout stays identical for that later phase.
EnvCreateResult EnvCreateFunc(int index) {
	EnvResetInfo* resetInfo = new EnvResetInfo();

	int playersPerTeam = 1;
	GameMode gameMode = GameMode::SOCCAR;

	std::vector<WeightedReward> rewards = {
		{ new NextoReward(), 1.0f }
	};

	std::vector<WeightedReward> gcrlGatedRewards = {};
	std::vector<WeightedReward> curriculumRewards = {};
	std::vector<WeightedReward> aerialGCRLGatedRewards = {};
	std::vector<WeightedReward> aerialCurriculumRewards = {};

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
	result.userInfoDeleter = [](void* ptr) {
		delete (EnvResetInfo*)ptr;
	};

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
	cfg.ppo.stepPerMiniBatch = true;
	// Hard speed limit: abort the iteration's remaining epochs when the batch-mean KL
	// exceeds this (SB3-style early stop). Makes minibatch stepping self-limiting.
	cfg.ppo.maxMeanKL = 0.015f;

	// Using 2 epochs seems pretty optimal when comparing time training to skill
	// Perhaps 1 or 3 is better for you, test and find out!
	cfg.ppo.epochs = 2;

	// This scales differently than "ent_coef" in other frameworks
	// This is the scale for normalized entropy, which means you won't have to change it if you add more actions
	cfg.ppo.entropyScale = 0.035f;
	cfg.ppo.adaptiveEntropy = true;
	cfg.ppo.targetEntropy = 0.70f;
	cfg.ppo.adaptiveEntropyLR = 5e-3f;
	cfg.ppo.minEntropyScale = 0.0f;
	cfg.ppo.maxEntropyScale = 0.10f;

	// Rate of reward decay
	// PHASE 2: back to 0.995. The 0.99 phase-1 value concentrated credit on approach->touch,
	// which worked (touch ratio ~0.010) -- but its ~100-step horizon is shorter than the
	// 250-450 step episodes, so the -275 concede penalty never reached the positioning
	// mistakes that caused it. Goals exist now; propagate their credit.
	cfg.ppo.gaeGamma = 0.995;

	// With stepPerMiniBatch (8 optimizer steps/iter) 4e-4 moved the policy 4-5x too fast
	// (KL 0.01-0.03, clip fraction 0.15, entropy collapse in run bgksd0wi). 1.5e-4 targets
	// the healthy ~0.03/iter update magnitude; maxMeanKL below is the hard backstop.
	cfg.ppo.policyLR = 1.5e-4;
	cfg.ppo.criticLR = 2e-4;
	cfg.ppo.gcrlLR = 2e-4;
	cfg.ppo.sorsLR = 2e-4f;

	// faster but broken on rocm
	cfg.ppo.useHalfPrecision = false;

	// Plain GCRL/HER: learn future ball-goal critics and blend their advantage into PPO.
	cfg.ppo.useGCRL = true;
	cfg.ppo.gcrlAdvScale = 0.65f;
	cfg.ppo.gcrlAdvScaleAnnealStart = 400'000'000;
	cfg.ppo.gcrlAdvScaleAnnealSteps = 100'000'000;
	cfg.ppo.gcrlAntiScale = 1.0f;
	cfg.ppo.gcrlCarScale = 0.0f;
	cfg.ppo.gcrlSurgery = false;
	cfg.ppo.gcrlBaselineSamples = 0;
	cfg.ppo.gcrlHorizon = 128;       // max HER goal offset in steps (upper bound; ~4.3s at tickSkip 4)
	cfg.ppo.gcrlMinHorizon = 32;     // min HER goal offset in steps (lower bound; ~1.05s at tickSkip 4)
	cfg.ppo.gcrlUseVariableHER = true; // sample goal offset uniformly in [minH, H]; else fixed H
	cfg.ppo.gcrlTau = 0.05f;         // embedding temperature; 0.02 scaled logits 50x and was the likely NaN source
	cfg.ppo.gcrlReprDim = 256;       // phi/psi embedding dimension (the metric-space capacity)
	cfg.ppo.gcrlInfoNCECoef = 0.75f;  // weight of the InfoNCE loss in the combined objective
	cfg.ppo.gcrlInfoNCEPenalty = 0.01f; // logsumexp penalty inside InfoNCE
	cfg.ppo.gcrlVarReg = 0.3f;       // embedding variance regularization (anti-collapse)
	cfg.ppo.gcrlInfoSubSample = 256; // contrastive sub-batch size
	cfg.ppo.useGCRLRewardGate = false;
	cfg.ppo.gcrlRewardGateInfluence = 0.0f;
	cfg.ppo.gcrlRewardGateAnnealStart = 400'000'000;
	cfg.ppo.gcrlRewardGateAnnealSteps = 100'000'000;
	cfg.ppo.gcrlRewardGateSharpness = 1.0f;
	cfg.ppo.gcrlRewardGateAntiScale = 0.85f;
	cfg.ppo.gcrlRewardGateTargetVel = 1200.0f;
	cfg.ppo.gcrlRewardGateLookahead = 32;
	cfg.ppo.gcrlAerialRewardGateInfluence = 0.0f;
	cfg.ppo.gcrlAerialRewardGateStartInfluence = 0.0f;
	cfg.ppo.gcrlAerialRewardGateAnnealStart = 400'000'000;
	cfg.ppo.gcrlAerialRewardGateAnnealSteps = 1'000'000'000;
	cfg.ppo.curriculumRewardScale = 0.0f;
	cfg.ppo.curriculumRewardAnnealStart = -1;
	cfg.ppo.curriculumRewardAnnealSteps = 2'500'000'000;
	cfg.ppo.aerialCurriculumRewardScale = 0.0f;
	cfg.ppo.aerialCurriculumRewardAnnealStart = -1;
	cfg.ppo.aerialCurriculumRewardAnnealSteps = 2'500'000'000;
	cfg.ppo.curriculumAnnealTouchRatioGate = 0.006f;
	cfg.ppo.aerialCurriculumAnnealAirTouchRatioGate = 0.0002f;

	cfg.ppo.useFrontierResets = false;
	cfg.ppo.useDifficultyHER = false;
	cfg.ppo.herCandidates = 1;
	cfg.ppo.useAdaptiveGateTargetVel = false;
	cfg.ppo.useAdaptiveStrongTouchFloor = false;
	cfg.ppo.useOptionality = false;
	cfg.ppo.optCommitReliefScale = 0.0f;
	cfg.ppo.optCommitReliefSharpness = 0.0f;
	cfg.ppo.optDeficitFloorStd = 0.0f;
	cfg.ppo.optDeficitClip = 0.0f;
	cfg.ppo.optValueWeight = 0.0f;
	cfg.ppo.optValueClip = 0.0f;
	cfg.ppo.optRefineGoals = false;
	cfg.ppo.optRefineTopK = 0;
	cfg.ppo.optRefineSteps = 0;
	cfg.ppo.optRefineMaxStates = 0;
	cfg.ppo.optRefineStepSize = 0.0f;
	cfg.ppo.optRefineMaxDelta = 0.0f;
	cfg.ppo.optRefineTrustPenalty = 0.0f;
	cfg.ppo.useSORS = false;
	cfg.ppo.sorsRewardScale = 0.0f;
	cfg.ppo.sorsRewardScaleAnnealStart = -1;
	cfg.ppo.sorsRewardScaleAnnealSteps = 0;
	cfg.ppo.sorsRewardClipRange = 0.0f;
	cfg.ppo.sorsWarmupIters = 0;
	cfg.ppo.sorsTrainPairs = 0;
	cfg.ppo.sorsMaxReplayWindows = 0;
	cfg.ppo.sorsMinLabelDelta = 0.0f;
	cfg.ppo.sorsWindowBefore = 0;
	cfg.ppo.sorsWindowAfter = 0;
	cfg.ppo.sorsLabels = {};

	cfg.ppo.sharedHead.layerSizes = {};
	cfg.ppo.policy.layerSizes = { 512, 512, 256 };
	cfg.ppo.critic.layerSizes = { 768, 512, 256 };
	cfg.ppo.gcrlCritic.layerSizes = { 768, 512, 256 };
	cfg.ppo.sorsReward.layerSizes = {};

	cfg.skillTracker.enabled = true;
    cfg.skillTracker.numArenas = 24;        // eval arenas, keep near CPU thread count
    cfg.skillTracker.simTime = 45;          // seconds simulated per eval run
    cfg.skillTracker.maxSimTime = 240;      // continuation cap if too few goals happen
    cfg.skillTracker.updateInterval = 16;   // run Elo every N training iterations
    cfg.skillTracker.ratingInc = 5;         // Elo K-ish scale per goal
    cfg.skillTracker.initialRating = 0;
    cfg.skillTracker.deterministic = true;  // optional: rate greedy policy instead of sampled policy

	cfg.evolutionStrategy.enabled = false;

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
	auto learner = std::make_unique<Learner>(EnvCreateFunc, cfg, StepCallback);

	// Start learning!
	learner->Start();

	return EXIT_SUCCESS;
}
