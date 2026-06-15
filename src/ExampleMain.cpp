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
#include <RLGymCPP/StateSetters/FrontierState.h>
#include <RLGymCPP/ActionParsers/DefaultAction.h>

#include <atomic>
#include <cstdlib>
#include <cmath>
#include <memory>

using namespace GGL; // GigaLearn
using namespace RLGC; // RLGymCPP

// ── Self-tuning curriculum shared state ──
// Created in main() before the Learner (EnvCreateFunc runs inside the Learner ctor),
// raw pointers/globals like `resetInfo`. All inert unless the matching flag is set.
static RLGC::FrontierStateBuffer* g_FrontierBuffer = nullptr; // Feature A buffer (also fed to PPO config)
static float g_FrontierResetFraction = 0;
static int g_FrontierBufferMinFill = 256;
// Feature C: worker-read adaptive StrongTouch floor, seeded at the reward's hand value
// (20 kph). The learner publishes the ratcheted value here once per iteration.
static std::atomic<float> g_StrongTouchMinVel{ RLGC::Math::KPHToVel(20) };
static bool g_UseAdaptiveStrongTouchFloor = false;

struct EnvResetInfo {
	bool isKickoffReset = true;
};

class ResetModeStateSetter : public StateSetter {
public:
	StateSetter* child;
	EnvResetInfo* resetInfo;
	bool isKickoffReset;
	// Optional frontier reset-source tagging (Feature A episode-return metric); NULL = off.
	FrontierStateBuffer* frontierBuffer;
	uint8_t frontierTag;

	ResetModeStateSetter(StateSetter* child, EnvResetInfo* resetInfo, bool isKickoffReset,
		FrontierStateBuffer* frontierBuffer = nullptr, uint8_t frontierTag = FrontierStateBuffer::TAG_NONE) :
		child(child), resetInfo(resetInfo), isKickoffReset(isKickoffReset),
		frontierBuffer(frontierBuffer), frontierTag(frontierTag) {
	}

	virtual ~ResetModeStateSetter() override {
		delete child;
	}

	virtual void ResetArena(Arena* arena) override {
		resetInfo->isKickoffReset = isKickoffReset;
		if (frontierBuffer && frontierTag != FrontierStateBuffer::TAG_NONE)
			frontierBuffer->TagArenaReset(arena, frontierTag);
		child->ResetArena(arena);
	}
};

// ── Aerial curriculum state-setter ──
// Spawns the ball up in the air and one car positioned to go up and TOUCH it, so the policy
// actually experiences aerial touches (the existing aerial rewards then reinforce them) instead
// of having to stumble into one in open play -- the deep-basin reachability problem behind the
// ~0 high-air-touch ratio. The other car is parked downfield so it doesn't contest the rep.
// difficulty in [0,1]: 0 = car already airborne just under the ball, moving at it, pointed at it,
// full boost (a near-gimme touch); 1 = car on the ground, must jump + boost + air-control the
// whole way. Start easy so reps succeed; ramp up as high-air-touch competence grows (a fixed
// value here; an adaptive hook can publish a rising difficulty later, like the StrongTouch floor).
class AerialState : public StateSetter {
public:
	float difficulty;
	AerialState(float difficulty) : difficulty(RS_CLAMP(difficulty, 0.0f, 1.0f)) {}

	virtual void ResetArena(Arena* arena) override {
		using RocketSim::Math::RandFloat;
		using RocketSim::Math::RandInt;
		arena->ResetToRandomKickoff();

		float d = difficulty;

		// Ball: high, roughly central, gently moving -- a catchable aerial target.
		float ballZ = RandFloat(700, 1300);
		Vec ballPos = Vec(RandFloat(-2200, 2200), RandFloat(-2200, 2200), ballZ);
		BallState bs = {};
		bs.pos = ballPos;
		bs.vel = Vec(RandFloat(-250, 250), RandFloat(-250, 250), RandFloat(-100, 150));
		arena->ball->SetState(bs);

		int nCars = (int)arena->_cars.size();
		int aerialist = RandInt(0, nCars);
		int idx = 0;
		for (Car* car : arena->_cars) {
			CarState cs = {};
			if (idx == aerialist) {
				// Start height: just under the ball (easy) down to the floor (hard).
				float topZ = ballZ - 250.0f;
				float carZ = RS_MAX(17.0f, topZ - d * (topZ - 17.0f));
				// Horizontal: under / short of the ball, farther back as it gets harder.
				float back = 250.0f + 500.0f * d;
				cs.pos = Vec(ballPos.x + RandFloat(-250, 250), ballPos.y - back, carZ);
				Vec toBall = (ballPos - cs.pos).Normalized();
				cs.vel = toBall * (RandFloat(300, 700) * (1.0f - 0.7f * d)); // toward the ball
				float yaw = std::atan2(toBall.y, toBall.x);
				float pitch = std::asin(RS_CLAMP(toBall.z, -1.0f, 1.0f)) * (1.0f - 0.6f * d);
				if (carZ <= 18.0f) { cs.pos.z = 17.0f; cs.vel.z = 0; pitch = 0; } // grounded at high difficulty
				cs.rotMat = Angle(yaw, pitch, 0).ToRotMat();
				cs.boost = RandFloat(70, 100); // aerials need boost
			} else {
				// Park the other car downfield on the ground, out of the rep.
				float side = (RandInt(0, 2) == 0) ? -1.0f : 1.0f;
				cs.pos = Vec(RandFloat(-1500, 1500), side * 4200.0f, 17.0f);
				cs.rotMat = Angle(side < 0 ? (float)(-M_PI / 2) : (float)(M_PI / 2), 0, 0).ToRotMat();
				cs.boost = RandFloat(0, 60);
			}
			car->SetState(cs);
			idx++;
		}
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
		case 2:  chaseWeight = 4.0f; break;
		default: chaseWeight = 3.0f; break;
	}

	std::vector<WeightedReward> rewards = {

		// Ball-goal. 15 (was 5): the negative side of this zero-sum term is the only dense
		// defensive gradient in the stack -- at 5 it was drowned out by the ~90-weight
		// offensive rewards and the bot never paid for letting the ball travel at its net.
		{ new ZeroSumReward(new VelocityBallToGoalReward(), 1), 15.0f },

		// Ungated aerial bootstrap; completed aerial touches/followups remain GCRL-gated.
		{ new AerialCommitReward(), 6.0f },

		// Boost
		{ new PickupBoostReward(), 10.f },
		// 0.05: at 0.2 the max passive income (0.2/step at full boost) beat the time penalty
		// (0.075/step) and run bgksd0wi converged to boost-camping as its best strategy.
		{ new SaveBoostReward(), 0.05f },

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

		// Ground/contact rewards. These live in the legacy gcrlGatedRewards bucket,
		// but with useGCRLRewardGate=false below they pass through ungated.
		{ new ZeroSumReward(new StrongTouchReward(20, 100,
			g_UseAdaptiveStrongTouchFloor ? &g_StrongTouchMinVel : nullptr), TEAM_SPIRIT, 0.0f), 90.f },

		// // Small energy reward: encourages speed, boost, flip availability, and forward velocity.
		// // Keep this off for now; it is easy to farm if added ungated.
		// { new EnergyReward(), 1.0f }
	};

	std::vector<WeightedReward> curriculumRewards = {

		// Temporary chase incentive.
		{ new VelocityPlayerToBallReward(), chaseWeight },

		// Bottom rung of the touch ladder: StrongTouchReward pays zero below 20kph hit
		// force, so the first weeks of grazing touches earn nothing from it. This pays for
		// ANY contact and anneals away with the rest of the curriculum.
		{ new TouchBallReward(), 2.0f }
	};

	std::vector<WeightedReward> aerialGCRLGatedRewards = {

		// Aerial rewards. These live in the legacy aerialGCRLGatedRewards bucket,
		// but with useGCRLRewardGate=false below they pass through ungated.
		// NOTE: No unconditional AirReward here -- per-step "be airborne" income taught the bots
		// to hover (80%+ air time, ground-level touch heights). The rewards below all require
		// the ball to actually be up and/or productive contact.
		// 3.0 (was 1.5): going up for a high ball has to compete with the ground chase reward
		// (weight 6, pays on every ball). This only pays airborne with ball z >= 500, so it
		// can't be hover-farmed like AirReward was.
		{ new HeightWeightedAerialApproachReward(), 3.0f },
		{ new UsefulAirTouchReward(), 25.0f },
		{ new SecondTouchReward(), 12.0f },
		{ new FlipResetFollowupReward(), 8.0f },
		{ new UsefulFlickReward(), 8.0f }
	};

	std::vector<WeightedReward> aerialCurriculumRewards = {

		// Bootstrap rung for real aerial touches: pays only on airborne high-ball contact,
		// scaled by both time spent airborne and ball height. Unlike AirReward, there is no
		// per-step hover income; unlike UsefulAirTouchReward, it does not require goalward quality.
		{ new AirTimeTouchReward(), 10.0f },

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
	// PHASE 2: 20% random resets back in. Touch competence is established (touch ratio
	// ~0.010 in ryp4gxwv), and random states are the data source for defensive and
	// high-ball positions -- with 100% kickoffs the bot never spawned behind the ball
	// facing a threat, so saves/defense had no training data.
	// Feature A: when the frontier buffer exists, route frontierResetFraction of resets to
	// FrontierState (sampling learner-harvested critic-uncertainty states; RandomState
	// fallback below min fill), splitting the remainder across the existing kickoff/random
	// mix. When it's null, build exactly today's two-entry tree (bit-identical off-path).
	if (g_FrontierBuffer) {
		float f = g_FrontierResetFraction;
		result.stateSetter = new CombinedState({
			{ new ResetModeStateSetter(new KickoffState(), resetInfo, true,
				g_FrontierBuffer, FrontierStateBuffer::TAG_KICKOFF), 0.80f * (1 - f) },
			{ new ResetModeStateSetter(new RandomState(true, true, false), resetInfo, false,
				g_FrontierBuffer, FrontierStateBuffer::TAG_RANDOM), 0.20f * (1 - f) },
			{ new ResetModeStateSetter(
				new FrontierState(g_FrontierBuffer, new RandomState(true, true, false), g_FrontierBufferMinFill),
				resetInfo, false), f }
		});
	} else {
		// Aerial curriculum: 15% of resets seed an aerial situation so the bot actually gets
		// aerial-touch reps (high-air-touch ratio has been ~0 -- the basin exists but PPO never
		// reaches it). difficulty 0.35 = mostly-airborne, achievable touches to start; raise it
		// as the high-air-touch competence EMA climbs. If this alone cracks aerials, no ERL needed.
		result.stateSetter = new CombinedState({
			{ new ResetModeStateSetter(new KickoffState(), resetInfo, true), 0.65f },
			{ new ResetModeStateSetter(new RandomState(true, true, false), resetInfo, false), 0.20f },
			{ new ResetModeStateSetter(new AerialState(0.35f), resetInfo, false), 0.15f }
		});
	}
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

	// Feature D offline harness: `--score-opt <file>` loads the checkpoint, scores the
	// hand-supplied states in <file> through the frozen optionality scorer, prints
	// phi_opt per row, and exits. Requires useOptionality or useOptionEntropy (set below).
	std::string scoreOptPath;
	for (int i = 1; i + 1 < argc; i++)
		if (std::string(argv[i]) == "--score-opt")
			scoreOptPath = argv[i + 1];

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
	cfg.trainAgainstOldPreservedDrainFraction = 0.5f;
	cfg.trainAgainstOldMaxPreservedBatches = 4;
	cfg.tsPerVersion = 25'000'000;
	cfg.maxOldVersions = 32;

	int tsPerItr = 150'000;
	cfg.ppo.tsPerItr = tsPerItr;
	cfg.ppo.batchSize = tsPerItr;
	cfg.ppo.miniBatchSize = 75'000; // Lower this if too much VRAM is being allocated
	cfg.ppo.overbatching = true;
	// 2 optimizer steps per epoch (one per minibatch) instead of one accumulated step.
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
	// Back to 0.65/0.10 (the ryp4gxwv values). The 0.70/0.20 bump was premised on healthy
	// reward income from a competent checkpoint; from scratch, e79pvv92's entropy fell to
	// 0.48 ANYWAY with the controller pinned at the 0.20 ceiling -- the bonus term ran ~10x
	// the policy loss (Policy Relative Entropy Loss ~9.8) without buying any exploration,
	// it just diluted the reward gradient during the income-starved bootstrap (the rwkkyfej
	// failure mode). Re-raise only when resuming a checkpoint with established income.
	cfg.ppo.targetEntropy = 0.65f;
	cfg.ppo.adaptiveEntropyLR = 5e-3f;
	cfg.ppo.minEntropyScale = 0.0f;
	cfg.ppo.maxEntropyScale = 0.10f;

	// Rate of reward decay
	// PHASE 2: back to 0.995. The 0.99 phase-1 value concentrated credit on approach->touch,
	// which worked (touch ratio ~0.010) -- but its ~100-step horizon is shorter than the
	// 250-450 step episodes, so the -275 concede penalty never reached the positioning
	// mistakes that caused it. Goals exist now; propagate their credit.
	cfg.ppo.gaeGamma = 0.995;

	// With stepPerMiniBatch (4 optimizer steps/iter) 4e-4 moved the policy 4-5x too fast
	// (KL 0.01-0.03, clip fraction 0.15, entropy collapse in run bgksd0wi). 1.5e-4 targets
	// the healthy ~0.03/iter update magnitude; maxMeanKL below is the hard backstop.
	cfg.ppo.policyLR = 1.5e-4;
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
	// Intentional 0.65 scale: the stronger GCRL channel has been more useful than the
	// conservative 0.3 setting in this curriculum. Keep watching Final-vs-Reward advantage
	// balance; if GCRL/Final Advantage persistently dwarfs GCRL/Reward Advantage, the critics
	// are again steering most of the policy gradient.
	cfg.ppo.gcrlAdvScale = 0.5f;
	cfg.ppo.gcrlAdvScaleAnnealStart = 400'000'000;
	cfg.ppo.gcrlAdvScaleAnnealSteps = 100'000'000;
	// The anti critic now scores own-goal danger (it queries the own-goal target instead of
	// duplicating the goal critic), so this weighs real defensive signal, not twin-network
	// noise. Match goal pursuit strength so last-man defense can veto rebound waiting.
	cfg.ppo.gcrlAntiScale = 1.0f;
	cfg.ppo.gcrlCarScale = 0.5f;     // car-positioning critic weight in the GCRL advantage
	// ── Gradient surgery: OFF ──
	// The magnitude-gated diagnostics settled it: HiMag Conflict ~0.45 (<0.5, i.e. reward and
	// game-sense are if anything mildly aligned, not opposed) -> no systematic misalignment to
	// fix, and at strength 1.0 it was cutting ~10% of reward signal on chance disagreements (a
	// drag). Left wired (gcrlSurgery + magThreshold) for a future run if mechanics rewards
	// create real shaped-vs-true tension.
	cfg.ppo.gcrlSurgery = false;
	cfg.ppo.gcrlBaselineSamples = 2; // counterfactual action baseline (K shuffled-action forwards)
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
	// Gate knobs are retained for future re-enable, but inactive while useGCRLRewardGate=false.
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
	cfg.ppo.curriculumRewardScale = 1.0f;
	cfg.ppo.curriculumRewardAnnealStart = -1;
	cfg.ppo.curriculumRewardAnnealSteps = 2'500'000'000;
	cfg.ppo.aerialCurriculumRewardScale = 1.0f;
	cfg.ppo.aerialCurriculumRewardAnnealStart = -1;
	cfg.ppo.aerialCurriculumRewardAnnealSteps = 2'500'000'000;
	// Competence gates: anneal progress only accumulates while the EMA ratio is at/above
	// the gate, so the curricula literally cannot expire while the bot still needs them
	// (rwkkyfej spent 2.3B steps at touch ratio ~0.001 watching its chase reward anneal
	// away; lengthening the window keeps losing that bet, the gate ends it). Reference
	// points: blended-competent touch ratio across this mode mix is ~0.007 (good 1v1 run
	// sat at 0.009-0.0105); 0.0002 high-air touches/step ≈ a few percent of touches being
	// real aerials. Watch "Curriculum/Touch Ratio EMA" + "Curriculum/High Air Touch Ratio
	// EMA" against these gates in wandb.
	cfg.ppo.curriculumAnnealTouchRatioGate = 0.006f;
	cfg.ppo.aerialCurriculumAnnealAirTouchRatioGate = 0.0002f;

	// ── Self-tuning curriculum ──
	// A: uncertainty-triggered frontier resets. B: difficulty/coverage-aware HER goal sampling.
	// C: adaptive ratcheted-quantile gate target / StrongTouch floor. D: optionality
	// potential shaping. See PPOLearnerConfig.h for the per-feature knobs and rationale.
	cfg.ppo.useFrontierResets = true;          // Feature A
	cfg.ppo.useDifficultyHER = true;           // Feature B
	cfg.ppo.herCandidates = 8;                 // Wider candidate set so coverage-HER has room to select rare/useful goals
	cfg.ppo.useCoverageHER = true;             // Feature B.2: dynamic rare/useful goal coverage for HER + reset harvesting
	cfg.ppo.herCoverageBankSize = 8192;
	cfg.ppo.herCoverageCompareSamples = 64;
	cfg.ppo.herCoverageBankInsertCap = 2048;
	cfg.ppo.herCoverageNoveltyStrength = 1.0f;
	cfg.ppo.herCoverageUtilityStrength = 0.75f;
	cfg.ppo.herCoverageMaxBoost = 3.0f;
	cfg.ppo.herCoverageResetMaxInsertsPerIter = 512;
	cfg.ppo.useAdaptiveGateTargetVel = true;   // Feature C.1
	cfg.ppo.useAdaptiveStrongTouchFloor = true;// Feature C.2
	cfg.ppo.useOptionality = false;             // Feature D
	cfg.ppo.useOptionEntropy = true;            // use optionality breadth to suppress/allow entropy, without reward shaping
	cfg.ppo.optionEntropyCommitPenalty = 0.01f;
	cfg.ppo.optionEntropyMinQualityZ = -0.5f;
	cfg.ppo.optionEntropyQualitySharpness = 0.5f;
	cfg.ppo.optionEntropyBreadthTemp = 1.0f;
	cfg.ppo.optionEntropyKickoffSteps = 90;
	cfg.ppo.optionEntropyKickoffWeight = 0.0f;
	cfg.ppo.optionEntropyOpenNetWeight = 0.10f;
	cfg.ppo.optionEntropyOwnNetWeight = 0.10f;
	cfg.ppo.optBankSize = 1024;                  // Option-entropy bank; cheaper than the 2048 default while retaining breadth.
	cfg.ppo.optCommitReliefScale = 0.75f;      // Do not punish option loss as hard when GCRL terminal prospects improve
	cfg.ppo.optCommitReliefSharpness = 1.0f;
	cfg.ppo.optDeficitFloorStd = 0.75f;        // Only punish unusually low optionality; no reward above the floor
	cfg.ppo.optDeficitClip = 3.0f;
	cfg.ppo.optValueWeight = 1.0f;             // Value-weighted optionality: reachable AND terminal-useful futures
	cfg.ppo.optValueClip = 3.0f;
	cfg.ppo.optRefineGoals = false;             // Refine the collapse detector without rewarding extra options above the floor
	cfg.ppo.optRefineTopK = 4;
	cfg.ppo.optRefineSteps = 3;
	cfg.ppo.optRefineMaxStates = 4096;
	cfg.ppo.optRefineStepSize = 0.08f;
	cfg.ppo.optRefineMaxDelta = 0.25f;
	cfg.ppo.optRefineTrustPenalty = 0.1f;

	std::unique_ptr<RLGC::FrontierStateBuffer> frontierBufferOwner;
	if (cfg.ppo.useFrontierResets) {
		frontierBufferOwner = std::make_unique<RLGC::FrontierStateBuffer>(cfg.ppo.frontierBufferSize);
		g_FrontierBuffer = frontierBufferOwner.get();
		cfg.ppo.frontierBuffer = g_FrontierBuffer;
		g_FrontierResetFraction = cfg.ppo.frontierResetFraction;
		g_FrontierBufferMinFill = cfg.ppo.frontierBufferMinFill;
	}
	if (cfg.ppo.useAdaptiveStrongTouchFloor) {
		g_UseAdaptiveStrongTouchFloor = true;
		cfg.ppo.adaptiveStrongTouchFloorAtomic = &g_StrongTouchMinVel;
	}

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
	cfg.evolutionStrategy.updateInterval = 25;    // run ES/ERL every N training iters; main dial for wall-time cost (watch ES/StepTime or ERL/Step Time)
	cfg.evolutionStrategy.antithetic = true;       // evaluate +/- pairs (variance reduction)
	cfg.evolutionStrategy.rankNormalize = true;    // centered-rank fitness shaping
	// cfg.evolutionStrategy.scope = EvolutionStrategyConfig::Scope::POLICY_ONLY; // default; only perturb the policy head

	// ── Update rule: SELECTION instead of the ES gradient ──
	// Same 8192-member low-rank eval on goal-differential (the true objective), but the update is
	// CEM_BEST -- re-anchor the live policy onto the single best member's perturbation, a decisive
	// jump PPO doesn't dilute. With a larger sigma this reaches behaviors PPO is stuck short of,
	// while optimizing true scoring rather than the shaped proxy. (RANK_GRADIENT = old ES gradient;
	// CEM_ELITE = top-cemElites mean.) LEFT OFF: run the aerial curriculum first; flip
	// `enabled = true` if you want selection-ES alongside it. Watch "ES/MeanGoalDiff" + "ES/UpdateNorm".
	cfg.evolutionStrategy.updateMode = EvolutionStrategyConfig::UpdateMode::CEM_BEST;
	cfg.evolutionStrategy.cemElites = 256;           // only used by CEM_ELITE
	cfg.evolutionStrategy.sigma = 0.04f;             // larger than the gradient's 0.02 -- selection tolerates it for reach
	// enabled stays false above -- flip it to true to run selection-ES.


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

	if (!scoreOptPath.empty()) {
		learner->DebugScoreOptionality(scoreOptPath);
		return EXIT_SUCCESS;
	}

	// Start learning!
	learner->Start();

	return EXIT_SUCCESS;
}
