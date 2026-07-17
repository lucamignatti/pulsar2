#pragma once
#include <RLGymCPP/BasicTypes/Lists.h>

#include "../Util/ModelConfig.h"

namespace RLGC {
	class DrillBank;
}

namespace GGL {

	// Learned goal-reachability (InfoNCE + HER), used ONLY as:
	//  (a) an auxiliary representation loss on the shared head, and
	//  (b) the input of a multiplicative reward gate on the WeightedReward::gated components.
	// It never enters advantages, returns, or the critic (the coupling that froze prior designs).
	// Two heads share one phi(trunk(obs), action) encoder:
	//  CAR  head: "can this car reach the ball" (HER on the car-local ball)
	//  BALL head: "can the ball reach the opponent net" (HER on the canonical ball state)
	struct ReachabilityConfig {
		bool enabled = false;     // Train the heads (aux loss) + compute gate diagnostics
		bool gateEnabled = false; // Actually scale the gated rewards (requires enabled)

		// When the gate is OFF, the rho/gate read block (3 full-buffer model passes + CPU smoothing)
		// feeds nothing but the Reach/* diagnostic panels — so only run it every Nth iteration.
		// Ignored (every iteration) when gateEnabled=true, since rewards then depend on it.
		int diagEveryIters = 1;

		// Model
		int representationSize = 128;
		float tau = 0.02f;          // Contrastive temperature; Score = cosine(phi, psi) / tau
		float lr = 3e-4f;
		float auxLossWeight = 0.5f; // Scale of the InfoNCE losses added to the PPO backward
		int infoSubSample = 512;    // Rows per PPO minibatch used for InfoNCE (the NxN logits matrix)
		float varReg = 0.3f;
		float logsumexpPenaltyCoeff = 0.01f;
		PartialModelConfig phi, psi;

		// HER goal sampling
		int ballHerMinOffset = 1;
		int ballHerMaxOffset = 45;  // ~3s at tickSkip 8 (15Hz steps) - a REAL-TIME window; re-derive if tickSkip changes (was 90 at ts4)
		float ballHerShortBiasPower = 2;
		// Chance of targeting the most-goalward achieved state in the window instead of a
		// short-biased random one, so near-net states populate the goal space and the fixed
		// scoring query goal stays in-distribution
		float ballHerGoalwardBias = 0.5f;
		int carHerMinOffset = 1;
		int carHerMaxOffset = 10;   // ~0.67s at tickSkip 8 - REAL-TIME window feeding the steering rho-band contact
		                            // gate ("commit where the race is a coin-flip"); re-derive if tickSkip changes (was 20 at ts4)
		float carHerShortBiasPower = 2;
		// Third goal-space head: canonical CAR pos+vel ("where can my car be, moving
		// how") - the movement-capability frontier for the META steering system. The
		// head + HER machinery predate this (car-proposer era); this flag trains it
		// INDEPENDENT of the proposer. Offline (conservative frozen-phi test,
		// analysis/probes/carstate_head_validate.py): its calibration curve is monotone
		// with ~7x the ball head's margin at every candidate window - the sharpest
		// frontier detector of the three heads. The window below was chosen BY
		// calibration margin across {20,45,90}, not by hand - AT tickSkip 4; that
		// calibration is VOID at ts8. Deliberately HELD at 45 (2026-07-17): it is an
		// empirical choice, not a real-time design like the two windows above, so it
		// waits for a carstate_head_validate.py re-run on ts8 data instead of blind
		// halving. Actuation stays gated
		// live (meta head-validity + per-cluster causal gates). Resume-safe: a missing
		// reach_psi_carstate.lt initializes fresh (ModelSet::Load allowNotExist).
		bool carStateHead = false;
		int carStateHerMinOffset = 1;
		int carStateHerMaxOffset = 45; // ts4-calibrated (void at ts8); re-calibration pending, see comment above
		// Gradient coupling of the car-state InfoNCE into the shared state-action encoder
		// (phi -> trunk). 0 = fully detached: only psi_carstate trains, exactly the
		// offline-validated frozen-phi regime. 2026-07-14 incident: this head shipped
		// fully coupled (the equivalent of 1.0) while at chance level - its loss alone
		// (~2x every other aux term combined, Reach/Aux Loss 0.5 -> 1.0+) churned the
		// shared trunk and Rating slid ~125 across ALL modes in ~500 iterations with
		// every behavioral guard green (this path had none). Values in (0,1] gradient-
		// scale the coupling (value-preserving: loss magnitude unchanged, trunk/phi
		// gradient scaled). Raising it is a one-lever experiment with the rating
		// latches watching - never ship it coupled while the head is fresh.
		float carStateCouple = 0.0f;
		// The ball head only trains on episodes where the ball exceeded this speed;
		// a dead never-touched episode would just reteach the stationary-ball manifold
		float minBallMoveSpeed = 300;

		// Goal normalization (canonical frame: +y is always the attacked net)
		float posScaleX = 4096, posScaleY = 6000, posScaleZ = 2044, velScale = 6000;
		float carLocalScale = 2300;    // Car-local ball goals are normalized uniformly by this
		float scoringGoalSpeed = 1500; // Canonical +y ball speed of the fixed scoring query goal

		// rho evaluation: rho(s -> goal) = mean over K UNIFORM VALID actions of Score.
		// Uniform (not policy-sampled) = capability ("can we", not "would we")
		int numActionSamples = 16;
		int64_t scoreChunkSize = 4096;

		// Gate: level x delta, floored, positive-part-applied:
		//  level = control^alpha * scoring^(1-alpha)
		//  D     = sigmoid(wc*deltaControl + ws*deltaScoring)   (windowed progress deltas)
		//  gRaw  = clamp(level * 2 * D, 0, 1)                   (delta=0 => exactly the level gate)
		//  gEff  = floor + (1 - floor) * gRaw
		//  mult  = 1 - beta * (1 - gEff)                        (beta=0 => bit-identical baseline)
		//  reward = total + (mult - 1) * gatedPos               (positive parts only; penalties never muted)
		float gateFloor = 0.2f;
		float gateAlpha = 0.5f;          // Control weight in the level mix
		float controlTemp = 10;          // T_c on (rhoCarUs - rhoCarOpp); rho are cosine/tau logits
		float scoringTemp = 10;          // T_s
		float scoringBias = 0;           // b_s
		float deltaControlWeight = 0.5f; // wc
		float deltaScoringWeight = 0.5f; // ws
		int deltaWindow = 8;             // W steps for the windowed deltas (~0.53s at tickSkip 8)
		int deltaSmooth = 4;             // Boxcar length for the smoothed rho reads
		int touchPredHorizon = 45;       // Steps ahead a next-touch label may be found (validity metric)

		// Anneal: beta = min(smoothstep(accEMA over [accLo,accHi]), smoothstep(agreeEMA over [aucLo,aucHi]))
		// accEMA = InfoNCE categorical accuracy (min of the two heads);
		// agreeEMA = does sign(deltaControl) predict which team touches next (measured on real touches)
		float accLo = 0.30f, accHi = 0.55f;
		float aucLo = 0.55f, aucHi = 0.65f;
		float emaDecay = 0.99f; // Per-iteration decay of both EMAs
		float betaOverride = -1; // >= 0 forces beta (smoke tests); < 0 uses the EMAs

		ReachabilityConfig() {
			// Optimizer stays ADAM for both heads: contrastive InfoNCE embeddings train
			// poorly under orthogonalized updates (phi/psi = Adam, not Muon)
			phi = {};
			phi.layerSizes = { 256, 256 };
			phi.activationType = ModelActivationType::LEAKY_RELU;
			psi = {};
			psi.layerSizes = { 256, 256 };
			psi.activationType = ModelActivationType::LEAKY_RELU;
		}
	};

	// Deliberate-practice goal proposer: g_t = clamp(g_{t-1} + Delta(detached_trunk_feat_t, g_{t-1})),
	// goals living in the SAME 6D canonical-ball space as the reachability BALL head. Trained by
	// advantage-weighted hindsight (targets = achieved ball state N steps ahead; CRR-binary weights
	// from an N-step-advantage aspiration percentile). Requires reachability.enabled (reuses phi/psiBall).
	//
	// Stage 1 (enabled by default) is PASSIVE: it trains the proposer and logs calibration metrics,
	// but never touches rewards/advantages/returns/values.
	// Stage 2 (shapingBeta, default 0) adds a centered, advantage-only shaping term built from the
	// SAME proposed goal on both sides of the potential difference (gamma*rho(s',g) - rho(s,g)) so
	// goal-motion never gets charged to the policy — only its progress toward a goal that stood.
	// Stage 3 (practiceEnabled, default off) detects reachability "drop" events (a committed mistake),
	// banks a restorable snapshot + the goal that was in play, and periodically replays that snapshot
	// under an amplified beta so the policy gets repeated at-bats on its own near-misses.
	struct ProposerConfig {
		// ---- Stage 1 (ships ENABLED; passive) ----
		bool enabled = true;              // requires reachability.enabled (validated in PPOLearner ctor)
		PartialModelConfig delta;         // proposer_delta net; Muon (like policy/critic, not Adam like reach)
		float lr = 1e-4f;
		int horizonSteps = 45;            // N ~ 3s at tickSkip 8
		float goalClamp = 1.5f;           // box the unrolled goal is kept within, normalized units
		float aspirationPercentile = 0.75f;  // top (1-p) of A^(N) get weight 1
		float belowAspirationWeight = 0.05f; // CRR-binary low arm; NEVER 0 (dilution, not repulsion)
		int trainMinibatchSize = 4096;
		int trainEpochs = 1;
		int64_t featureChunkSize = 4096;  // shared-head feature pass chunk size (mirrors reach.scoreChunkSize)
		int dumpEveryNItrs = 25;          // JSONL calibration dump cadence to disk; 0 = never
		int dumpMaxRows = 512;

		// ---- Stage 2 (code-complete; DISABLED by default) ----
		float shapingBeta = 0.0f;         // 0 => stage-2 path fully skipped (bit-identical to stage-1-only)

		// ---- Car proposer head (canonical CAR-state goals, not ball-relative; DISABLED by default) ----
		// A second proposer head proposing where the CAR should go, delivered as its own potential
		// shaping term. Reuses the SAME A^(N) aspiration weights + trunk features as the ball head
		// (aspiration is goal-space-agnostic); adds a psi_carstate reach head + a second delta net.
		// Passive when carEnabled && carShapingBeta==0 (trains + logs car-space tilt, no shaping).
		bool carEnabled = false;          // build + train the car head (requires enabled)
		float carShapingBeta = 0.0f;      // 0 => car shaping skipped (car head stays passive)

		// ---- Stage 3 (code-complete; DISABLED by default) ----
		bool practiceEnabled = false;
		RLGC::DrillBank* drillBank = NULL; // owned by user code (e.g. ExampleMain); required when practiceEnabled
		int snapshotEveryK = 8;           // per-arena snapshot cadence, in collection steps
		float phiSquashTemp = 10;         // Phi = sigmoid(rho / T) for drop detection
		// Phi-drop detector thresholds. Phi is a squashed InfoNCE logit - uncalibrated and NON-
		// stationary (the rho distribution drifts as the reach critic + policy improve), so absolute
		// thresholds silently flood or starve. Default = self-calibrate each iteration from the
		// batch's own Phi distribution over non-practice rows: "high" = the phiHighPercentile-th Phi,
		// "drop" = that minus the phiDropPercentile-th Phi. Scale-free, tracks the drift automatically
		// (same dimensionless-dial pattern as aspirationPercentile). Set phiCalibratePerIter=false to
		// fall back to the fixed phiHighThresh/phiDropThresh below.
		bool phiCalibratePerIter = true;
		float phiHighPercentile = 0.85f;  // batch percentile that counts as a "high" (reachable) moment
		float phiDropPercentile = 0.40f;  // low anchor; drop-magnitude bar = Phi(pHigh) - Phi(pDrop)
		float phiHighThresh = 0.7f;       // absolute fallback: Phi was "high" if it reached at least this
		float phiDropThresh = 0.3f;       // absolute fallback: a drop of at least this = a "mistake"
		int phiDropWindow = 30;           // steps
		int maxNewDrillsPerItr = 16;      // bank at most this many per iter, the LARGEST-drop candidates
		int drillDumpEveryNItrs = 25;     // JSONL dump of the bank's contents for eyeballing; 0 = never
		int drillDumpMaxRows = 128;
		int practiceWindowSteps = 90;     // tagged window length after a drill reset (~6s at tickSkip 8)
		float practiceBetaScale = 3.0f;   // shaping amplification on practice-tagged rows
		float drillJitterPos = 100, drillJitterVel = 150;
		float successProximity = 0.15f;   // normalized canonical ball-pos distance counted as a "recovery"
		int drillMaxTries = 20;
		int drillMinTriesForRetire = 5;
		float drillRetireSuccessRate = 0.7f;
		int maxDrillBankSize = 512;

		ProposerConfig() {
			delta = {};
			delta.layerSizes = { 256, 256 };
			delta.activationType = ModelActivationType::LEAKY_RELU;
			delta.optimType = ModelOptimType::MUON;
		}
	};

	// Secondary GOAL-ONLY critic (multi-horizon value decomposition). The dense shaped reward needs a
	// moderate gamma or target variance swamps the critic; the true objective (goal/concede, the one
	// unfarmable signal) carries meaning at much longer horizons. This trains a SECOND critic on a
	// goal-only reward channel (+1 team scored / -1 conceded, computed straight from game outcomes —
	// independent of the reward stack) at its own, much higher gamma, and blends its advantages into
	// the policy gradient: A = A_dense + betaEff * A_goal, betaEff std-matched so beta is the FRACTION
	// of dense-advantage scale contributed (a bounded influence by construction).
	// Episodes terminate on goals, so V_goal is inherently bounded in [-1,1]: it converges toward
	// P(score) - P(concede) with mild time preference — no standardization or clipping needed.
	// The net is fully independent (raw obs in, own trunk): zero gradient interference with the
	// proven policy/critic/shared-head path. Validation: watch GoalCritic/Value-Outcome Corr — the
	// critic's prediction must correlate POSITIVELY with realized episode outcomes; a channel or
	// sign bug reads negative there within minutes on a goal-dense checkpoint.
	struct GoalCriticConfig {
		bool enabled = false;
		float gamma = 0.9997f;      // STEP-denominated "huge distance" horizon: ~77s at tickSkip 4 (30Hz) but ~154s at
		                            // tickSkip 8 (15Hz) — re-derive per tickSkip in your main (5.0 overrides to 0.9994 ≈ 77s at 15Hz)
		float beta = 0.25f;         // blended advantage fraction (std-matched); 0 = train critic, no blend
		float lr = 1.5e-4f;
		PartialModelConfig model;   // independent net, raw obs -> 1; set layerSizes in your main
	};

	// https://github.com/AechPro/rlgym-ppo/blob/main/rlgym_ppo/ppo/ppo_learner.py
	struct PPOLearnerConfig {

		int64_t tsPerItr = 50'000;
		int64_t batchSize = 50'000;
		int64_t miniBatchSize = 0; // Set to 0 to just use batchSize

		// On the last batch of the iteration, 
		//	if the amount of remaining experience exceeds the batch size, 
		//	all remaining experience is used as a larger batch.
		// This prevents experience loss due to batch size rounding.
		// This will only happen if the amount of remaining experience is < batchSize*2.
		bool overbatching = true;

		double maxEpisodeDuration = 120; // In seconds

		// Actions with the highest probability are always chosen, instead of being more likely
		// This will make your bot play better (usually), but is horrible for learning
		// Trying to run a PPO learn iteration with deterministic mode will throw an exception
		bool deterministic = false;

		// Use half-precision models for inference
		// This is much faster on GPU, not so much for CPU
		bool useHalfPrecision = false;

		PartialModelConfig policy, critic, sharedHead;

		int epochs = 2;
		float policyLR = 3e-4f; // Policy learning rate
		float criticLR = 3e-4f; // Critic learning rate

		float entropyScale = 0.018f; // The scale of the normalized entropy loss
		// Whether to ignore invalid actions in the entropy calculation.
		// True means that entropy will be determined only from available actions.
		// False means that entropy for unavailable actions will be zero, 
		//	meaning the entropy of the state is limited to the fraction of available actions in that state.
		bool maskEntropy = false; 

		float clipRange = 0.2f;
		
		// Temperature of the policy's softmax distribution
		float policyTemperature = 1;

		float gaeLambda = 0.95f;
		float gaeGamma = 0.99f;
		float rewardClipRange = 10; // Clip range for normalized rewards, set 0 to disable

		bool useGuidingPolicy = false;
		std::filesystem::path guidingPolicyPath = "guiding_policy/"; // Path of the guiding policy model(s)
		float guidingStrength = 0.03f;

		ReachabilityConfig reachability;
		ProposerConfig proposer;
		GoalCriticConfig goalCritic;

		PPOLearnerConfig() {
			policy = {};
			policy.layerSizes = { 256, 256, 256 };
			critic = {};
			critic.layerSizes = { 256, 256, 256 };
			sharedHead = {};
			sharedHead.layerSizes = { 256 };
			sharedHead.addOutputLayer = false;
		}
	};
}