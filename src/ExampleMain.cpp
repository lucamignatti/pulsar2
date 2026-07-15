#include <GigaLearnCPP/Learner.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>

#include <RLGymCPP/Rewards/CommonRewards.h>
#include <RLGymCPP/Rewards/ZeroSumReward.h>
#include <RLGymCPP/TerminalConditions/NoTouchCondition.h>
#include <RLGymCPP/TerminalConditions/GoalScoreCondition.h>
#include <RLGymCPP/TerminalConditions/AttemptResolutionCondition.h>
#include <RLGymCPP/ObsBuilders/DefaultObs.h>
#include <RLGymCPP/ObsBuilders/AdvancedObs.h>
#include <RLGymCPP/ObsBuilders/AdvancedObsPadded.h>
#include <RLGymCPP/StateSetters/KickoffState.h>
#include <RLGymCPP/StateSetters/RandomState.h>
#include <RLGymCPP/StateSetters/BallNearCarState.h>
#include <RLGymCPP/StateSetters/AirDrillState.h>
#include <RLGymCPP/StateSetters/FrontierDrillState.h>
#include <RLGymCPP/StateSetters/CombinedState.h>
#include <RLGymCPP/ActionParsers/DefaultAction.h>

using namespace GGL; // GigaLearn
using namespace RLGC; // RLGymCPP

// Discount factor, shared by the learner's GAE and every PBRS reward (they MUST match or the
// potential terms stop telescoping against GAE). tickSkip 4 => 30 actions/sec, so 0.9985 gives a
// ~15s half-life (462 steps) — a deliberately long horizon so full scoring/defensive possessions
// reach the value target. If you change tickSkip, re-derive this: half-life_s = ln2 / (-ln gamma) / (120/tickSkip).
static constexpr float TRAIN_GAMMA = 0.9985f;

// ---- 4.0: one net for 1v1/2v2/3v3 (team-play program) ------------------------------------
// The 4.0 lineage trains a SINGLE policy across team sizes via a padded obs
// (AdvancedObsPadded: fixed width at MAX_PLAYERS_PER_TEAM, zero-padded slots + presence
// flags, slots shuffled per build so every slot's weights train even while the mix is
// mostly 1v1). PHASE A (this config): every arena is 1v1 — the proven bootstrap; June's
// multi-mode-from-scratch runs diluted touch data 4x and froze, pure-1v1 phase-1 fixed it.
// PHASE B (once 1v1 performs decently): raise FRAC_2V2/FRAC_3V3 — a config-only change,
// checkpoint-compatible because the obs width never changes.
static constexpr int MAX_PLAYERS_PER_TEAM = 3;

// PHASE B arena fractions, applied by arena index with the team arenas at the END of the
// index range (deterministic across restarts). Keeping them at the end means: (a) the
// steering practice arenas (the FIRST practiceArenaFrac of indices) stay 1v1, and (b) the
// league eval EnvSet — which clones this create-func with small numArenas, so only low
// indices — stays 1v1. The skill tracker gets its OWN create-func (SkillEnvCreateFunc)
// with the same trailing-team layout scaled to its small eval fleet, so PHASE B also
// plays 2v2/3v3 eval matches and wandb gains Rating/2v2 + Rating/3v3; its arena 0 is
// always 1v1, keeping Rating/1v1 continuous across the phase switch (it is also the
// guard key: the Learner tracks arena 0's mode). Keep the fractions + practiceArenaFrac
// well under 1 so a healthy 1v1 match population remains for steering derivation.
// 2026-07-14: the derivation EXCLUDES team-arena rows (steerPractice group 3) - the
// self-only possession labels scored a teammate's race win as "nobody got it", and the
// polluted pool coincided with the gate duty-cycling around zero all PHASE B (wandb).
// Team arenas get their own correctly-team-labeled Steer/PossWin TeamMatch panel
// instead: measurement first, team-mode steering only as its own later experiment.
static constexpr float PHASE_B_FRAC_2V2 = 0.20f;
static constexpr float PHASE_B_FRAC_3V3 = 0.15f;

// AUTOMATIC PHASE-B TRIGGER ("once 1v1 performs decently", made mechanical): when
// Rating/1v1 posts PHASE_B_TRIGGER_STREAK consecutive skill-tracker evals at or above
// PHASE_B_RATING_TRIGGER, the iteration callback below writes the PHASE_B_MARKER file
// into the checkpoint folder, checkpoints, and exits with PHASE_B_RESTART_EXIT_CODE.
// run_trainer.sh treats any nonzero exit as restart -> the relaunch finds the marker and
// builds the PHASE B fleet, resuming the same checkpoint (obs width never changes).
// 1200 sits ~1 noise-band below the measured 1v1 plateau (~1250-1300 on the 3.1 lineage);
// the 3-eval streak filters single-eval noise (band is +-30-50). The marker is
// lineage-scoped: wiping/branching checkpoints_4.0 resets the curriculum with it.
static constexpr float PHASE_B_RATING_TRIGGER = 1200.0f;
static constexpr int PHASE_B_TRIGGER_STREAK = 3;
static constexpr int PHASE_B_RESTART_EXIT_CODE = 99; // nonzero and outside the wrapper's stop set {0,130,143}
static constexpr const char* PHASE_B_MARKER = "PHASE_B_ENGAGED";

// Team spirit for the zero-sum reward terms: own*(1-ts) + teamMean*ts - oppTeamMean.
// An algebraic no-op in 1v1 (teamMean == own), so this only changes the 2v2/3v3 arenas.
// 0.0 -> 0.3 (2026-07-14, PHASE B team-play fix, its own lever now that the mode-mix flip
// has landed): at ts=0 teammates were pure competitors for every per-player term
// (TouchAccel/Demo/Boost/AerialTouch/OpposedSave) — the only shared signal was GoalReward.
// 0.3 is the low end of the Necto/Nexto-lineage range (0.3-0.6): teammates now bank 30%
// of each other's touches/demos/saves, weak enough that personal credit still dominates
// data efficiency early. Shipped together with the team-closest BallProximityPotential
// (see BuildRewards below) — both are inert on 1v1 rows by construction.
//   Watch: Rating/2v2 + Rating/3v3 slope vs the pre-change trend; teammate-proximity /
//   double-commit behavior in the 2v2 viz. Revert = set back to 0.0 (resume-compatible,
//   nothing checkpointed depends on it). Escalate toward 0.5 only as its own change.
static constexpr float TEAM_SPIRIT = 0.3f;

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

// FRONTIER-9 reward stack (workflow-designed: understand -> research -> 5 competing designs ->
// adversarial red-team+math+integration verification -> synthesis). Replaces SURGICAL-7.
//
// Backbone kept BIT-IDENTICAL to the proven 1132-Elo lineage (B2G 75, TouchAccel 10, Demo 37.5,
// Goal 150 = 272.5 of 299 incumbent weight): the regression lesson was "the proposer broke it, not
// the reward", so this changes only the terms the measurements convicted (TouchHeight's avg-shape
// gradient produced Aerial Touch Ratio 0.00125 after 17B+ steps; per-player proximity/boost were
// the biggest PSD-fitness polluters).
//
// Whole-stack invariant: every component is exactly zero-sum or antisymmetric (nothing gated) ->
// the stack sums to 0 across both players every step, so PSD probe fitness is a pure competitive
// margin instead of being dominated by farmable per-player income. No PBRS is ever gated (breaks
// telescoping). ShotReward is deliberately NOT used: GameEventTracker's shot detector attributes
// the shooter as the team OPPOSITE the threatened net, so a solo player blasting the ball at their
// OWN net (after any opponent touch) arms a phantom shot credited to the opponent - a red-teamed,
// source-verified exploit (GameEventTracker.cpp). OpposedSaveReward below closes the matching
// self-save farm with a last-touch guard instead.
std::vector<WeightedReward> BuildRewards(float gamma) {
	return {
		// ---- Proven core: bit-identical economics to the 1132-Elo run -------------
		// Ball->goal potential (Nexto state_quality). Antisymmetric between teams, so ALREADY
		// zero-sum - no ZeroSum wrapper (that would silently 2x it). exp() concentrates credit at
		// the goal mouth and refunds rolled-back balls in full. gamma MUST equal the learner's
		// gaeGamma (threaded from TRAIN_GAMMA) or the potential stops telescoping against GAE.
		{ new BallToGoalPotentialReward(gamma), 75.f },

		// Touch quality (Nexto touch_accel): pays only for adding ball speed, ~10 total 0->110kph,
		// zero-sum so touches can't be co-farmed. Produced the 1933uu/s goal speed.
		{ new ZeroSumReward(new TouchAccelReward(), TEAM_SPIRIT), 10.f },

		// Demo: unchanged pair swing (75 = goal/2). TEAM_SPIRIT is 0 in PHASE A (was 0.5 once,
		// an algebraic no-op in 1v1 that misleadingly implied a halving living in the weight).
		{ new ZeroSumReward(new DemoReward(), TEAM_SPIRIT), 37.5f },

		// ---- Hygiene conversions (PSD fitness cleanup) ----------------------------
		// Proximity race: ZeroSum of an exact PBRS is still exact PBRS (Psi = Phi_own - Phi_opp),
		// so policy-invariance holds while the stack's biggest per-player PSD polluter becomes "be
		// closer to the ball than the opponent". Weight halved (7.5 -> 4) since the ZS swing
		// doubles. Never gate a potential.
		// TEAM-CLOSEST since 2026-07-14 (4.0 team play): Phi = the closest alive TEAM car's
		// proximity, so in 2v2/3v3 the race is "our nearest man vs theirs" and the 2nd/3rd man
		// earns nothing for crowding the ball — per-player proximity was the stack's biggest
		// ball-chasing gradient. Identical in 1v1 (team of one), see BallProximityPotentialReward.
		{ new ZeroSumReward(new BallProximityPotentialReward(gamma), TEAM_SPIRIT), 4.f },

		// Boost economy: demo-respawn guarded (mandatory under ZeroSum - without it the demoer is
		// charged for the victim's respawn tank), raised 4 -> 6, UNGATED. Zero-sum replaces the
		// gate as anti-farm: mutual pad cycling cancels, denial is a real 1v1 skill, and gating a
		// ZS term breaks its symmetry (positive side muted, mirror charged in full).
		{ new ZeroSumReward(new GuardedPickupBoostReward(), TEAM_SPIRIT), 6.f },

		// ---- The frontier terms ----------------------------------------------------
		// Aerial STRIKE (replaces gated ZeroSum(TouchHeight) 15): AND of sustained flight and a
		// genuinely-high ball, impulse-scaled, ~0.8s refire cooldown. Pays exactly 0 for everything
		// the bot currently does (ground strikes, wall pins, 193uu hop-pokes). UNGATED: the gate
		// structurally discounts never-achieved states and made the old ZS pair net-negative.
		{ new ZeroSumReward(new AerialTouchReward(), TEAM_SPIRIT), 25.f },

		// Pre-touch aerial approach potential: pays the jump-and-climb toward a high ball
		// immediately, refunds the whiff - the gradient that exists BEFORE the first air touch
		// ever lands. Exact PBRS: telescopes to ~0 net, cannot be farmed. NEVER gate.
		{ new ZeroSumReward(new AirInterceptPotentialReward(gamma), TEAM_SPIRIT), 10.f },

		// THE defensive signal (the stack's first): engine-refereed save, guarded so only
		// genuinely opponent-created shots pay. Deliberately NO paired ShotReward (see file header
		// - phantom-farmable). UNGATED - the gate's attack-oriented level is lowest exactly in the
		// own-half states where saves fire.
		{ new ZeroSumReward(new OpposedSaveReward(), TEAM_SPIRIT), 25.f },

		// The objective. Scorer +150 / conceder -150, exactly zero-sum.
		{ new GoalReward(), 150 }
	};
}

// Steered-practice arena split (set in main() from cfg.steering before the Learner is built;
// the Learner applies the SAME first-N rule for row tagging + collection steering, so these
// two sites agree by construction). 0 = feature off = every arena is a normal match arena.
static int g_NumPracticeArenas = 0;

// Frontier reset pool (STEERING_ROADMAP phase 3): the Learner banks feasible-but-declined
// readings; FrontierDrillState resets PRACTICE arenas into perturbed copies. Created in
// main() when steering is on (never in render mode). The practice-slice arithmetic in
// IsPracticeArena (below the team-split globals) MUST match the Learner's per-mode
// steerBlocks computation (leading practiceArenaFrac of each mode's contiguous arena
// block, >= 2 or none).
static std::shared_ptr<RLGC::FrontierPool> g_FrontierPool;
static float g_PracticeArenaFrac = 0.f;

// Team-mode arena split (set in main(): zero in PHASE A, the PHASE_B_FRAC_* fractions once
// the phase marker exists). Team arenas occupy the END of the index range — see the
// PHASE_B_FRAC_2V2 comment for why.
static int g_NumGames = 0;
static int g_NumArenas2v2 = 0;
static int g_NumArenas3v3 = 0;

// Same trailing-team layout for the skill tracker's own small eval fleet (set in main()
// alongside the split above; zero in PHASE A). Separate from the training split because the
// eval EnvSet has its own numArenas — cloning the training create-func there would land every
// eval arena on a low (= 1v1) index, which is exactly why Rating/2v2 / Rating/3v3 never
// appeared in wandb after the flip.
static int g_SkillNumArenas = 0;
static int g_SkillArenas2v2 = 0;
static int g_SkillArenas3v3 = 0;

// Phase-B trigger state (iteration callback below)
static bool g_PhaseB = false;
static int g_PhaseBStreak = 0;

// Per-mode practice-slice membership - MUST mirror Learner.cpp's steerBlocks arithmetic
// (per-mode contiguous blocks, leading practiceArenaFrac slice, none when < 2)
static bool IsPracticeArena(int index) {
	int n2 = g_NumArenas2v2, n3 = g_NumArenas3v3, n1 = g_NumGames - n2 - n3;
	int starts[3] = { 0, n1, n1 + n2 }, counts[3] = { n1, n2, n3 };
	for (int md = 0; md < 3; md++) {
		int numPractice = RS_CLAMP((int)(counts[md] * g_PracticeArenaFrac), 0, counts[md]);
		if (numPractice < 2)
			continue;
		if (index >= starts[md] && index < starts[md] + numPractice)
			return true;
	}
	return false;
}

// Render mode only (GGL_RENDER_TEAM_SIZE, set in main()): the viewer's single arena plays
// this team size regardless of the curriculum split. 0 = not in render mode. The padded obs
// makes any size checkpoint-compatible, so the viewer can watch 1v1/2v2/3v3 off the same run.
static int g_RenderTeamSize = 0;

// Shared env body for the training and skill-eval create-funcs: everything except the
// team size and the practice-terminal decision is identical (and the skill tracker
// overwrites eval rewards/state setters/terminal conditions anyway — for eval arenas only
// the arena's team size, the obs builder and the action parser matter).
static EnvCreateResult MakeEnv(int playersPerTeam, bool practiceArena) {
	std::vector<TerminalCondition*> terminalConditions = {
		new NoTouchCondition(10),
		new GoalScoreCondition()
	};

	// Practice arenas (steered-practice collection): episodes additionally end - as a TRUE
	// terminal, no value bootstrap - the moment an airborne-ball attempt resolves, so a whiff's
	// counterattack never enters the return. Match arenas keep full consequences.
	if (practiceArena)
		terminalConditions.push_back(new AttemptResolutionCondition());

	auto arena = Arena::Create(GameMode::SOCCAR);
	for (int i = 0; i < playersPerTeam; i++) {
		arena->AddCar(Team::BLUE);
		arena->AddCar(Team::ORANGE);
	}

	EnvCreateResult result = {};
	result.actionParser = new DefaultAction();
	// Padded team obs (4.0): fixed 230-dim for any team size up to 3v3 (51 header + 6x29
	// player slots + 5 presence flags). Ball@0 and self@51 offsets match AdvancedObs exactly,
	// which the steering landing sims and analysis tooling rely on. NOT weight-compatible
	// with the 3.1 lineage's 109-dim AdvancedObs checkpoints - 4.0 is a fresh run.
	result.obsBuilder = new AdvancedObsPadded(MAX_PLAYERS_PER_TEAM);
	// The proven cf993b7 reset mix - effective near-ball share 0.55 (0.35 ground + 0.20 aerial
	// drill), no drill-replay slice (that came with the drill bank in the regression).
	result.stateSetter = new CombinedState({
		// Ground touch bootstrap - the proven anti-freeze state
		{ new BallNearCarState(600, 900), 0.35f },
		// Aerial drill, REVERSE CURRICULUM: car spawns ALREADY AIRBORNE and climbing at an overhead
		// ball, boost-fed, so it only has to COMPLETE the touch. Replaces the old ground-ring drill
		// (ball overhead but both cars grounded + ball descending), which let the bot wait for the
		// ball to fall and hit it on the ground -> no aerial ever required (0.1% aerial touches after
		// 17B steps). Now that AerialTouch/AirInterceptPotential pay for the airborne touch, this
		// gives the exposure the reward alone can't buy.
		{ new AirDrillState(), 0.20f },
		{ new KickoffState(), 0.15f },
		// Sole source of chaotic/defensive/air-recovery states (bounds widened to corners/goal lines)
		{ new RandomState(true, true, false), 0.30f },
	});
	result.terminalConditions = terminalConditions;
	result.rewards = BuildRewards(TRAIN_GAMMA);

	result.arena = arena;

	return result;
}

// Create the RLGymCPP environment for each of our games (the TRAINING fleet).
// PHASE A: every arena is 1v1. PHASE B raises FRAC_2V2/FRAC_3V3, putting team modes on the
// trailing indices — the padded obs keeps the net identical either way, so the phase switch
// is checkpoint-compatible.
EnvCreateResult EnvCreateFunc(int index) {
	// Render viewer: one arena, team size picked by GGL_RENDER_TEAM_SIZE (never a practice arena)
	if (g_RenderTeamSize > 0)
		return MakeEnv(g_RenderTeamSize, false);
	int playersPerTeam = 1;
	if (g_NumGames > 0 && index >= g_NumGames - g_NumArenas3v3)
		playersPerTeam = 3;
	else if (g_NumGames > 0 && index >= g_NumGames - g_NumArenas3v3 - g_NumArenas2v2)
		playersPerTeam = 2;
	EnvCreateResult result = MakeEnv(playersPerTeam, index < g_NumPracticeArenas);
	// Phase 3: practice arenas draw a share of their resets from the frontier pool
	// (perturbed copies of feasible-but-declined readings; offline-validated dose:
	// useFrac 0.35, pos/vel noise 250). Falls back to the normal mix while the pool
	// is empty or stale, so cold boots and quiet iterations behave exactly as before.
	if (g_FrontierPool && IsPracticeArena(index))
		result.stateSetter = new FrontierDrillState(g_FrontierPool, result.stateSetter, 0.35f, 250, 250);
	return result;
}

// Create-func for the skill tracker's eval fleet: the same trailing-team layout, scaled to
// the eval EnvSet's g_SkillNumArenas, so PHASE B evals also play 2v2/3v3 and per-mode Elo
// (Rating/2v2, Rating/3v3) reaches wandb. Arena 0 is always 1v1 — it feeds the render
// sender, and Rating/1v1 (the guard/trigger key) stays continuous across the flip. Never
// a practice arena: eval matches must keep full consequences.
EnvCreateResult SkillEnvCreateFunc(int index) {
	int playersPerTeam = 1;
	if (g_SkillNumArenas > 0 && index >= g_SkillNumArenas - g_SkillArenas3v3)
		playersPerTeam = 3;
	else if (g_SkillNumArenas > 0 && index >= g_SkillNumArenas - g_SkillArenas3v3 - g_SkillArenas2v2)
		playersPerTeam = 2;
	return MakeEnv(playersPerTeam, false);
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

	// Per-platform default, deliberately NOT AUTO: the training box must fail LOUDLY if CUDA
	// goes missing (AUTO would silently fall back to CPU at ~1/100 speed on a driver hiccup),
	// and the Mac checkout (render/analysis/smoke) should use the Apple GPU without env vars.
#ifdef __APPLE__
	cfg.deviceType = LearnerDeviceType::GPU_MPS;
#else
	cfg.deviceType = LearnerDeviceType::GPU_CUDA;
#endif
	// Override the device with GGL_DEVICE=cpu|cuda|mps|auto. Handy for running a live viewer
	// (GGL_RENDER=1) on CPU while the GPU is fully occupied by a training process — a single
	// render arena is cheap.
	if (const char* d = std::getenv("GGL_DEVICE")) {
		std::string dev = d;
		if (dev == "cpu" || dev == "CPU")        cfg.deviceType = LearnerDeviceType::CPU;
		else if (dev == "auto" || dev == "AUTO") cfg.deviceType = LearnerDeviceType::AUTO;
		else if (dev == "mps" || dev == "MPS")   cfg.deviceType = LearnerDeviceType::GPU_MPS;
		else                                     cfg.deviceType = LearnerDeviceType::GPU_CUDA;
	}

	// tickSkip 4 (30 actions/sec vs the old 15): finer control for mechanics — the biggest ceiling
	// lever for aerials/dribbles/speed-flips. Halves game-time throughput per policy-step, so it is
	// paired with a fresh run (not a resume) and the longer-horizon gamma below (halving tickSkip
	// halves game-time horizon at fixed gamma, so gamma is raised to keep it).
	cfg.tickSkip = 4;
	cfg.actionDelay = cfg.tickSkip - 1; // Normal value in other RLGym frameworks

	cfg.numGames = 1024;

	// Pipelined collection: collect iteration N+1 (worker, frozen policy snapshot) while N
	// processes+learns. Collection and consumption are near-equal (~0.6s each at ts8) and fully
	// sequential without this — the single biggest throughput lever, ~1.5-2x.
	// KILL SWITCH if anything looks off (ratio/KL spikes, entropy crash, Elo bleed):
	// set false, rebuild, restart — the flag-off path is the exact pre-pipeline sequential code.
	cfg.pipelinedCollection = true;

	// Leave this empty to use a random seed each run
	cfg.randomSeed = 123;

	// 200k, the proven-good-era value (the recent uncommitted 100k edit was NOT in the good run).
	int tsPerItr = 200'000;
	cfg.ppo.tsPerItr = tsPerItr;
	cfg.ppo.batchSize = tsPerItr;
	// 100k (was 200k): at 512-wide the learn pass holds autograd for policy+critic+goal-critic
	// simultaneously — 200k rows peaked ~12.7GB of saved activations and OOM'd the 16GB card at the
	// goal critic's forward. Halving the minibatch halves that peak; the minibatch loop accumulates
	// gradients so the update is mathematically identical, just two passes instead of one.
	// 50k (2026-07-12, late): OOM'd again mid-run with ~6GB of the card held by DESKTOP GRAPHICS
	// (viz viewer + browser) - the trainer's burst headroom was ~700MB. 50k halves the learn peak
	// again (same accumulated math, 4 passes). Side effect kept in mind: the reachability InfoNCE
	// subsample (512/minibatch) now runs 4x per epoch instead of 2x - mildly more aux training,
	// same as the accepted 200k->100k change. Revert to 100k if the graphics pressure goes away.
	cfg.ppo.miniBatchSize = 50'000;

	// BF16 inference for collection + GAE value preds. rho/gate evals request fp32 explicitly and
	// grad-enabled forwards (InfoNCE training) always run fp32, so the gate is unaffected.
	cfg.ppo.useHalfPrecision = true;

	// GGL_SMOKE=1: shrink the fleet + iteration so an offline sandbox (Mac CPU, resuming a
	// COPY of the real checkpoints with WANDB_MODE=offline) can complete iterations in
	// tens of seconds and exercise the full loop - env creation, checkpoint resume, boot
	// probe, steering derivation/gates, metric send. Everything else stays production.
	// NEVER set on the training box.
	if (const char* s = std::getenv("GGL_SMOKE"); s && s[0] && std::string(s) != "0") {
		cfg.numGames = 128;
		int smokeTs = 25'000;
		cfg.ppo.tsPerItr = smokeTs;
		cfg.ppo.batchSize = smokeTs;
		cfg.ppo.miniBatchSize = smokeTs;
		// bf16 is a CUDA fast path; on CPU (the smoke device) it hits conversion-per-op
		// slow paths that stretch one iteration into tens of minutes
		cfg.ppo.useHalfPrecision = false;
		// Sequential collection: (a) a smoke should be deterministic; (b) torch-cpu on
		// macOS produced non-finite trunk outputs from FINITE weights+obs exactly when the
		// learn pass started overlapping the collect worker's inference (concurrent
		// Accelerate GEMMs from two threads) - an environment bug this box's CUDA build
		// does not have. Diagnosed 2026-07-14 via the InferPolicyProbsFromModels probe.
		cfg.pipelinedCollection = false;
		RG_LOG("GGL_SMOKE: numGames 128, tsPerItr 25k, fp32, sequential (offline sandbox smoke)");
	}

	cfg.ppo.epochs = 2;
	cfg.ppo.entropyScale = 0.035f;

	// Reachability: aux InfoNCE heads on the shared trunk. gateEnabled = false under FRONTIER-9:
	// every reward component is now zero-sum/antisymmetric and ungated by design (gating breaks ZS
	// symmetry - positive side muted, mirror charged in full - and PBRS telescoping), so the gate
	// would be a mathematical no-op anyway; this makes that explicit. Keep enabled=true: the heads
	// still feed the InfoNCE trunk aux + Reach/* plasticity canaries.
	cfg.ppo.reachability.enabled = true;
	cfg.ppo.reachability.gateEnabled = false;
	// Third goal-space head (2026-07-14): canonical CAR pos+vel - the movement-capability
	// frontier for META steering. Offline (conservative frozen-phi test): calibration
	// DECISIVELY monotone (~7x the ball head's margin; window 45 chosen by margin across
	// {20,45,90}); per-cluster causal steerability NOT yet demonstrated offline (0/3
	// clusters clean). Revert = false. Record: analysis/probes/carstate_head_validate.py.
	// INCIDENT (2026-07-14, same day): the first deployment let this head's InfoNCE
	// co-train phi AND the shared trunk while the fresh head was at chance - its loss
	// alone (~2x every other aux term combined; Reach/Aux Loss 0.5 -> 1.0+) churned the
	// policy's representation and Rating slid ~125 off its 1462 peak across ALL modes in
	// ~500 iterations, with every guard green (no guard watched this gradient path). The
	// head now trains DETACHED by default (carStateCouple = 0, PPOLearnerConfig.h): only
	// psi_carstate gets gradients - exactly the offline-validated frozen-phi regime the
	// detector gate passed in. Its loss reports separately (Reach/Car State Loss) so
	// Reach/Aux Loss keeps its ~0.5 baseline = the live verification that this is fixed.
	cfg.ppo.reachability.carStateHead = true;
	// With the gate off, the rho/gate reads (3 full-buffer model passes/iter, the biggest single
	// consumption cost after PPO Learn) feed only the Reach/* panels — refresh those every 16
	// iterations instead. The InfoNCE trunk aux (the part that helps learning) is unaffected.
	cfg.ppo.reachability.diagEveryIters = 16;

	// Explicitly OFF - the regression. ProposerConfig defaults enabled=true upstream, so this
	// override is what actually keeps the proposer / drill bank / car-proposer / HRL machinery
	// out of this build (no drill bank is ever constructed or attached, either).
	cfg.ppo.proposer.enabled = false;

	// Wide clip (cold return-sigma is ~2-4 under this near-sparse stack; default 10 compressed the
	// first goals). 50 releases the full 150 once sigma >= 3 and bounds the tail.
	cfg.ppo.rewardClipRange = 50;

	cfg.ppo.gaeGamma = TRAIN_GAMMA; // ~15s half-life at 30Hz; MUST match the PBRS reward gammas above

	// Secondary goal-only critic: long-horizon credit on the one unfarmable signal. Independent net,
	// raw obs in; advantages blended at beta = 25% of dense-advantage scale (std-matched, centered).
	// VALIDATION: GoalCritic/Value-Outcome Corr and Adv-Outcome Corr must be POSITIVE once goals flow;
	// negative = channel/sign bug -> set beta = 0 (critic still trains, no blend) and investigate.
	cfg.ppo.goalCritic.enabled = true;
	cfg.ppo.goalCritic.gamma = 0.9997f;   // ~77s half-life at 30Hz ("huge distance", per plan)
	cfg.ppo.goalCritic.beta = 0.25f;
	cfg.ppo.goalCritic.lr = 1.5e-4f;
	cfg.ppo.goalCritic.model.layerSizes = { 512, 512, 512 };

	cfg.ppo.policyLR = 1.5e-4;
	cfg.ppo.criticLR = 1.5e-4;

	// 512-wide (was 256): ~4x params. The saturation probe read the 256 net at 49-74% spectral
	// utilization (not saturated), but the head's effective rank was grinding down all run and this
	// is a FRESH run where underused capacity is nearly free while missing capacity compounds over
	// an unattended week. Inference is latency-bound on the 5080 at these sizes, so the SPS cost is
	// modest (~10-25%), not proportional to the param increase.
	cfg.ppo.sharedHead.layerSizes = { 512, 512 };
	cfg.ppo.policy.layerSizes = { 512, 512, 512 };
	cfg.ppo.critic.layerSizes = { 512, 512, 512 };
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
	cfg.ppo.goalCritic.model.activationType = activation;

	bool addLayerNorm = true;
	cfg.ppo.policy.addLayerNorm = addLayerNorm;
	cfg.ppo.critic.addLayerNorm = addLayerNorm;
	cfg.ppo.sharedHead.addLayerNorm = addLayerNorm;
	cfg.ppo.reachability.phi.addLayerNorm = addLayerNorm;
	cfg.ppo.reachability.psi.addLayerNorm = addLayerNorm;
	cfg.ppo.goalCritic.model.addLayerNorm = addLayerNorm;

	// Skill rating: Elo-style eval matches vs saved versions (logged as Rating/1v1). Also turns on
	// savePolicyVersions. This was ON in the good era. NOTE: this only EVALUATES against old
	// versions - it does NOT train against them. The separate trainAgainstOldVersions (self-play
	// league) is deliberately left OFF: it was not part of the proven-good config (it was a recent
	// uncommitted addition). Flip it on later as its own experiment if desired.
	cfg.skillTracker.enabled = true;

	// FRESH RUN (4.0): the team-play lineage - padded 230-dim obs (AdvancedObsPadded(3)),
	// PHASE A all-1v1 curriculum, otherwise the proven 3.1 config (tickSkip 4, 512-wide,
	// gamma 0.9985, secondary goal critic). Its own checkpoint folder + wandb run name so it
	// can NEVER accidentally resume the 3.1 lineage (obs 109 -> 230; the loader would abort
	// on the trunk's first Linear anyway, but the folder split keeps the failure impossible
	// rather than merely loud).
	cfg.checkpointFolder = "checkpoints_4.0";
	cfg.metricsRunName = "4.0-team";

	// 1M default => a save every ~6s at ~170k SPS, making the 8-deep rotation window ~50
	// SECONDS wide - which is why the 2026-07-13 GPU lockup poisoned EVERY checkpoint in
	// it. 25M = a save every ~2.5 min, window ~20 min, and far less IO. Worst-case crash
	// loss rises from ~6s to ~2.5min of training - the wrapper restart costs more anyway.
	cfg.tsPerSave = 25'000'000;
	// Golden archive + boot sanity probe use LearnerConfig defaults (keep 3 best-rated
	// checkpoints outside rotation; probe loaded checkpoints rated >= 400).

	cfg.sendMetrics = true; // Send metrics

	// Render/visualization mode. Off by default (this binary trains). Set GGL_RENDER=1 to instead
	// run a single-arena live viewer against an existing run: it loads the newest checkpoint in
	// cfg.checkpointFolder and hot-swaps in newer ones as the trainer writes them, so you can watch
	// the model improve in real time without disturbing the training process. GGL_RENDER_RELOAD_SECS
	// overrides the poll interval (seconds; <= 0 pins to the checkpoint loaded at startup).
	cfg.renderMode = false;
	if (const char* r = std::getenv("GGL_RENDER"); r && r[0] && std::string(r) != "0")
		cfg.renderMode = true;
	if (const char* s = std::getenv("GGL_RENDER_RELOAD_SECS"))
		cfg.renderReloadSecs = (float)std::atof(s);
	// GGL_RENDER_TEAM_SIZE=1|2|3 picks the viewer arena's mode (default 1v1). Render mode only;
	// the training fleet's split is owned by the phase curriculum, never by an env var.
	if (cfg.renderMode) {
		g_RenderTeamSize = 1;
		if (const char* t = std::getenv("GGL_RENDER_TEAM_SIZE")) {
			int n = std::atoi(t);
			if (n < 1 || n > MAX_PLAYERS_PER_TEAM)
				RG_ERR_CLOSE("GGL_RENDER_TEAM_SIZE must be 1.." << MAX_PLAYERS_PER_TEAM << ", got \"" << t << "\"");
			g_RenderTeamSize = n;
		}
		RG_LOG("Render mode: " << g_RenderTeamSize << "v" << g_RenderTeamSize << " arena");
	}

	// ---------------------------------------------------------------------------------------------
	// Basin-Racing (PSD) + QD league. Both ADDITIVE and OFF by default: with these two flags false
	// the trainer is byte-for-byte the proven 2.6 baseline. Enable to layer the outer loop on top.
	//
	// PSD alternates ordinary DESCEND (== the baseline above) with EGGROLL PROBE rounds: K antithetic
	// low-rank perturbations of the policy each get a short factor-only finetune, are scored on
	// held-out arenas, and the fitness-weighted sum of the ORIGINAL directions is folded into the
	// base weights. warmupUntilPlateau keeps pure DESCEND until Rating/1v1 stalls, so the K-cost is
	// only paid where PPO alone plateaus (resuming the live 2.6 checkpoint trips this immediately).
	// FULL RUN: Basin-Racing + QD league both ON.
	// warmupUntilPlateau keeps pure DESCEND (== the 2.6 baseline) until Rating/1v1 stalls, so the
	// K-cost is only paid at a plateau: resuming a competent 2.6 checkpoint trips it quickly; a
	// from-scratch start descends normally until its first plateau (~GExploit iters). K=16 -> 32
	// antithetic probes; numGames 1024 splits cleanly (24 probe + 8 eval arenas per slot).
	// DISABLED 2026-07-13 by pre-registered measurement (see the evalWindowSteps note below
	// for the full trial design): with 12x eval data per slot, the ES slot-ranking
	// reliability read 0.20 and 0.23 across two probe rounds - the fitness signal is
	// structurally unresolvable (episode-level variance in zero-sum 1v1 swamps
	// perturbation-scale effects at any affordable budget), so every fold is a noise kick.
	// A live-bracketed fold (reliability 0.57, norm 17.3) measurably degraded behavior
	// (engagement -7.6pp ~3sigma). Disabling also disarms the plasticity interventions,
	// which were past their 25k-iteration warmup. The theory of ES stands; its
	// precondition - a cheap reliable fitness - does not exist in this domain.
	cfg.psd.enabled = false;
	cfg.psd.warmupUntilPlateau = true;
	// Pure-ES probe (EGGROLL-faithful; arXiv 2511.16652). The 32-slot Baldwinian probe measured
	// fitness reliability ~0 for 20 rounds -> the ranking was noise, so folds were a random walk.
	// The paper's own sweeps show ES needs LARGE populations; K=256 -> S=512 antithetic slots puts N
	// inside their proven envelope. numGames 1024 -> ~2 eval arenas/slot; the round budget goes to a
	// long level-fitness window instead of a per-slot finetune that couldn't discriminate anyway.
	cfg.psd.pureES = true;
	cfg.psd.K = 256;                      // S = 512 slots (was 16 -> S=32, 16x below the paper's floor)
	cfg.psd.rank = 4;
	cfg.psd.sigma = 0.02f;                // adaptive: widens on low reliability, tightens if most slots hurt
	// 200 -> 2400 (2026-07-13): fold direction-SNR scales with sqrt(total eval steps), and at
	// 200 steps/slot the slot ranking's measured reliability sat at 0.13-0.65 (mostly ~0.55 -
	// half the fold was noise) while norm-15..20 folds landed anyway; a bracketed fold at
	// reliability 0.57 measurably degraded behavior (engagement -7.6pp, ~3 sigma). 12x the
	// eval window costs ~30-60s of wall per round (slots parallelize across all arenas) and
	// should lift reliability toward 0.8+. PRE-REGISTERED VERDICT: if reliability still reads
	// ~0.5 after 2-3 rounds at this budget, the noise is structural (non-stationary base /
	// antithetic pairing under drift) and PSD gets disabled per the measurement doctrine.
	cfg.psd.evalWindowSteps = 2400;
	cfg.psd.GExploit = 2500;
	// Validation-gated fold: A/B the base policy's held-out return before vs after each fold and revert
	// if it regressed, so no noise-fold ever lands unchecked (aggregate-level Baldwinian validation).
	cfg.psd.valGateEnabled = true;
	// 20 -> 200: the fold-approval A/B gets the same medicine - a 20-step window green-lit a
	// fold with fracHurt 0.58 and another at ranking reliability 0.13.
	cfg.psd.valWindowSteps = 200;

	// Plasticity interventions — the canaries now ACT (handoff §3.6/§4.2), not just log. Every knob
	// is gated to be a no-op under healthy training and fires only on real plasticity loss:
	//   (1) ReDo recycles dead policy units when >=10% collapse.
	//   (2) EffRank collapse response shrink-and-perturbs the head when its effective rank drops
	//       below 70% of its running peak.
	//   (3) Critic gets a gentle 10%-toward-init partial reset every 5 probe rounds (it loses value
	//       plasticity first and tolerates resets).
	//   (4) Distill reset (deepest) reinits + behavior-distills the policy, but only when the head
	//       rank has collapsed below 60% of peak AND >=8 rounds since the last distill.
	//   (5) Competence freeze is wired but dormant: set freezeRatingThresh to the Rating/1v1 at which
	//       you want the trunk locked in (the default never fires).
	cfg.psd.redoEnabled = true;
	cfg.psd.effRankResponseEnabled = true;
	cfg.psd.criticResetPeriod = 5;
	cfg.psd.criticPartialResetFrac = 0.10f;
	cfg.psd.distillPeriod = 8;
	cfg.psd.distillTrigger = 0.6f;
	cfg.psd.freezeEnabled = true;
	// FRESH run: interventions LOG from step 0 but only ACT after 25k iterations, so the naturally
	// steep early effective-rank drop of a cold net doesn't trip distill/perturb with nobody watching.
	cfg.psd.interventionWarmupIters = 25000;
	// cfg.psd.freezeRatingThresh = <Rating/1v1 to freeze the trunk at>;  // leave unset = never freeze

	// QD league: a MAP-Elites archive of behaviorally-diverse opponents so the main doesn't converge
	// to one playstyle and gets exposure to others. Now actually WIRED into training: descendOpponentFrac
	// of iterations face a PFSP-sampled league member (was dead code before - the archive existed but
	// never fed the training loop). Members are past selves (re-seeded each era) + their mutations;
	// exploiterSlots members hill-climb to attack the current main's weaknesses. Fitness = member goals
	// minus main goals over a match (the sign was inverted before, breeding the worst losers).
	cfg.league.enabled = true;
	cfg.league.gridAxes = { "in_air_ratio", "field_y", "boost_economy" };
	// 6 bins/axis + QUANTILE-ADAPTIVE edges (default-on in LeagueConfig). The uniform-[0,1] 4-bin
	// grid was measured nearly dead at 919M steps: the whole population lived in in_air [0.51,0.92],
	// field_y [0.37,0.62], boost [0.016,0.060] -> only 3 of 64 cells occupied, boost axis never left
	// bin 0, and 58 exploiter-unmapped wins said the grid couldn't name the styles that matter.
	// Quantile edges (rolling window of observed BDs, refreshed each reseed era) put every bin where
	// the population actually lives and track it as the bot improves all week.
	cfg.league.binsPerAxis = 6;
	cfg.league.exploiterSlots = 2;
	// 0.25 -> 0.35 (2026-07-12): the steered style measurably beats its recent self (42-24)
	// but loses to ARCHIVED styles (12-19 vs the 4.16B era) - nontransitive exploitability.
	// PFSP already prefers members that beat the main, so more league iterations = targeted
	// training against exactly the styles currently winning. Revert to 0.25 if Elo variance
	// rises without the deficit closing.
	cfg.league.descendOpponentFrac = 0.35f;
	cfg.league.reseedEveryIters = 1000;     // snapshot the current main as a fresh lineage this often
	cfg.league.competenceFloor = -25.0f;    // keep sparring partners that lose by a bit (style > winning)

	// ---------------------------------------------------------------------------------------------
	// Steered-practice collection ("optimism surgery"), ENABLED - carried over from the 3.1
	// lineage, where v2 (possession-outcome derivation + rho-band gate) ran with healthy guards.
	// The commitment direction is derived LIVE each iteration from the collected buffer itself
	// (ball-landing sims + went/declined trunk contrast, match-arena rows only, EMA-smoothed), and
	// a slice of practice arenas runs unsteered as controls so a causal auto-gate can drop alpha
	// to 0 the moment steering stops out-engaging the controls. No sidecar, nothing to babysit.
	// Protocol, measured effects, and REVERT path (flag off + resume the branch-point backup in
	// build/checkpoints_3.1_branch_backup/): analysis/probes/STEERED_PRACTICE.md.
	//   Watch: Steer/* panels (Engagement Steered vs Control, Gate Active, Dir Drift, Pairs);
	//   Player/Aerial Touch Ratio + contest metrics (must rise within ~a day or the mechanism
	//   isn't engaging); GAE ratio/KL (same off-policyness class as pipelinedCollection);
	//   Rating/1v1 slope vs the pre-switch trend as the revert trigger.
	// STAGE-1 steered collection ("optimism surgery", terminationless), ENABLED.
	// History: the first deployment (resolution-terminated practice episodes) tanked Elo twice
	// - both failures from the TERMINATION half (phantom -V(s_end) penalty; shared critic
	// cannot price aliased truncations). Stage 1 keeps episodes 100% NORMAL: the whiff tax
	// stays, reality filters the attempts, and the only change is WHERE experience comes from.
	//   - ~15% of arenas collect with the live-derived commitment direction (+1 sigma) added
	//     to the policy head's trunk input, RHO-BAND GATED: only in states the ball head rates
	//     as hard-but-plausible for scoring (per-batch quantile band). A few more arenas are
	//     unsteered controls for the causal engagement gate.
	//   - Guards, all automatic: engagement gate (steered must out-engage controls), rating
	//     drawdown guard (Rating/1v1 falling >75 below its slow EMA LATCHES steering off -
	//     the LearnerConfig default; 75 sits outside the +-30-50 noise band), ratio/KL logs,
	//     branch backup + quarantine ritual.
	//   - Watch: Steer/* panels (Alpha, Engagement Steered/Control/Match, Gate Delta EMA,
	//     RhoGate In-Band Frac, Rating Guard Tripped), aerial/contest metrics, Rating slope.
	// Post-mortems + stage-2 escalation path: analysis/probes/STEERED_PRACTICE.md.
	// STAGE-1 v2 (2026-07-12, after ~200M treated steps of v1): v1's guidance metric ("landing
	// attendance") aged out - pool-Elo drifted down while the style beat its predecessor 42-24
	// and lost to older selves 12-19. v2 re-aims the SAME machinery at a possession-outcome
	// definition: the direction contrasts "went and WON the race to the ball" vs "declined and
	// nobody got it" (first-touch events within the landing window), and the gate measures
	// possession-win rate steered-vs-control. Unfakeable by empty flight; doesn't age with
	// style. League old-style exposure raised below to patch the measured exploitability.
	cfg.steering.enabled = true;
	// 1.0 -> 0.5 (2026-07-12, the ratchet fix): steered rows learn through PPO's clipped IS,
	// and for actions steering makes MUCH likelier than the base policy (ratio << 1-clip) the
	// clip zeroes the gradient exactly when the advantage is NEGATIVE - successes reinforce,
	// punished failures are discarded. That one-way ratchet is how overcommit-then-concede
	// compounded into an Elo bleed despite real head-to-head gains. A smaller push keeps the
	// induced ratios mostly inside the clip window so both outcome signs teach.
	cfg.steering.alpha = 0.5f;
	cfg.steering.practiceArenaFrac = 0.18f;      // of EACH mode's arenas (leading slice per block)
	cfg.steering.resolutionTermination = false;  // STAGE 1: normal episodes, no exceptions
	// PER-MODE steering (2026-07-14): 2v2/3v3 arenas get their own steered/control slices,
	// gates, and sigmas; a team mode applies the 1v1 direction (offline-validated transfer,
	// teamWon +3.4pp @ +0.5 in 2v2) until its own WON-vs-NONE pool is rich enough to derive
	// one. The commitment frontier is much larger in team modes (collective declines:
	// 74%/88%/92% of feasible balls for 1v1/2v2/3v3). Revert = false (team arenas revert
	// to measurement-only panels).
	cfg.steering.steerTeamModes = true;
	// META frontier steering (2026-07-14, prior-free phase 4): goals from the agent's own
	// achieved bank, frontier by its own self-model, emergent clusters in its own psi
	// geometry, model-free attainment outcomes, per-cluster causal gates + a dwell
	// scheduler. Offline: ball-head calibration monotone (car head self-disables for
	// arbitrary goals), and at least one emergent cluster shows a monotone causal
	// attainment uplift (deep-own-half high ball: -1.002 -> -0.901 across alpha 0..1)
	// with clean canaries - plus cluster heterogeneity, the scheduler's raison d'etre.
	// INCIDENT (2026-07-14): v1 of this handed meta the steering slot PERMANENTLY - the
	// proven incumbent commitment direction (fresh off driving Rating 1380 -> 1462)
	// stopped actuating, rotating unproven cluster directions took its place, and
	// benching could never engage (150-iter warmup / 10-iter dwells). Contributed to a
	// ~125 all-mode Elo slide together with the carstate aux-loss churn (see above).
	// The slot is now TIME-MULTIPLEXED (LearnerConfig.h metaProbeEvery/metaPromoteMin):
	// the incumbent is the default actuator, meta probes one cluster every 3rd dwell,
	// and a cluster only owns exploit dwells after its measured effect EMA clears the
	// promotion bar - actuation is earned, never granted. Measurement attribution
	// follows the slot owner (the incumbent gate no longer grades meta-steered buffers
	// and vice versa). meta=false remains the pinned fallback AND the pre-registered
	// baseline: meta must beat it on Elo slope over a matched window or it reverts.
	// Watch: Meta/Owns Slot, Meta/* panels, Steer/Rating Peak.
	// RE-ENABLE SEQUENCE (2026-07-14, post-incident): steering returns in its PROVEN
	// configuration first - pinned commitment, meta OFF. This is the exact config that
	// drove 1380 -> 1462, it re-establishes value with clean attribution, and it IS the
	// pre-registered baseline the meta system must beat. Flipping meta back on is the
	// next one-lever experiment, judged on Elo slope vs this baseline with the peak
	// latch armed. (The meta machinery, banks and panels are all still built and the
	// probe/promote scheduler smoke-passed - this flag is the only thing holding it.)
	cfg.steering.meta = false;
	// Phase 1 (steered league opponents): style directions the opponent side occasionally
	// plays - hesitant/overcommit (commitment direction at offline-validated negative /
	// mild positive dose) and shadow (live challenge-vs-shadow contrast). Synthesized
	// LIVE in the trainer from the same per-iteration derivations that drive collection
	// steering (2026-07-15, replacing the frozen steering_styles.json: pinned vectors rot
	// within ~75M steps, so a checkpoint-stale file was diversity in name only; exploiter
	// styles FAILED their offline bar and remain absent). Set chance 0 to turn off.
	cfg.steering.opponentStyleChance = 0.25f;
	// AttemptResolutionCondition is a STAGE-2 semantic; only attach it when termination is on
	// (and only ever together with a dedicated practice-value baseline - see post-mortems).
	if (cfg.steering.enabled && cfg.steering.resolutionTermination && !cfg.renderMode)
		g_NumPracticeArenas = (int)(cfg.numGames * cfg.steering.practiceArenaFrac);

	// Team-mode arena split, decided by the lineage-scoped phase marker (see the
	// PHASE_B_RATING_TRIGGER comment). Must be set before the Learner is built -
	// EnvCreateFunc reads these.
	g_PhaseB = std::filesystem::exists(cfg.checkpointFolder / PHASE_B_MARKER);
	g_NumGames = cfg.numGames;
	g_NumArenas2v2 = g_PhaseB ? (int)(cfg.numGames * PHASE_B_FRAC_2V2) : 0;
	g_NumArenas3v3 = g_PhaseB ? (int)(cfg.numGames * PHASE_B_FRAC_3V3) : 0;
	RG_LOG("Team curriculum: PHASE " << (g_PhaseB ? "B" : "A") << " - "
		<< (cfg.numGames - g_NumArenas2v2 - g_NumArenas3v3) << " 1v1 / "
		<< g_NumArenas2v2 << " 2v2 / " << g_NumArenas3v3 << " 3v3 arenas");

	// Skill tracker eval mix: same fractions over its own small fleet, team arenas trailing,
	// at least one arena per team mode in PHASE B (16 arenas -> 11 1v1 / 3 2v2 / 2 3v3).
	// This is what puts Rating/2v2 / Rating/3v3 on wandb once the phase flips; with few
	// arenas per team mode those Elos move slower per eval than Rating/1v1 - expect them to
	// take some evals to leave their initial value.
	g_SkillNumArenas = cfg.skillTracker.numArenas;
	g_SkillArenas2v2 = g_PhaseB ? RS_MAX(1, (int)(g_SkillNumArenas * PHASE_B_FRAC_2V2)) : 0;
	g_SkillArenas3v3 = g_PhaseB ? RS_MAX(1, (int)(g_SkillNumArenas * PHASE_B_FRAC_3V3)) : 0;
	cfg.skillTracker.envCreateFn = SkillEnvCreateFunc;
	if (cfg.skillTracker.enabled)
		RG_LOG("Skill tracker eval fleet: "
			<< (g_SkillNumArenas - g_SkillArenas2v2 - g_SkillArenas3v3) << " 1v1 / "
			<< g_SkillArenas2v2 << " 2v2 / " << g_SkillArenas3v3 << " 3v3 arenas");

	// Phase 3 (frontier resets): the pool is shared between the Learner (writer, learn-prep)
	// and the practice arenas' FrontierDrillState (readers, env threads). Never in render
	// mode - the viewer's single arena would otherwise land in the 1v1 practice slice and
	// draw drill resets. Must exist before the Learner is built (EnvCreateFunc reads it).
	if (cfg.steering.enabled && !cfg.renderMode) {
		g_FrontierPool = std::make_shared<RLGC::FrontierPool>();
		cfg.steering.frontierPool = g_FrontierPool;
		g_PracticeArenaFrac = cfg.steering.practiceArenaFrac;
	}

	// Make the learner with the environment creation function and the config we just made
	Learner* learner = new Learner(EnvCreateFunc, cfg, StepCallback);

	// The automatic PHASE A -> PHASE B flip. Runs at the tail of every iteration; ratings
	// only appear in the report on iterations where the skill tracker actually evaluated,
	// so the streak counts consecutive EVALS, not iterations. On trigger: marker -> save ->
	// exit(99) -> wrapper relaunches this same binary, which now boots into PHASE B on the
	// checkpoint just saved. Render mode never sets the callback (no ratings there anyway).
	if (!cfg.renderMode) {
		learner->iterationCallback = [](Learner* learner, Report& report) {
			report["Curriculum/Team Phase"] = g_PhaseB ? 1.0f : 0.0f;
			report["Curriculum/Phase B Streak"] = (float)g_PhaseBStreak;
			if (g_PhaseB || !report.Has("Rating/1v1"))
				return;
			float rating = (float)report["Rating/1v1"];
			g_PhaseBStreak = (rating >= PHASE_B_RATING_TRIGGER) ? g_PhaseBStreak + 1 : 0;
			if (g_PhaseBStreak < PHASE_B_TRIGGER_STREAK)
				return;

			auto markerPath = learner->config.checkpointFolder / PHASE_B_MARKER;
			std::filesystem::create_directories(learner->config.checkpointFolder);
			std::ofstream(markerPath) << "engaged at ts " << learner->totalTimesteps
				<< ", Rating/1v1 " << rating << "\n";
			RG_LOG("=============================================================");
			RG_LOG("TEAM CURRICULUM: PHASE B ENGAGED - Rating/1v1 held >= "
				<< PHASE_B_RATING_TRIGGER << " for " << PHASE_B_TRIGGER_STREAK
				<< " consecutive evals (now " << rating << ", ts " << learner->totalTimesteps << ")");
			RG_LOG("Wrote " << markerPath << "; saving and exiting " << PHASE_B_RESTART_EXIT_CODE
				<< " for the wrapper to relaunch into the 2v2/3v3 mix");
			RG_LOG("=============================================================");
			learner->RequestSaveAndExit(PHASE_B_RESTART_EXIT_CODE);
		};
	}

	// Start learning!
	learner->Start();

	return EXIT_SUCCESS;
}
