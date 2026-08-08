#include <GigaLearnCPP/Learner.h>

#ifdef GGL_VIZ_RLBOT
// Viewer-only: hosts an RLBot bot as the opponent (see CMake's GGL_VIZ_RLBOT).
#include "VizRLBotServer.h"
#endif

#include <cstdlib>
#include <filesystem>
#include <fstream>

#include <RLGymCPP/Rewards/CommonRewards.h>
#include <RLGymCPP/Rewards/ZeroSumReward.h>
#include <RLGymCPP/TerminalConditions/NoTouchCondition.h>
#include <RLGymCPP/TerminalConditions/GoalScoreCondition.h>
#include <RLGymCPP/ObsBuilders/AdvancedObs.h>
#include <RLGymCPP/ObsBuilders/AdvancedObsPadded.h>
#include <RLGymCPP/StateSetters/KickoffState.h>
#include <RLGymCPP/StateSetters/RandomState.h>
#include <RLGymCPP/StateSetters/BallNearCarState.h>
#include <RLGymCPP/StateSetters/AirDrillState.h>
#include <RLGymCPP/StateSetters/AirPlayState.h>
#include <RLGymCPP/StateSetters/CombinedState.h>
#include <RLGymCPP/ActionParsers/DefaultAction.h>

using namespace GGL; // GigaLearn
using namespace RLGC; // RLGymCPP

// Discount factor, shared by the learner's GAE and every PBRS reward (they MUST match or the
// potential terms stop telescoping against GAE). If you change tickSkip, re-derive:
// half-life_s = ln2 / (-ln gamma) / (120/tickSkip).
// 5.0 (tickSkip 8, 15 Hz): 0.9969 = ~14.88s half-life (223 steps).
// 6.0 (tickSkip 1, 120 Hz, user-directed 2026-08-01): the SAME 14.88s wall-clock horizon
// re-timed to the 8x finer decision rate => gamma_new = 0.9969^(1/8) = 0.99961197 (1786 steps).
static constexpr float TRAIN_GAMMA = 0.99961197f;

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
// eval EnvSets that clone this create-func with a small numArenas (so only low indices)
// stay 1v1. The skill tracker gets its OWN create-func (SkillEnvCreateFunc)
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
// EVEN MODE SPLIT (2026-07-19, user-directed): 0.20/0.15 -> 1/3 each. The old
// split left 3v3 at 15% of arenas (~30% of rows; rows scale with players), and
// Rating/3v3 sat flat in the 250-400 band since 15.3B while 1v1 climbed - the
// exposure-starvation read. Equal ARENAS deliberately over-weights team play in
// DATA terms: row share becomes ~17% 1v1 / 33% 2v2 / 50% 3v3 (1v1 keeps ~680
// rows/step - maintenance-level, and its rating is pool-inflated ~6x anyway per
// H2_TRUNCATION.md). Watch: Rating/2v2 + /3v3 slopes (the point), Rating/1v1
// for maintenance (guard bands 200/150 absorb a plateau; a hard 1v1 slide =
// revert the fractions), SPS + GPU memory (numPlayers 3068 -> ~4094, bigger
// inference batches).
static constexpr float PHASE_B_FRAC_2V2 = 0.3333f;
static constexpr float PHASE_B_FRAC_3V3 = 0.3333f;

// AUTOMATIC PHASE-B TRIGGER ("once 1v1 performs decently", made mechanical): when
// Rating/1v1 posts PHASE_B_TRIGGER_STREAK consecutive skill-tracker evals at or above
// PHASE_B_RATING_TRIGGER, the iteration callback below writes the PHASE_B_MARKER file
// into the checkpoint folder, checkpoints, and exits with PHASE_B_RESTART_EXIT_CODE.
// run_trainer.sh treats any nonzero exit as restart -> the relaunch finds the marker and
// builds the PHASE B fleet, resuming the same checkpoint (obs width never changes).
// 1200 sits ~1 noise-band below the measured 1v1 plateau (~1250-1300 on the 3.1 lineage);
// the 3-eval streak filters single-eval noise (band is +-30-50). The marker is
// lineage-scoped: wiping/branching checkpoints_4.0 resets the curriculum with it.
// 6.2 (2026-08-05, user-directed): TEAM MODES DISABLED FOR THIS LINEAGE. The run stays 1v1
// forever — no automatic flip, and a stray marker cannot engage it either (both the trigger
// below and the marker READ at startup are guarded on this). Reasons it is off here:
//   * The actuation swap (injection -> SIL + entropy gate) is the ONE variable under test.
//     Flipping two thirds of the fleet to game modes the lineage has never played, partway
//     through, confounds it — 6.1b flipped at 41.1Ms game time, right at the boundary of the
//     band its own frequency comparison was supposed to resolve in, and made that read
//     ambiguous for good.
//   * The trigger reads Rating/1v1, which is the INFLATED pool metric (~6x overstatement,
//     measured). Gating a curriculum change on the one number this project knows not to trust
//     is how 6.1b flipped without anyone deciding to.
// Flip it back by setting this true; the trigger constant below is unchanged and still correct
// if you do.
static constexpr bool PHASE_B_ENABLED = false;
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
// 5.0: PHASE-SCHEDULED (PULSAR5.md) - 0.3 while 1v1 dominates (individual credit
// for skill formation, free-rider bounded), 0.6 from PHASE B (shared fate makes
// deferring to the better-placed teammate payoff-neutral - the trust forcing;
// steering the belief failed, research/reports/archive/TRUST_PAIR.md). Set in main() after the phase
// marker is read, before the envs (and their reward stacks) are built.
static float TEAM_SPIRIT = 0.3f;

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
// / DrillSetter, the car-proposer head, and the halving of tsPerItr to 100k. None of those were in
// the proven-good era.
//
// AMENDED 2026-07-25: `trainAgainstOldVersions` used to be on this list. It is now ON (see the
// assignment further down). Stating the override explicitly because this paragraph is the run's
// own record of what the 9uz761ua regression cost. The honest position: that flag was never
// individually convicted - it arrived bundled in the same uncommitted edit as the proposer and
// the drill bank, and the proposer is what the measurement blamed for the ~10x deceleration.
// It is on now because the QD league was removed and past selves are the only remaining
// self-play opponent source; if throughput or Elo slope regresses, this flag is the first
// thing to put back to false.

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
		// 5.0 SCAFFOLD WEIGHT (2026-07-16, user-directed "massively incentivize"):
		// 50 -> 120, near goal-scale (deliberately < Goal 150). On a fresh run the
		// drills produce accidental aerial touches from birth; this weight decides
		// how hard each accident is reinforced during the formative window. Zero-sum,
		// impulse-scaled, 0.8s cooldown - bounded and unfarmable in sum. ANNEAL LATER:
		// once aerial-touch share establishes (PULSAR5.md), step back toward 50 so the
		// mature style isn't permanently air-warped.
		{ new ZeroSumReward(new AerialTouchReward(), TEAM_SPIRIT), 120.f },

		// Pre-touch aerial approach potential: pays the jump-and-climb toward a high ball
		// immediately, refunds the whiff - the gradient that exists BEFORE the first air touch
		// ever lands. Exact PBRS: telescopes to ~0 net, cannot be farmed. NEVER gate.
		// 5.0 SCAFFOLD: 20 -> 40 (denser pre-touch climb credit for the formative
		// window; exact PBRS, same guarantees at any weight; anneal with AerialTouch)
		{ new ZeroSumReward(new AirInterceptPotentialReward(gamma), TEAM_SPIRIT), 75.f },


		// THE defensive signal (the stack's first): engine-refereed save, guarded so only
		// genuinely opponent-created shots pay. Deliberately NO paired ShotReward (see file header
		// - phantom-farmable). UNGATED - the gate's attack-oriented level is lowest exactly in the
		// own-half states where saves fire.
		{ new ZeroSumReward(new OpposedSaveReward(), TEAM_SPIRIT), 25.f },

		// TEMPO CREDIT (2026-07-16, user-directed; see CarEnergyPotentialReward for the
		// full farm-proof): total mechanical energy as exact PBRS - instant local
		// credit for momentum decisions, loop-farming impossible by telescoping,
		// climbs untaxed (PE in the sum), ball half already covered by TouchAccel.
		// ACTIVE since 2026-07-16 (user-directed, deployed in the overnight EMERGENCE
		// sequence a few hours behind RC1). REBALANCE-1: 6 -> 15 (pace unmoved at
		// weight 6 over 4B steps - emergence_check.json; farm-proof by telescoping,
		// so the weight is a pure credit-density knob).
		// 15 -> 75 (user-directed): a further 5x on the same knob. UNMEASURED ON THIS
		// LINEAGE - it rode uncommitted through the 6M and residual cold starts and was
		// only recorded 2026-07-25 during the dead-code audit. Static extrapolation from
		// REWARD_SHAPING.md's measured table puts 75 near 60% of per-step credit density,
		// which would make tempo the dominant shaping term - that is an extrapolation, not
		// a measurement. OPEN: re-measure per-term credit share on the resid lineage and
		// either justify 75 or walk it back. See research/reports/DEAD_CODE_AUDIT.md Q11.
		{ new ZeroSumReward(new CarEnergyPotentialReward(gamma), TEAM_SPIRIT), 75.f },

		// ESCALATE-1 (2026-07-16, user-directed): small time cost - urgency vs the
		// measured stall/hover pathology. 0.01 -> ~4.5 per 30s episode (15Hz steps
		// at tickSkip 8), 3% of a goal.
		// 6.0 ts1: PER-STEP term, so its weight is a per-SECOND density divided by the
		// decision rate. 0.01 / 8 = 0.00125 keeps the same ~4.5 per 30s episode at 120 Hz.
		{ new TimeCostReward(), 0.00125f },

		// TEAM PRESSURE (2026-07-19, user-directed; RLGym-PPO-guide item): someone
		// must be on the ball. -0.15/step (~ -2.25/s; a 5s collective lapse costs
		// ~7% of a goal) while no alive teammate is near the ball or closing on it
		// (see TeamPressureReward - shadow defense counts as pressure by
		// construction, so this bites only on genuine collective disengagement).
		// Zero-sum: charged relative to the opponent's own pressure state; mutual
		// passivity cancels (accepted - TimeCost still taxes it). Watch
		// Rewards/TeamPressureReward and the next aerial-census conversion read.
		// 6.0 ts1: PER-STEP term, /8 for the 120 Hz rate — 0.15 -> 0.01875 preserves the
		// documented -2.25/s pressure density (a 5s collective lapse still costs ~7% of a goal).
		{ new ZeroSumReward(new TeamPressureReward(), TEAM_SPIRIT), 0.01875f },

		// KICKOFF RACE (2026-07-20, user-directed: net losses come from conceded
		// kickoff goals). Zero-sum, time-decayed first-touch reward, GATED on the
		// kickoff being contested (see KickoffRaceReward) so an opponent's delay
		// kickoff can't farm us into committing. Fires once per kickoff episode
		// (~the KickoffState reset slice), inert otherwise. Weight 25 (~1/6 of a
		// goal for a fast contested win) - meaningful on kickoffs, small in the
		// stack average since it only fires on kickoff resets.
		{ new ZeroSumReward(new KickoffRaceReward(), TEAM_SPIRIT), 60.f },

		// The objective. Scorer +150 / conceder -150, exactly zero-sum.
		{ new GoalReward(), 150 }
	};
}

// Steered-practice arena split (set in main() from cfg.steering before the Learner is built;
// the Learner applies the SAME first-N rule for row tagging + collection steering, so these
// two sites agree by construction). 0 = feature off = every arena is a normal match arena.
static int g_NumPracticeArenas = 0;


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


// Render mode only (GGL_RENDER_TEAM_SIZE, set in main()): the viewer's single arena plays
// this team size regardless of the curriculum split. 0 = not in render mode. The padded obs
// makes any size checkpoint-compatible, so the viewer can watch 1v1/2v2/3v3 off the same run.
static int g_RenderTeamSize = 0;

// Optimistic-Critic Ladder: impossible-control drill family (LADDER.md 3.2) on the
// LAST N arenas of the 1v1 block - the SAME rule the Learner uses for row tagging
// and the injection mask, so the two sites agree by construction. Trailing indices
// keep the low-index eval clones clear.
static int g_NumImpossibleArenas = 0;
static bool IsImpossibleArena(int index) {
	if (g_NumImpossibleArenas <= 0)
		return false;
	int n1 = g_NumGames - g_NumArenas2v2 - g_NumArenas3v3; // 1v1 block = [0, n1)
	return index >= n1 - g_NumImpossibleArenas && index < n1;
}

// Shared env body for the training and skill-eval create-funcs: everything except the
// team size and the practice-terminal decision is identical (and the skill tracker
// overwrites eval rewards/state setters/terminal conditions anyway — for eval arenas only
// the arena's team size, the obs builder and the action parser matter).
static EnvCreateResult MakeEnv(int playersPerTeam, bool practiceArena) {
	// NoTouch 10 -> 20 (2026-07-16, user hypothesis): the timeout truncation was
	// cutting dead-play episodes before RECOVERY could ever be experienced - the
	// re-enter-play trajectory never existed in the data (same shape as the takeoff
	// finding: states past the cut can't be learned). 20s doubles the recovery
	// window and lets TimeCost's idle penalty accumulate into a real signal, while
	// keeping the anti-freeze bound and the reset/curriculum throughput.
	std::vector<TerminalCondition*> terminalConditions = {
		new NoTouchCondition(20),
		new GoalScoreCondition()
	};

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
	// Slot shuffling OFF in the viewer: it draws from a clock-seeded engine, so with it on
	// the same game state yields a different obs (and can yield a different action) every
	// build — which makes the control panel's rewind unable to replay a play. Training
	// keeps it on; the policy is trained slot-invariant, so the viewer loses nothing.
	result.obsBuilder = new AdvancedObsPadded(MAX_PLAYERS_PER_TEAM, /*shuffleSlots=*/g_RenderTeamSize == 0);
	// The proven cf993b7 reset mix - effective near-ball share 0.55 (0.35 ground + 0.20 aerial
	// drill), no drill-replay slice (that came with the drill bank in the regression).
	// Aerial drill, REVERSE CURRICULUM: classically the car spawns ALREADY AIRBORNE and
	// climbing at an overhead ball, boost-fed, so it only has to COMPLETE the touch
	// (replaced the old ground-ring drill that let the bot wait the ball down - 0.1%
	// aerial touches after 17B steps). ALTITUDE ANNEALING (2026-07-15, AERIAL_GAP.md):
	// that curriculum stopped one stage early - at 30.3B the bot completes from mid-air
	// (44%, touches above goal height) but from the GROUND converts 0/300 (jumps 98%,
	// reaches z>500 only 9%): the jump->boost-climb transition was never in the training
	// (D=0) toward grounded takeoff (D=1), gated on the live aerial-conversion EMA.
	AirDrillState* airDrill = new AirDrillState();
	result.stateSetter = new CombinedState({
		// Ground touch bootstrap - the proven anti-freeze state
		{ new BallNearCarState(600, 900), 0.30f },
		// Aerial takeoff/climb-to-touch drill (the completion rung)
		{ airDrill, 0.20f },
		// ADVANCED-AIR SEEDING (2026-07-20, user-directed): flip-reset-ready +
		// air-carry setups so the ZERO-RATE mechanics get VISITED - reward alone
		// can't shape unentered states. Strong 0.15 slice (biggest single air-
		// exposure lever this run has had); paired with FlipResetReward. Carved
		// from RandomState (0.30->0.25) and Kickoff (0.15->0.10).
		{ new AirPlayState(), 0.15f },
		{ new KickoffState(), 0.10f },
		// Sole source of chaotic/defensive/air-recovery states (bounds widened to corners/goal lines)
		{ new RandomState(true, true, false), 0.25f },
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
	// Render viewer: one arena, team size picked by GGL_RENDER_TEAM_SIZE (never a
	// practice arena). MATCH-LIKE VIZ (2026-07-16, user request): kickoff-only
	// resets and goal-only terminals - no drill-mix spawns, no NoTouch cutoff - so
	// the viewer shows continuous real games incl. dead-play recovery, instead of
	// the training reset mix truncating every stall.
	if (g_RenderTeamSize > 0) {
		EnvCreateResult r = MakeEnv(g_RenderTeamSize, false);
		delete r.stateSetter;
		r.stateSetter = new KickoffState();
		for (TerminalCondition* tc : r.terminalConditions)
			delete tc;
		r.terminalConditions = { new GoalScoreCondition() };
		return r;
	}
	int playersPerTeam = 1;
	if (g_NumGames > 0 && index >= g_NumGames - g_NumArenas3v3)
		playersPerTeam = 3;
	else if (g_NumGames > 0 && index >= g_NumGames - g_NumArenas3v3 - g_NumArenas2v2)
		playersPerTeam = 2;
	EnvCreateResult result = MakeEnv(playersPerTeam, index < g_NumPracticeArenas);
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

	// 6.0 (2026-08-01, USER-DIRECTED): tickSkip 1, actionDelay 0 => a decision on EVERY
	// physics tick, 120 Hz. This is an 8x finer decision rate than 5.0's ts8/15 Hz.
	//
	// History this overrides: 4.0 ran ts4 + actionDelay 3 and the 4.0 record read that as
	// LEARNABILITY-bound, not control-bound (wavedash stuck at 11% for billions of steps,
	// per-instance SNR ~0.18), which is why 5.0 went the other way to ts8. ts1 is the
	// opposite bet: maximum control resolution, accepting ~8x more decisions per game-second
	// (so ~8x less game time per timestep, and inference — already 2/3 of collection at ts8 —
	// becomes the throughput wall).
	//
	// actionDelay stays 0. Today's sim-to-real probe fit real match data at lags 0/1/2/3 with
	// median position error 0.679 / 0.708 / 0.851 / 0.916 uu: lag 0 fits best, because the
	// game's 2-tick send->applied delay is cancelled by the ~2-tick staleness of the packet
	// the policy acts on. See research/reports/SIM2REAL_AUDIT.md S4.
	//
	// EVERYTHING RATE-DERIVED WAS RE-DERIVED WITH THIS (each is marked "6.0 ts1"):
	//   TRAIN_GAMMA 0.9969 -> 0.99961197      gaeLambda 0.95 -> 0.993767
	//   goalCritic.gamma 0.9994 -> 0.99992498  TimeCost / TeamPressure  /8 (per-step terms)
	//   reachability HER + gate windows  x8 (they are real-TIME windows expressed in steps)
	// NOT rescaled, deliberately: PBRS terms (telescoping makes them rate-invariant), event
	// terms (Goal/Demo/TouchAccel/AerialTouch/OpposedSave/KickoffRace/PickupBoost — they fire
	// on events, not per step), and NoTouchCondition (already in SECONDS, uses deltaTime).
	cfg.tickSkip = 1;
	cfg.actionDelay = 0;

	// 6.0 ts1: 1024 -> 128. This is NOT a throughput cut — it is the tickSkip change applied to
	// the fleet. Each arena now yields 8x more decisions per game-second, so 1024 arenas at ts1
	// produce the timestep rate that 8192 would have at ts8; the fleet was oversized by exactly
	// the rate change.
	//
	// It is also the fix for the OOM that killed three 6.0 launches, which no amount of model
	// shrinking touched. Episodes are appended WHOLE to combinedTraj at finalize, and the
	// collection loop's exit test is `combinedTrajNext.Length() < tsPerItr` — a test on
	// FINALIZED rows only. A newborn policy never touches the ball, so every episode runs the
	// full NoTouchCondition(20s) and, because all arenas reset together, every player finalizes
	// on the SAME step. That clump is numPlayers x episodeSteps, independent of tsPerItr:
	//     ts8:  2048 players x  300 steps =  0.6M rows  (3x over tsPerItr - survivable)
	//     ts1:  2048 players x 2400 steps =  4.9M rows  (24x over - OOM)
	// Measured on the third launch: the failing torch::cat asked for 11.60 GiB =
	// 3.47M rows x 896 (trunk width) x 4B, i.e. 1,697 steps/player = 14.1s, just under the cap.
	// 128 arenas cuts the clump 8x back to ts8's proportions. It decays on its own once the
	// policy starts touching the ball, but the run has to survive birth first.
	//
	// 128 -> 256 -> 512 (2026-08-01, each step measured). The birth clump DECAYS: by ~100M
	// steps the policy touches the ball, episodes end on goals/touches instead of all hitting
	// the 20s NoTouch cap in lockstep, and 128->256 cost only +160 MiB rather than the +1.6 GB
	// the birth arithmetic predicts. So the clump bounds the fleet only at step 0; a run that
	// RESUMES from a checkpoint never re-pays it.
	//
	// 512 is a THROUGHPUT change, and the target is inference. Measured at 256: collection
	// 3.9s vs consumption 2.9s, so with pipelinedCollection the iteration is collection-bound,
	// and inference is 2.08s of that 3.9s. Each forward batches only numPlayers rows (512 at
	// 256 arenas) which badly underuses the GPU. Doubling the fleet HALVES the number of
	// forward passes needed to fill tsPerItr while doubling each batch - strictly better GPU
	// efficiency for identical total env work. Env/Prep/Record are CPU-side and roughly
	// constant per timestep, so they do not benefit (the box is already at load ~26 on 24 cores).
	// 512 REVERTED -> 128 (2026-08-01, measured). 512 was a REGRESSION and the metric that
	// said otherwise is a trap: `Overall Steps/Second` and `Total Timesteps` count
	// collectSteps (simulated player-steps), but the LEARN pass consumes combinedTraj, which
	// the collection loop caps at tsPerItr — `Headroom/Rows N` reads ~200k at every fleet size.
	// So learning per iteration is CONSTANT and the only thing that matters is wall-clock per
	// iteration. Measured: 512 -> collection 4.3-5.2s (SPS 185k), 256 -> 3.2-3.9s (SPS 91k).
	// The bigger fleet bought a 2x higher SPS counter and a 25% SLOWER iteration.
	// With pipelinedCollection the iteration floor is consumption (~2.8s), so the optimum is
	// the smallest fleet whose collection still fills tsPerItr in ~2.8s.
	//
	// MEASURED SWEEP (2026-08-02, collection time per iteration, consumption ~2.7-2.8s):
	//     128 arenas -> 5.4-8.4s   (inference 3.1-5.5s)
	//     256 arenas -> 3.2-3.9s   (inference ~1.8-2.1s)   <-- optimum
	//     512 arenas -> 4.3-5.2s   (inference 2.3-2.5s)
	// It is a real optimum, not monotone. Below 256 the per-forward batch (2 x numGames rows)
	// is too small to use the GPU and inference latency dominates; above 256 the whole-episode
	// collection overshoot makes the loop simulate far more player-steps than the 200k it
	// needs. Do not "tune" this by reading Overall Steps/Second - that counter tracks
	// simulated steps and rises with fleet size even as the iteration gets slower.
	// 6.1 (2026-08-03): 128 FOR BIRTH ONLY. A fresh cold start pays the full birth clump again —
	// at the 6.1 widths the full-batch FP32 trunk cat is clump_rows x 1280 x 4B, which at 256
	// arenas (512 players x ~2400 NoTouch-capped steps = 1.23M rows) is a ~6.3 GB single
	// transient on top of everything else. 128 arenas halves it to the proportions 6.0's birth
	// measurably survived — and it worked: birth iteration was exactly 614,656 rows, no OOM,
	// 9.6 GiB peak. NOTE the clump outlives "first touch" by a lot: at 100M steps iterations
	// still batched ~580k rows (arenas that reset together stay near-synchronized until touches
	// desync their episode lengths); deltas only settled into the ~100-320k band by ~300M.
	// FLIPPED to 256 at 300M (2026-08-04, measured deltas, not the 100M guess): 256 is the
	// measured throughput optimum, and a resumed run never re-pays the clump. If this lineage
	// ever cold-starts again, set 128 first, and flip on MEASURED iteration deltas ~<350k,
	// not on a step count.
	// 6.1b (2026-08-04): back to 128 — the entropy-collapse restart IS a fresh cold start, so
	// the birth clump is re-paid, per the rule just above. Flip to 256 on measured deltas.
	// FLIPPED to 256 at 794M: 12 consecutive iteration deltas < 350k (the measured decay
	// condition). Note the healthy-entropy policy took ~2.6x LONGER to desync its arenas than
	// collapsed 6.1 did (794M vs ~300M) — exploration keeps more arenas on the NoTouch cap in
	// lockstep for longer, so the flip point moves with entropy, another reason it must be
	// measured rather than scheduled.
	// BACK TO 128 at 1.92B (2026-08-04 13:13 OOM, systematic): the birth clump has a GROWN-UP
	// SIBLING with no cap. Episodes end only on goals or a 20s touch drought, so once the
	// policy rallies competently episode length is UNBOUNDED, and at ts1 every rally-second is
	// 120 rows/player. Batches of 1.0-1.3M rows (5-6.5 GiB fp32 trunk cats) occurred 121 times
	// in the 256-arena log, fattening as skill grew, until one 4.85 GiB alloc failed. 128
	// halves the coincidence size at TODAY'S episode lengths (~2000 steps avg) but does NOT
	// bound it — lengths grow with skill. The durable fix is chunking the Learner's full-batch
	// FP32 trunk cat (semantics-free, same class as the existing chunked forwards); an episode
	// timeout bounds it only weakly (batch ~ players x episode_len, and a cap loose enough to
	// spare real rallies barely lowers the worst case).
	// BACK TO 256 (2026-08-04 evening, USER DECISION): "the time lost to the occasional crash
	// is worth the extra sps, and we handle the crashes gracefully anyways." Measured at 128
	// the iteration was collection-bound (5.7-14s vs consumption ~4.8s) — the fleet, not the
	// minibatch, is the SPS lever.
	//
	// 512 (2026-08-04, "get me more sps"), and the OOM reasoning above is now KNOWN WRONG.
	// Profiling at 256 put INFERENCE at 3.86s of the 5.13s collection (env step: 0.32s) — the
	// loop is bound by forward-pass OVERHEAD at a small batch, not by physics. Each forward
	// batches 2 x numGames rows, so doubling the fleet halves the NUMBER of forwards while
	// doubling each one: strictly better on a latency-bound GPU, for identical env work.
	// Two corrections that license this:
	//   * The mega-batch OOM does not live in the training path at all — it was the every-16-
	//     iterations Reach diagnostic, now row-capped in Learner.cpp. Fleet size never drove it.
	//   * MORE arenas make the buffer SMALLER, not bigger: measured Headroom/Rows N is a tight
	//     202k at 256 but spiked to 857k at 128, because more players finalize in finer lumps.
	//     The old "512 is a regression" note below was measured at 6.0's BIRTH, when every
	//     episode ran the full NoTouch cap in lockstep — a regime a competent policy has left.
	// MEASURED AND REVERTED to 256 (2026-08-04, same evening). 512 made collection WORSE:
	// 6.7-9.0s vs 256's 5.13s, with inference 4.6-6.7s vs 3.86s — so the "halve the forward
	// count" reasoning above is WRONG, and the 6.0-era verdict was right for a better reason
	// than it gave. Inference is NOT latency-bound at 512 rows/forward: doubling the batch
	// bought nothing and the extra 512 arenas' CPU-side work (obs build, action parse, record)
	// contends with the env thread pool on an already-saturated 24-core box. Rows N stays ~200k
	// at every fleet size, so the bigger fleet adds cost without adding learning.
	// 256 is a genuine optimum, now confirmed twice on two different net sizes. The remaining
	// collection cost is inference itself, which is the standing price of ts1 (120 Hz = 8x the
	// forwards of ts8) and needs a code-level fix, not a knob.
	cfg.numGames = 256;

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
	// 2026-07-22 (6M cold start): the 512->1152 width bump ~2.25x'd the per-row saved activations
	// across the trunk + all three dense heads, so the learn-pass autograd peak scales the same.
	// Cut 50k -> 20k to hold that peak roughly constant (gradient accumulation => mathematically
	// identical update, 10 passes instead of 4). Raise back toward 25-30k only if the card shows
	// headroom in practice (no desktop-graphics contention). The reachability InfoNCE subsample
	// (512/minibatch) now runs 10x/epoch - same benign side effect as the earlier 200k->50k cuts.
	// 2026-07-28: the theory-from-parts merge added the rhat1/rhat2 twins, which mirror the
	// critic's config like the vdag twins do - so the 5x1280 critic-family head count went 3 -> 5
	// and it is that count, times the per-row saved activations, that dominates the learn peak
	// (measured: 11.91GB of PyTorch tensors, OOM on a 20000x1280 head forward = the 98MiB alloc
	// in the crash, on a card with ~13.2GB free after desktop graphics). That path is now
	// DISABLED (PPOLearnerConfig::vdagTheoryEnabled), putting the head count back at 3.
	// Held at 10k rather than restored to 20k: the overnight 2026-07-28 run (theory ON, 10k)
	// trained fine for ~5h but sat at 12.0-12.7GB with ~240MB free and took 8 memory-pressure
	// crash-restarts in 8h as desktop graphics moved. Dropping to 3 heads at this same 10k is
	// what buys the headroom back; 20k would spend it again.
	// ^ That paragraph is stale: the value has been 20k since (committed), so the run has in fact
	// been spending that headroom. Recorded rather than deleted because the crash-restart count is
	// the only measured datapoint on this card's real margin.
	// 2026-07-29 (critic trunk): the value heads were factored behind a shared critic_trunk, so
	// per-row saved activations fall to ~57% of the previous config. Counting 1280-equivalent
	// width x layers per row, ignoring the LN/act intermediates that scale everything alike:
	//     before  trunk 3456 x2 forwards + critic 6400 + vdag 12800 + goal 6400 + policy 2304
	//             ~= 34.8k
	//     after   trunk 3456 x1 + critic_trunk 3840 + 4 heads x 2560 + policy 2304 ~= 19.8k
	// So the ROW BUDGET at a constant peak rises ~1.75x: 20k -> ~35k. Spending only part of that
	// (20k -> 25k, ~0.71x the previous peak) deliberately banks the rest as OOM margin, because
	// this run has NO automatic memory guard and the measured history above is 8 crash-restarts in
	// 8h at ~240MB free. 40k is the next rung and is ~1.14x the previous peak - i.e. RISKIER than
	// what was already crash-looping; only go there after watching actual GPU memory across a
	// learn pass with the viz running.
	// miniBatchSize MUST DIVIDE batchSize (200k) - PPOLearner's ctor hard-fails otherwise, so the
	// tempting round numbers 32k/30k are not available. Legal rungs from here: 25k, 40k, 50k.
	// DO NOT put this back to 100k: that value belongs to the 512-wide era two architecture
	// changes ago (see the 200k->100k->50k->20k ladder above, each step forced by a capacity
	// increase). At this width 100k is ~10x the per-minibatch activation peak of a config that
	// already measured ~12GB of 16.3GB - it OOMs on the first learn pass.
	// 6.0 (2026-08-01): 25k -> 12.5k for the ~8 GB VRAM budget, then BACK to 25k once the
	// card stopped being shared. 200k/25k = 8 accumulation chunks (an exact divisor, same
	// constraint that rules out 32k/30k above). Gradient accumulation makes minibatch size
	// MATHEMATICALLY IDENTICAL for the update — it trades activation peak against GEMM
	// efficiency — so this is a pure throughput knob. It targets PPO Learn (2.28s of the
	// 2.9s consumption half). Give it back first (12.5k) if VRAM bites.
	// 6.1 (2026-08-04): 25k -> 12.5k, the "give it back first if VRAM bites" lever above,
	// pulled on evidence: at the 6.1 widths + 256 arenas the steady watermark reached
	// ~12.7 GiB and the run took a real OOM at 1.0B (148 MiB request, 297 MiB free) — the
	// same watermark regime that produced 8 crash-restarts in 8h on 2026-07-28. 200k/12.5k
	// = 16 accumulation chunks (exact divisor). Identical update math; costs only GEMM
	// efficiency in PPO Learn. Do NOT raise back toward 25k without watching a full learn
	// pass's actual peak with the desktop loaded.
	// 6.1b (2026-08-04): 12.5k -> 25k, now that it actually buys something. Measured at 256
	// arenas: consumption 4.01s of which PPO Learn is 3.35s, against collection 5.13s. Under
	// pipelinedCollection the iteration is max(collection, consumption), so cutting ONLY
	// collection (the 512-arena change above) just moves the wall to consumption — both halves
	// have to come down together, which is why the same bump measured worthless at 128 arenas
	// and is worth doing here. Gradient accumulation keeps the update mathematically identical;
	// this is purely activation-peak traded for GEMM efficiency. 200k/25k = 8 exact chunks.
	// The learn-pass peak this re-spends is affordable because the Reach diagnostic — the
	// actual site of every mega-batch OOM — is now row-capped.
	// 6.2 (2026-08-07): 25k -> 12.5k. Previously this was held at 25k because the OOMs cost
	// ~1% of throughput and were judged acceptable. That judgement was correct at the time and
	// is now void: on 2026-08-07 the run took THREE checkpoint-corruption incidents that
	// destroyed 32 checkpoints and every golden-archive entry, and the loader eventually
	// exhausted all of them and started a fresh model on an 11.6B-step lineage. Only an
	// out-of-band manual backup survived.
	// The link to memory pressure is CORRELATIONAL, not proven: each corruption window ended in
	// an OOM, onset was ~40 min before it, and the run sat at ~13.3 GiB of 15.5 with ~250 MiB
	// free when it died. Allocator exhaustion is a plausible way to perturb the device-to-host
	// copies a save is made of, and it is the only recurring stressor. Halving the minibatch is
	// the one lever that touches it, is mathematically identical under gradient accumulation
	// (200k/12.5k = 16 exact chunks), and costs only GEMM efficiency.
	// Against losing 1.2B steps and the entire recovery chain, that trade is no longer close.
	cfg.ppo.miniBatchSize = 12'500;

	// BF16 inference for collection + GAE value preds. rho/gate evals request fp32 explicitly and
	// grad-enabled forwards (InfoNCE training) always run fp32, so the gate is unaffected.
	cfg.ppo.useHalfPrecision = true;

	// 6.1b (2026-08-04, user-directed): bf16 autocast for the LEARN pass, the one throughput
	// lever that does not trade against the experiment (the alternatives were epochs 2->1,
	// which halves sample reuse, and shrinking the net the user asked to be full-size).
	// Profiling: GPU 92-98% busy, iteration = max(collection 5.3s, consumption 3.9s), and
	// collection is slow because the collect worker's forwards QUEUE BEHIND the learn pass —
	// so learn-pass cost sets both halves. Weights/optimizer stay fp32; see
	// PPOLearnerConfig::learnAutocastBF16 for the deliberately partial scope (geo HJB and
	// InfoNCE stay fp32) and the revert.
	cfg.ppo.learnAutocastBF16 = true;

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
		// NOTE: any override of a value assigned LATER in this function does not belong
		// here - this block runs first and would be clobbered. Re-apply it next to the
		// production assignment instead (see the Nexto serveFrac and reference-cadence
		// smoke overrides).
		RG_LOG("GGL_SMOKE: numGames 128, tsPerItr 25k, fp32, sequential (offline sandbox smoke)");
	}

	cfg.ppo.epochs = 2;
	// 6.0 ts1: 0.035 -> 0.004375 (= 0.035/8), the same /8 the other rate-derived per-step
	// quantities took. Entropy regularization is a RATE — nats per unit time, not per
	// decision — so a coefficient tuned at 15 Hz applies 8x more entropy pressure per
	// game-second at 120 Hz. Symptom that convicted it: the viewer showed cars flipping
	// continuously. At ts8 the env granted 8 ticks of free action persistence per decision;
	// ts1 removed it, so a stochastic policy re-rolls its jump/dodge bits 120x/s and
	// exploration becomes dithering rather than committed behaviour. Measured Policy Entropy
	// 0.73 on 6.0 vs 0.58-0.60 on 5.3 at the same coefficient.
	// WATCH `Policy Entropy`: target is a drift back toward ~0.60. This is the aggressive end
	// of the defensible range — if entropy instead COLLAPSES (say under ~0.35) that is
	// premature convergence, and the revert is this one number (0.035, or a middle rung
	// like 0.0175).
	// 6.1b (2026-08-04): 0.004375 -> 0.0175, the "middle rung" the paragraph above names, after
	// the collapse it warns about HAPPENED on 6.1: entropy fell 0.74 -> 0.05 by 245M steps and
	// ~6e-4 by 1.1B, a smooth exponential from birth (no event cliff). Discriminated against
	// everything else that changed: 6.0 ran THIS SAME 0.004375 (and the same advFilter, ts1,
	// birth fleet) for 3.07B steps holding 0.71-0.76 — the only relevant 6.1 diffs are the
	// full-size widths. More capacity fits the argmax faster, so the coefficient that was
	// merely "aggressive" at 896/640 is insufficient at 1280/768. The optimism injections are
	// exonerated by dose (Inj Std Ratio pinned 0.04). WATCH on 6.1b: entropy should hold a
	// 0.4-0.7 band; if instead the viewer shows the continuous-flipping dithering that forced
	// the original /8 cut, step DOWN to ~0.00875, not back to 0.004375. If it collapses under
	// ~0.35 again at 0.0175, suspect the advFilter interaction next (its own pre-registration
	// names entropy collapse; revert = TOP_SIGNED), not a further coefficient bump.
	cfg.ppo.entropyScale = 0.0175f;

	// ADVANTAGE FILTERING (2026-07-29, user-directed): the policy trains only on the top 50% of
	// rows by post-injection advantage; the threshold is a buffer-wide quantile taken after GAE
	// and after the HEADROOM/goal-critic injectors, so it selects on the number the policy loss
	// actually consumes. Value heads (critic, goal critic, V-dagger twins, reachability) keep
	// every row - only the PPO term is filtered - and the kept rows are renormalized by their
	// count, so this is a change of WHICH rows teach, not of step size. Revert = 1.0f.
	// Watch, in order: AdvFilter/Kept Positive Frac (near 1.0 = the update is pure
	// self-imitation, the clipping-ratchet asymmetry that once bled Elo while the viewer looked
	// better), Policy Entropy (a collapse means overcommitment), then the Nexto/* goal slope -
	// the only pool-inflation-proof read on whether this actually helped.
	cfg.ppo.advFilterFrac = 0.5f;
	// MAGNITUDE (2026-07-30, user-directed): rank by |advantage|, not signed advantage, so the
	// retained half is the two TAILS rather than "everything above the median". Reason: the PPO
	// per-row gradient is |A| * grad(log pi), so a median cut spends the entire budget on the
	// least informative half of the positives (rows near A ~ +0 that the critic already
	// predicted) while discarding the largest-|A| rows in the buffer. Same kept count, same
	// renormalization - only WHICH rows teach changes. This also defuses the known ratchet risk
	// above rather than merely watching it.
	// PRE-REGISTERED: Kept Positive Frac must fall off ~1.0 (confirms the switch took effect) and
	// Dropped Abs Adv Mean must fall relative to its TOP_SIGNED value (confirms the dropped rows
	// really are the low-information ones). Success = Nexto/* goal slope at or above its current
	// trend over the next ~1B steps; Policy Entropy must not collapse. Revert = TOP_SIGNED.
	cfg.ppo.advFilterMode = AdvFilterMode::MAGNITUDE;
	// 6.1b (2026-08-04, user-directed): make the filter a real ROW SUBSET of the learn pass
	// rather than a mask on the policy loss. Until now it selected rows and then ran the full
	// forward+backward on all of them anyway, so it cost full compute; the policy head is only
	// ~10% of the pass, so masking it could never save anything. See
	// PPOLearnerConfig::advFilterSubset for the precedent (GigaFlow / Stratego / Generals.io all
	// filter the critic too and report 2.3-2.5x) and for the counter-evidence (VSOP, PPG).
	// HELD AT advFilterFrac 0.5, deliberately, even though the papers above use 0.20-0.25: at
	// 0.5 this flag leaves the POLICY gradient bit-identical to yesterday's run and changes ONLY
	// which rows the value heads see, so it isolates the one variable the literature disagrees
	// about. Move to 0.25 only after entropy and KL hold here.
	cfg.ppo.advFilterSubset = true;

	// Reachability: aux InfoNCE heads on the shared trunk. gateEnabled = false under FRONTIER-9:
	// every reward component is now zero-sum/antisymmetric and ungated by design (gating breaks ZS
	// symmetry - positive side muted, mirror charged in full - and PBRS telescoping), so the gate
	// would be a mathematical no-op anyway; this makes that explicit. Keep enabled=true: the heads
	// still feed the InfoNCE trunk aux + Reach/* plasticity canaries.
	cfg.ppo.reachability.enabled = true;
	cfg.ppo.reachability.gateEnabled = false;

	// 6.0 ts1: the HER goal-sampling offsets and the gate's delta windows are REAL-TIME
	// windows that happen to be expressed in STEPS — PPOLearnerConfig.h says so inline and
	// tells you to re-derive them when tickSkip changes (they already went 90->45 / 20->10
	// on the ts4->ts8 move). At 120 Hz they are 8x too short in wall-clock unless scaled.
	// Held deliberately: carStateHerMaxOffset (see the block below — it is an EMPIRICAL
	// calibration choice, not a real-time design, and its calibration is already void).
	cfg.ppo.reachability.ballHerMaxOffset = 360; // was 45  (~3.0s preserved)
	cfg.ppo.reachability.carHerMaxOffset  = 80;  // was 10  (~0.67s preserved)
	cfg.ppo.reachability.deltaWindow      = 64;  // was 8   (~0.53s preserved)
	cfg.ppo.reachability.deltaSmooth      = 32;  // was 4   (~0.27s preserved)
	cfg.ppo.reachability.touchPredHorizon = 360; // was 45  (~3.0s preserved)
	// Third goal-space head (2026-07-14): canonical CAR pos+vel - the movement-capability
	// frontier for META steering. Offline (conservative frozen-phi test): calibration
	// DECISIVELY monotone (~7x the ball head's margin; window 45 chosen by margin across
	// {20,45,90}); per-cluster causal steerability NOT yet demonstrated offline (0/3
	// clusters clean). Revert = false. Record: research/tools/carstate_head_validate.py.
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

	// The goal proposer / drill bank / car-proposer bundle was REMOVED 2026-07-25. It had been
	// explicitly disabled since the 9uz761ua regression (enabling it decelerated the best run
	// ~10x), and its config block configured a delta net that was never constructed.
	// History: git log -- GigaLearnCPP/src/private/GigaLearnCPP/PPO/Proposer.cpp

	// Wide clip (cold return-sigma is ~2-4 under this near-sparse stack; default 10 compressed the
	// first goals). 50 releases the full 150 once sigma >= 3 and bounds the tail.
	cfg.ppo.rewardClipRange = 50;

	cfg.ppo.gaeGamma = TRAIN_GAMMA; // ~14.9s half-life at 120Hz (6.0 tickSkip 1); MUST match the PBRS reward gammas above

	// 6.0 ts1: gaeLambda MUST be re-derived with gamma, or GAE silently goes myopic.
	// The TD credit window is ~1/(1 - gamma*lambda) steps. At ts8 that was
	// 1/(1 - 0.9969*0.95) = 18.9 steps = 1.26s. Leaving lambda at 0.95 under the ts1 gamma
	// gives 19.9 steps = 0.165s — a 7.6x SHORTER real-time window, i.e. the bias/variance
	// tradeoff would move drastically without anyone touching lambda.
	// Preserving the 1.26s window at 120 Hz needs 151 steps => lambda = 0.993767.
	cfg.ppo.gaeLambda = 0.993767f;

	// Secondary goal-only critic: long-horizon credit on the one unfarmable signal. Independent net,
	// raw obs in; advantages blended at beta = 25% of dense-advantage scale (std-matched, centered).
	// VALIDATION: GoalCritic/Value-Outcome Corr and Adv-Outcome Corr must be POSITIVE once goals flow;
	// negative = channel/sign bug -> set beta = 0 (critic still trains, no blend) and investigate.
	cfg.ppo.goalCritic.enabled = true;
	cfg.ppo.goalCritic.gamma = 0.99992498f; // ~77s half-life at 120Hz (6.0 tickSkip 1: 0.9994^(1/8))
	cfg.ppo.goalCritic.beta = 0.25f;
	cfg.ppo.goalCritic.lr = 1.5e-4f;
	// Private head on top of critic_trunk (2026-07-29): 5x1280 on raw obs -> 2x1280 on the shared
	// value body, 6.87M -> 3.29M. This is the change that gives up its gradient isolation - see
	// the criticTrunk block below and PPOLearnerConfig::criticTrunk.
	cfg.ppo.goalCritic.model.layerSizes = { 1536, 1536 }; // 6.1: 1024 -> 1536 (FULL SIZE block below)

	cfg.ppo.policyLR = 1.5e-4;
	cfg.ppo.criticLR = 1.5e-4;

	// ASYMMETRIC RESIDUAL NETS (2026-07-24 cold start, user-directed). Three changes at once,
	// all shape-breaking, so this needs its own checkpoint folder (see cfg.checkpointFolder):
	//
	//   (a) RESIDUALS everywhere (PartialModelConfig::addResiduals). layerSizes[0] is a plain
	//       stem, every following PAIR becomes x <- Act(x + LN(W2 Act(LN(W1 x)))). {W,W} is
	//       unchanged by the flag, {W,W,W} = stem+1 block, 5 entries = stem+2 blocks. This is
	//       the BroNet shape (Nauman et al. 2024); the local bake-off (~/Projects/experiments/
	//       arch/RESULTS.md, 5 seeds x 10 min) measured residual+LN nets at ~20x the SAMPLE
	//       efficiency of a plain LN-MLP (60 vs 3 touches at matched 0.9M steps) with the
	//       tightest seed spread of any arm. It also directly targets the effective-rank decay
	//       noted below: a residual stream is the documented fix for on-policy plasticity loss
	//       (arXiv 2405.19153), where widening only treats the symptom.
	//
	//   (b) POLICY SHRUNK 1152 -> 768 (3 entries = stem + 1 block): ~4.09M -> ~2.14M params.
	//       SimBa (2410.09754) and BRO (2405.16158) both find actor scaling contributes ~nothing
	//       while critic scaling pays - SimBa's own defaults are actor 128 / critic 512, a 4x
	//       ratio, so 768/1280 here is mild. The freed SPS funds (c).
	//
	//   (c) VALUE HEADS WIDENED AND DEEPENED to 5x1280 (stem + 2 blocks): critic ~3.98M ->
	//       ~8.03M, goal critic ~2.92M -> ~6.85M. Depth is only safe now because of (a) - the
	//       bake-off's plain 16-layer stack LOST badly to a 4-layer one (142 vs 214).
	//       *** COST WARNING: vdag1/vdag2 mirror the critic's config automatically
	//       (PPOLearner.cpp), so every critic parameter is paid THREE times: critic-family goes
	//       ~11.9M -> ~24.1M. Total dense params ~20M -> ~38M. ***
	//       ^ SUPERSEDED 2026-07-29: those numbers described four INDEPENDENT 5x1280 value stacks.
	//       They are now factored behind a shared critic_trunk (see the criticTrunk block below),
	//       critic family ~17.9M / net ~24.9M. The "paid three times" mechanism is unchanged and
	//       still the reason head depth is the expensive dimension here.
	//       Widths stay 128-aligned (768=6x128, 1280=10x128) for tensor cores. The learn-pass
	//       activation peak grows with width x depth, so watch for OOM - the 512->1152 bump
	//       already forced a miniBatchSize cut, and this one may force another.
	//
	// The trunk keeps its 1152 width but gains a third layer so it gets a residual stream too
	// ({1152,1152} would be flag-immune); it is shared perception feeding both heads, so it is
	// deliberately NOT shrunk with the policy.
	// History kept: the 256->512 bump was licensed by a saturation probe (49-74% spectral
	// utilization at 256, effective rank grinding down all run); 512->1152 followed the same
	// "underused capacity is nearly free" logic. Inference is latency-bound on the 5080, so SPS
	// cost stays sub-proportional to dense FLOPs - but measure it from the log, and if the
	// policy shrink costs strength, revert (b) alone by putting policy back to 1152.
	bool addResiduals = true;
	cfg.ppo.sharedHead.addResiduals = addResiduals;
	cfg.ppo.policy.addResiduals = addResiduals;
	cfg.ppo.critic.addResiduals = addResiduals;
	cfg.ppo.criticTrunk.addResiduals = addResiduals;
	cfg.ppo.goalCritic.model.addResiduals = addResiduals;
	cfg.ppo.reachability.phi.addResiduals = addResiduals;
	cfg.ppo.reachability.psi.addResiduals = addResiduals;


	// 6.0 VRAM BUDGET (2026-08-01, user-directed): the whole trainer must fit in ~8 GB so the
	// card can be shared with other jobs. 5.3's widths peaked at 10.33 GiB and OOM'd three
	// times at 6.0's birth. Widths scaled ~0.8x here; the accompanying miniBatchSize cut is
	// the other half of the budget (see it below). Activation peak is LINEAR in both, so
	// 0.8 x 0.5 ~= 0.4x on the dominant term. Depth and head-count are unchanged — see the
	// critic_trunk block below for why two layers per value head is a floor, not a knob.
	// 6.1 FULL SIZE (2026-08-03, user-directed "use all the VRAM"): the card is no longer
	// shared (the possibility sweep finished), so the ~8 GB budget above is retired. Widths go
	// one rung PAST 5.3's, not just back to them: trunk 1152 -> 1280, value side 1280 -> 1536
	// (the critic-scaling-pays evidence in the resid block below), policy back to its designed
	// 768 and deliberately NO further — SimBa/BRO actor-scaling says wider actors buy ~nothing,
	// so VRAM spent there is wasted. reach phi/psi and geoModel are HELD at their sizes: each
	// carries its own written warning (InfoNCE overfits when over-parameterized; geo width buys
	// little and costs learn-pass peak). All widths stay 128-aligned for tensor cores
	// (1280 = 10x128, 1536 = 12x128). Depth untouched everywhere — the two-layer head floor
	// below is load-bearing for the vdag min() anti-ratchet.
	// Estimated learn-pass activation growth vs 6.0 widths is ~1.45x per row on the dominant
	// term; 6.0 measured ~5.5 GB peak, so expect ~7-8.5 GB with ~13 GB free. The remaining
	// headroom is deliberately banked for the post-birth throughput levers (miniBatchSize
	// 25k -> 50k is the next rung, numGames 128 -> 256): those are shape-safe to move mid-run,
	// widths are not — size the net high once at birth, tune the batch after measuring.
	cfg.ppo.sharedHead.layerSizes = { 1280, 1280, 1280 }; // 6.1: 896 -> 1280; stem + 1 residual block
	cfg.ppo.policy.layerSizes = { 768, 768, 768 };        // 6.1: 640 -> 768 (designed size); stem + 1 block

	// SECOND SHARED TRUNK, VALUE-SIDE ONLY (2026-07-29, user-directed). The four value heads used
	// to be four independent 5x1280 stacks reading the main trunk (critic + vdag1 + vdag2, which
	// mirror the critic's config automatically, + a standalone raw-obs goal critic) = ~31.0M
	// params, 82% of the whole net. The first three layers are now factored out into critic_trunk
	// and each head keeps two private layers:
	//     critic_trunk {1280,1280,1280} = 4.76M     (stem + 1 block, on the 1152 main trunk)
	//     critic / vdag1 / vdag2 / goal_critic {1280,1280} = 3.29M each
	//     critic family 31.0M -> 17.9M (-42%); whole net ~38.0M -> ~24.9M (-34%)
	// The point is NOT the parameter count - it is the learn-pass activation peak, which is what
	// actually caps miniBatchSize: saved 1280-wide layer activations per row go 20 -> 11, and the
	// main trunk is now materialized ONCE for all value heads instead of twice (PPOLearner::Learn's
	// fnValueTrunk). That is what funds the miniBatchSize bump above.
	// TWO LAYERS PER HEAD IS A FLOOR, NOT A TUNING CHOICE: vdag1/vdag2 are only useful as the
	// min() anti-ratchet if they are genuinely different functions. Give them one private layer
	// each on a shared body and they collapse toward the same function, disarming the guard that
	// COMPOSITION_CRITIC.md section 4.4 measured (H inflating 0.3 -> 11.8 with nothing behind it).
	// If params must come down further, take them from critic_trunk's depth, not the heads'.
	// 6.0: 1280 -> 1024 on the whole value side (the VRAM budget above). Depth is untouched,
	// which is what the floor rule below demands; only width moved. vdag1/vdag2 follow the
	// critic's config automatically, so this one line resizes three of the four heads.
	// 6.1: 1024 -> 1536 (the FULL SIZE block above) — past 5.3's 1280, same depth rules.
	cfg.ppo.criticTrunk.layerSizes = { 1536, 1536, 1536 };
	cfg.ppo.criticTrunk.addOutputLayer = false;  // it is a body; MakeModels asserts this
	cfg.ppo.critic.layerSizes = { 1536, 1536 };  // private head on top of critic_trunk
	// Reachability phi/psi are CONTRASTIVE/regression heads, not value
	// heads: the critic-scaling evidence above does not cover them, and over-parameterized
	// InfoNCE embeddings can overfit the contrastive task. Grown only modestly (256x2 ->
	// 384x3 = stem + 1 block); if reach accuracy or drill quality regresses, revert these two
	// lines first - they are the least-supported part of this change.
	cfg.ppo.reachability.phi.layerSizes = { 384, 384, 384 };
	cfg.ppo.reachability.psi.layerSizes = { 384, 384, 384 };
	cfg.ppo.reachability.lr = 3e-4f;

	// Speed knob kept from the post-good-era "speed 2" commit (2211cce): larger rho-read chunks
	// fill the GPU better. This only changes CHUNKING of the gate's rho reads, never their values,
	// so it's a pure throughput win with zero behavioral effect on the gate.
	// 16384 -> 4096 (2026-08-01): this is the trainer's largest transient allocation and it was
	// the exact site of three CUDA OOM crashes at 6.0's birth. The failing malloc was 72.00 MiB =
	// 16384 x 1152 x 4B — one LayerNorm activation of the chunked shared-trunk forward
	// (Learner.cpp, the "computed ONCE and reused" FP32 trunk pass), where 1152 is
	// sharedHead's width. The trunk is stem + 1 residual block, so several such tensors are
	// live per chunk; quartering the chunk quarters that whole transient (~0.4 GB of relief)
	// and is BEHAVIOR-NEUTRAL by the same argument as the comment above — chunking a forward
	// pass cannot change its result. Cost is a modest throughput loss from smaller GEMMs.
	// Raise it back once the card is not shared with other jobs.
	// Back to 16384 (2026-08-01): 4096 was an anti-OOM measure during the 6.0 birth crisis and
	// it bought only ~0.1 GiB of peak — the real cause was the birth episode clump, not this
	// chunk. With the card no longer shared, restore the throughput default.
	cfg.ppo.reachability.scoreChunkSize = 16384;

	// Muon for the dense nets (RMS-matched, Adam LRs transfer). Reachability heads stay Adam:
	// contrastive InfoNCE embeddings train poorly under orthogonalized updates.
	auto optim = ModelOptimType::MUON;
	cfg.ppo.policy.optimType = optim;
	cfg.ppo.critic.optimType = optim;
	cfg.ppo.criticTrunk.optimType = optim;
	cfg.ppo.sharedHead.optimType = optim;
	// The goal critic was the ONLY dense net still on Adam, and not by decision: its
	// activationType / addLayerNorm / addResiduals were all set explicitly on the lines around
	// here while optimType was never assigned, so it fell through to PartialModelConfig's ADAM
	// default. Muon now, like every other dense net (2026-07-29). The vdag twins get it for free
	// by mirroring the critic's config. NOTE this is a real change, not a cleanup: the LRs here
	// were chosen as Adam LRs that transfer to Muon RMS-matched, so it lands with the cold start.
	cfg.ppo.goalCritic.model.optimType = optim;

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
	cfg.ppo.criticTrunk.addLayerNorm = addLayerNorm;
	cfg.ppo.sharedHead.addLayerNorm = addLayerNorm;
	cfg.ppo.reachability.phi.addLayerNorm = addLayerNorm;
	cfg.ppo.reachability.psi.addLayerNorm = addLayerNorm;
	cfg.ppo.goalCritic.model.addLayerNorm = addLayerNorm;

	// GEOMETRY (4th rung) — RETIRED 2026-08-08 (audit; research/reports/EPSILON_CRITIC.md).
	// The eps-critic verdict was "geo trio + Phi-mix retire; everything rides the hull
	// critic", but only the injection half ever shipped (geoSeekBeta 0 below): the trio kept
	// training three nets + an HJB input-gradient per minibatch, and the SIL gate kept riding
	// it (hMix = 0.5 geo + 0.5 vdag). This flag completes the retirement: no geo nets are
	// built, the SIL gate degrades to unit(tH) alone (documented at its Learner.cpp site),
	// and the donor bank + hull chart are UNAFFECTED (feed and training gate on hullEnabled
	// independently). Known cost, accepted: geo's measured edge was the 0-3M cold-start
	// window (4.7x headroom signal over V-dagger when the policy is worst) — but the hull's
	// whole claim is pricing what the policy can't yet do; giving it a crutch there would
	// unfalsify exactly the window it must win. Revert is this flag (geoGamma/geoModel below
	// stay configured, inert while off).
	cfg.ppo.geoEnabled = false;
	// 6.1 (2026-08-03): the HJB residual's gamma is DECOUPLED from gaeGamma and pinned to the
	// value the rung was validated at (ts8's 0.9969). At ts1's gaeGamma the residual is
	// ill-conditioned — the level-anchoring (1-gamma) path fell 8x relative to the
	// gradient-shaping path, and 6.0 measured Geo/V Mean 2.9 against the ~39 dimensional
	// analysis predicts, i.e. an UNSOLVED field silently supplying half the seek potential
	// (geoMixW 0.5) while every panel looked healthy (the injector discards the level).
	// Full mechanism + why a residual rescale would NOT fix it (Adam is scale-invariant):
	// PPOLearnerConfig::geoGamma. WATCH Geo/V Mean: at this gamma the fixed point is ~1/8 of
	// rate-invariant, so healthy is a climb toward ~5, not ~39.
	cfg.ppo.geoGamma = 0.9969f;
	cfg.ppo.geoModel.layerSizes = { 384, 384 };
	cfg.ppo.geoModel.activationType = activation;
	cfg.ppo.geoModel.addLayerNorm = addLayerNorm;
	cfg.ppo.geoModel.addResiduals = false;   // 2 layers: addResiduals is a no-op at this depth
	// The implicit-world-model potential arrived by merge from a parallel experiment line
	// (e1b02b9) with enabled-by-default config. EXPLICITLY OFF: adding a second
	// theorised-value potential mid-lineage would confound the at-scale test.
	cfg.ppo.vdagWmEnabled = false;

	// ===== ACTUATION SWAP (2026-08-05, research/testbeds/inject2d + RESULTS.md) =====
	// The potential-injection channel is REPLACED by headroom-gated self-imitation + the
	// H-gated entropy multiplier. Measured on the offline injection testbed (8 seeds/arm,
	// like-for-like ladder): the sigma-matched potential injection ignites conducts FAST
	// and then CAPS them at ~1/3 of the un-injected baseline's converged rate (the
	// per-batch unit normalization means the delivered dose never retires -- as headroom
	// closes, residual field noise is re-amplified to constant dose forever); SIL+entgate
	// ignite 8/8 seeds AND converge highest, and running SIL ON TOP of the live injection
	// inherits the injection's cap (1.45 vs 5.15 final conduct rate) -- so this is a swap,
	// not an addition. At ts1 specifically the per-decision potential difference lost ~8x
	// SNR vs ts8 while SIL (return-level) and entgate (objective-level) are decision-rate
	// invariant. All four rungs keep TRAINING exactly as before (measurement, gates, and
	// representation pressure through the trunk); only the advantage-channel dose is zeroed.
	// Revert: geoSeekBeta back to 0.04, silEnabled false, vdagEntGateEnabled false.
	cfg.ppo.geoSeekBeta = 0.0f;   // delivered injection = exactly 0; dose panel confirms
	cfg.ppo.vdagSeekBeta = 0.0f;  // the pre-geo vdag-only injection too (it is overwritten
	                              // while geoEnabled, but a later geo toggle must not
	                              // silently resurrect it)
	cfg.ppo.silEnabled = true;
	cfg.ppo.silCoeff = 0.05f;     // toy-validated 0.1, halved for the opposed live game
	                              // (imitated overcommits are the un-derisked hazard --
	                              // the steering-v1 ratchet; watch Ref shares + SIL/*)

	// ===== HULL OPERATOR (2026-08-07; canonical: research/reports/EPSILON_CRITIC.md s7) =====
	// The record-licensed relaxed Bellman operator on the V-dagger targets: the bootstrap is
	// maxed over the real next state plus hullK candidates built by transplanting eps-scaled
	// WITNESSED displacement vectors between chart-matched states. This is what prices
	// never-assembled conducts (toy family F, 0.000% assembly in every record: best of all
	// estimators, 8/8 seeds) and unvisited-state value (family B: +0.13 rho over plain
	// V-dagger). OPEN configuration deliberately (no donor gates): the validated TRAINING
	// seat -- toy ignition 1.02M vs 1.22M, 8/8 -- where thin-record optimism is exploration
	// pressure bounded by SIL's realized-conversion requirement, not hallucination (the seat
	// theorem, ibid). Adversarial ledger in the report: resource-laundering, teleports,
	// illegal transplants, thin-record -- every harm channel measured and closed.
	// Deployment notes: takes effect at next restart; OLD checkpoints fresh-init the two
	// chart nets (Model::Load allowNotExist) -- no rotation risk; changing V-dagger's
	// targets mid-lineage is a LEVER (treat as a new deployment: watch Hull/Uplift Mean
	// [healthy = small and shrinking per-regime, not monotonically growing], Hull/Chart NLL
	// [down then flat], Headroom/Vdag Mean [no ratchet], SIL/* and Ref shares as always).
	// Revert: this flag, restart. The feed-side teleport filter this change adds also
	// removes the long-flagged respawn pollution from the geo Sigma reservoir.
	cfg.ppo.hullEnabled = true;
	cfg.ppo.hullHeadModel.layerSizes = { 64 };
	cfg.ppo.hullHeadModel.activationType = activation;
	cfg.ppo.hullHeadModel.addLayerNorm = false;

	// ===== COMPOSITE VALUE CRITIC (2026-08-07; inject2d RESULTS.md batches 10-11) =====
	// V is the denominator of the optimism stack (H, SIL weights, GAE, LP); these four
	// ingredients cut its noise/bias without touching its on-policy semantics. Toy
	// evidence (8 seeds, sil_hull base): ignition 1.25M -> 0.75-0.84M, finals 4.91 ->
	// 6.4, worst seed 0.1 -> 6.1; mirror alone cut the hallucinated-headroom floor 9x.
	// All independently flag-gated; each panel named below is its judge.
	//   - critic2 twin (Value/Twin Disagree): disjoint-half training, mean readout.
	//   - mirror pass (25% of rows): exact game symmetry; ObsMirror::Build hard-fails
	//     on layout drift. Slot-permutation is already trained in via shuffleSlots.
	//   - aux displacement head on the SHARED trunk (Value/Aux Disp NLL): the toy's
	//     single largest training lever; deliberately NOT inside the value head (wired
	//     there it biases V -- measured, ev 0.68 -> 0.49).
	//   - episodic blend (Value/Epi W, Value/EV): kNN over reservoir returns, weight
	//     epiWMax*(1-EV_ema) -- memory-backed baseline while V is young, self-retiring.
	//     Safe because it RETIRES (bank returns rot as the policy improves; the
	//     non-retiring peer-referee variant failed exactly there, batch 11).
	//   - privileged opponent conditioning (opp_embed, zero-init = no-op at load):
	//     the value family sees {isSelf, isOld, isExternal, ringAge}; the policy never
	//     does. Asymmetric actor-critic: variance reduction, provably unbiased.
	// Checkpoint compatibility: critic2/aux_disp/opp_embed fresh-init on old
	// checkpoints (allowNotExist); opp_embed's zero-init makes conditioning an exact
	// no-op until trained. Watch Value/EV (up), Value/Twin Disagree (down over time),
	// Headroom/H Mean (floor should DROP as V sharpens -- the compounding).
	// Goal-critic unification into this treatment is a DEFERRED separate lever.
	// Revert: these flags, restart.
	cfg.ppo.valueTwinEnabled = true;
	cfg.ppo.valueMirrorEnabled = true;
	cfg.ppo.mirrorMaxPlayersPerTeam = MAX_PLAYERS_PER_TEAM;
	cfg.ppo.auxDispEnabled = true;
	cfg.ppo.auxDispModel.layerSizes = { 256 };
	cfg.ppo.auxDispModel.activationType = activation;
	cfg.ppo.auxDispModel.addLayerNorm = false;
	// EPI BLEND: OFF — 7.0-vc post-mortem (2026-08-08, run ynt5j8np, ~415M steps).
	// Self-locking failure in prod: the anneal signal (Value/EV vs RETURNS) is
	// structurally ~0 at ts1 (returns std ~75 vs value-target scale ~0.5-5 over 100k-step
	// episodes), so Epi W pinned at 0.5 forever, permanently replacing half the GAE
	// baseline/bootstrap with kNN-over-230-dim-raw-obs noise (~random historical targets)
	// in a regime where per-step rewards are ~0.001-0.03. Advantages drowned; critic
	// could never earn the EV to retire the blend. The toy's validation rested on
	// same-scale returns/targets + 12-dim kNN, which do NOT transfer. Do not re-enable
	// without (a) a target-scale anneal signal and (b) a learned/low-dim kNN space.
	cfg.ppo.epiBlendEnabled = false;
	cfg.ppo.oppCondEnabled = true;
	// Resolved to THEIRS at the 7.2 merge (2026-08-08): disabling the gate is the stated
	// intent of that commit, and the cap-1.5 variant it replaces was mine — set while
	// chasing 7.0's entropy decline BEFORE the epi blend was identified as the cause, so
	// it was a fix for a symptom of a different bug. Recorded finding that outlives the
	// setting: this gate SATURATES at whatever cap it is given (2.95/3.0 on 6.2, exactly
	// 1.5000 on 7.0) whenever H is large, making it a constant multiplier on entropyScale
	// rather than a state-dependent gate. On 7.1, with the epi blend fixed, it finally read
	// BELOW its cap (1.4419) — i.e. sane advantages are what let it gate at all. If entropy
	// misbehaves on 7.2, entropyScale is the honest lever, not this flag.
	// ENT GATE: OFF — live 7.0 finding (2026-08-06), overriding the toy validation this line
	// used to cite (ignition seed-spread 5x -> 0). Mechanism: expectile H has a structural
	// noise floor once the critic bootstraps (V-dagger/V_exp sit above V by construction),
	// and the multiplier is mean-relative, so the gate NEVER retires -> permanent entropy
	// tax -> erratic play that severely limited skill acquisition. SIL alone is the
	// actuation. NOTE: this flag was still true in-repo while the box ran false (divergence
	// caught by the 2026-08-08 audit) — flipping it here is what makes push+update safe.
	// If attempt-supply ever needs reviving, LP = relu(H_old - H_now) is the fix-shaped
	// substitute (noise floor cancels, self-retiring) — 6.x design notes, untested.
	cfg.ppo.vdagEntGateEnabled = false;

	// Skill rating: Elo-style eval matches vs saved versions (logged as Rating/1v1). Also turns on
	// savePolicyVersions.
	//
	// Rating/1v1 IS INFLATED and is kept only for continuity. AddVersion copies the main's CURRENT
	// rating into each new version (PolicyVersionManager.cpp:50) and every goal moves BOTH sides,
	// so the pool's rating tracks the agent and the number is a treadmill: measured ~6x overstatement
	// (54% real win share against a predicted 75%). Read Ref/Oldest Share instead - the reference set
	// below is fixed, is never trained against, and therefore cannot inflate. Both series publish for
	// this run so the two can be compared directly; retire Rating/1v1 in a later change.
	cfg.skillTracker.enabled = true;

	// Permanent reference set: the honest yardstick. Log-spaced past checkpoints kept OUTSIDE the
	// rotating version ring, never evicted at the oldest end, never trained against.
	cfg.skillTracker.maxReferences = 8;
	cfg.skillTracker.referenceUpdateInterval = 64;

	// Train against archived past selves. ON since 2026-07-25, when the QD league was removed:
	// it is now the only self-play opponent source. The cascade is sequential and Nexto rolls
	// first, so the realized share is (1 - 0.15) * 0.30 = 0.255, and total non-self exposure is
	// 0.15 + 0.255 = 0.405 - deliberately equal to what the run measured before the strip
	// (Nexto 0.089 + league 0.319 = 0.408). That holds "how much non-self exposure" fixed while
	// "who the opponents are" changes, so the two are not confounded in one step. The exposure
	// level itself was tuned once (descendOpponentFrac 0.25 -> 0.35, after measuring
	// exploitability by archived styles) and should move only as its own experiment.
	cfg.trainAgainstOldVersions = true;
	cfg.trainAgainstOldChance = 0.30f;

	// GGL_SMOKE re-application (MUST live here, after the production assignments above - the
	// smoke block higher up runs first and would be clobbered). At production cadence a version
	// is archived every 25M steps and the reference battery runs every 64 iterations, so a short
	// smoke would report Ref/* never and prove nothing about the archive -> decimate -> save ->
	// reload -> battery path. These values exercise all of it within ~10 CPU iterations.
	// Never set on the trainer.
	if (const char* s = std::getenv("GGL_SMOKE"); s && s[0] && std::string(s) != "0") {
		cfg.tsPerVersion = 50'000;                        // ~2 smoke iterations per version
		cfg.skillTracker.updateInterval = 2;
		cfg.skillTracker.referenceUpdateInterval = 3;
		cfg.skillTracker.simTime = 20;                    // long enough that a random-init
		                                                  // policy actually scores, so the smoke
		                                                  // exercises Ref/Share and its persistence
		cfg.skillTracker.maxReferences = 4;               // small enough that decimation fires
		RG_LOG("GGL_SMOKE: tsPerVersion 50k, reference battery every 3 iters "
			"(exercises archive -> decimate -> save -> reload -> Ref/* panels)");
	}

	// FRESH RUN (4.0): the team-play lineage - padded 230-dim obs (AdvancedObsPadded(3)),
	// PHASE A all-1v1 curriculum, otherwise the proven 3.1 config (512-wide, secondary
	// goal critic; the 5.0 cold start re-timed it: tickSkip 4 -> 8, gamma 0.9985 ->
	// 0.9969, see TRAIN_GAMMA). Its own checkpoint folder + wandb run name so it
	// can NEVER accidentally resume the 3.1 lineage (obs 109 -> 230; the loader would abort
	// on the trunk's first Linear anyway, but the folder split keeps the failure impossible
	// rather than merely loud).
	// COLD START 2026-07-25 ("5.1"). The net is UNCHANGED from checkpoints_resid except for the
	// policy head's input width: conforming to COMPOSITION_CRITIC.md removed the 5-column Ladder
	// wire, so the head is 1152 (plain trunk width) where the resid lineage's checkpoints are
	// 1157. That alone forces a fresh lineage - resuming would abort on the policy head's first
	// Linear. Riding the same restart: the QD league removal, training against archived past
	// selves, and the permanent reference set.
	// A FRESH FOLDER is what makes this safe rather than merely loud: checkpoints_resid stays
	// fully intact (7.0 GB, 8 checkpoints, ~950M steps) instead of having every checkpoint in it
	// renamed corrupt_* by the loader's fallback, and PHASE A / an empty version pool / no
	// PHASE_B marker all follow by construction (the marker lives in the checkpoint folder).
	// 5.2 (2026-07-29): the critic trunk changes CRITIC.lt, GOAL_CRITIC.lt and both VDAG*.lt
	// shapes, so 5.1's checkpoints cannot load. The folder MUST move with the architecture and
	// this line is the whole safety mechanism: the loader walks newest->oldest and RENAMES every
	// checkpoint it fails to load to corrupt_<ts>, so booting this binary against checkpoints_5.1
	// would destroy that lineage (13.87B steps) rather than merely refuse to start. Same reason
	// the 5.1 cold start got its own folder instead of trusting the loader to abort loudly.
	// checkpoints_5.1 is therefore left fully intact as the restore point for this change: revert
	// = put this line and the criticTrunk/goalCritic/advFilter blocks back, rebuild, restart.
	// NOTE the flip side: with the code staged but not deployed, any restart of a REBUILT binary
	// starts 5.2 from step 0 rather than resuming 5.1. That is loud (Total Timesteps resets, new
	// wandb run) and non-destructive, but it is not a resume.
	// 5.3 (2026-07-30): the geometric critic (4th rung) is ON from step 0. Unlike 5.2 this is
	// NOT a shape break in the existing nets — geo_sigma/geo_rew/geo_v are three ADDITIONAL
	// files, so 5.2's checkpoints would technically load. The fresh folder is deliberate anyway,
	// for two reasons. First, the mechanism is validated as a cold-start accelerant actuating a
	// permanent potential; resuming a 175M-step policy into it measures neither the cold start
	// nor a clean ablation. Second, and the reason this is a hard rule rather than a preference:
	// a resumed 5.2 would carry a V_geo initialised at random into a live advantage stream, and
	// the HJB field needs to solve before its gap term means anything (Geo/Residual is the gate).
	// checkpoints_5.2 stays fully intact (2.3 GB, ~175M steps) as the restore point: revert =
	// put this line back and set cfg.ppo.geoEnabled = false, rebuild, restart.
	// 6.0 (2026-08-01, user-directed): tickSkip 8 -> 1 (15 Hz -> 120 Hz), actionDelay 0.
	// The fresh folder is MANDATORY here, and for a sharper reason than 5.2's shape break.
	// ts1 changes NO tensor shapes — obs is still 230-dim, the action table is still 90 — so
	// every checkpoint in checkpoints_5.3 would LOAD CLEANLY into this binary and silently
	// resume a 16.4B-step policy into an 8x different decision rate, a different gamma, a
	// different lambda and re-scaled per-step rewards. That is the one failure mode the
	// loader cannot catch: it only rejects checkpoints it cannot deserialize. checkpoints_5.3
	// (16.4B steps) stays fully intact as the restore point — revert = put this line, the
	// tickSkip/actionDelay block, TRAIN_GAMMA, gaeLambda, goalCritic.gamma, the two per-step
	// reward weights and the reachability windows back, rebuild, restart.
	// 6.1 (2026-08-03, user-directed): FULL-SIZE cold start — trunk 1280 / value side 1536 /
	// policy 768 (the FULL SIZE block above), geoGamma decoupled, still ts1. The width changes
	// break EVERY dense-net shape, so the fresh folder is load-bearing the same way 5.2's was:
	// booting this binary against checkpoints_6.0 would have the loader walk newest->oldest
	// renaming every checkpoint corrupt_<ts>, destroying that lineage (~3.07B steps) instead of
	// refusing to start. checkpoints_6.0 stays fully intact as the restore point: revert = put
	// this line, the 6.1 width lines, numGames and geoGamma back, rebuild, restart.
	// 6.1b (2026-08-04): cold restart of 6.1 with entropyScale 0.0175 (see that block) after
	// 6.1's from-birth entropy collapse — its policy spent the ENTIRE formative window at
	// entropy <= 0.05, the exact pathology (formative window without exploration) that
	// motivated the 5.0/6.0 cold starts. Same architecture, same everything else.
	// checkpoints_6.1 (1.1B steps, collapsed) is left fully intact for inspection/restore;
	// a fresh folder rather than a wipe, per the never-delete rule.
	// 6.2 (2026-08-05): the ACTUATION SWAP gets its own lineage, and this is mandatory for the
	// same reason the ts1 change was — not because anything shape-breaks, but because nothing
	// does. The swap (potential injection -> headroom-gated SIL + entropy gate) adds and removes
	// no models, so every checkpoint in checkpoints_6.1b loads CLEANLY into this binary and would
	// silently resume an 11.7B-step policy onto a completely different actuation channel, with
	// geoSeekBeta and vdagSeekBeta now 0. That is the one failure mode the loader cannot catch:
	// it only rejects checkpoints it cannot deserialize.
	// checkpoints_6.1b (11.7B steps, healthy — Nexto share 31%, entropy 0.55 and rising, 0 dead
	// units) stays fully intact as the restore point. Revert = put these two lines back.
	// If a mid-run swap ON TOP of 6.1b was actually intended, that is also these two lines — but
	// it confounds the actuation comparison with 11.7B steps of injection-trained history.
	// 7.0 (2026-08-07, user-directed): the composite value critic gets its own lineage.
	// MANDATORY, and for two reasons this time. The usual one: the incoming branch left this at
	// checkpoints_6.2, and parts of the new machinery (hull heads, aux displacement head)
	// FRESH-INIT rather than shape-break on an old checkpoint, so a stale folder could silently
	// resume old weights into a different critic — the failure the loader cannot catch.
	// Second: checkpoints_6.2 is not a real lineage any more. It was destroyed on 2026-08-07 by
	// three checkpoint-corruption incidents that exhausted every numbered checkpoint AND every
	// golden-archive entry, after which the loader started a fresh model; the folder now holds
	// only a 6.1b checkpoint staged there by hand to prove the boot probe still worked.
	// Surviving lineages, both intact and probe-verified: checkpoints_6.1b (12.63B) and
	// checkpoints_5.3 (16.7B). 6.2 has no recoverable state.
	// 7.1 (2026-08-08, user-directed cold start after the 7.0 post-mortem). 7.0 stagnated:
	// the episodic blend's anneal compared V against discounted RETURNS (std ~75 at ts1's
	// gamma) while V regresses GAE TARGET VALUES (~0.5-5), so EV was structurally ~0, Epi W
	// pinned at 0.5 forever, and half the GAE baseline/bootstrap was permanently replaced by
	// kNN-over-raw-obs noise. epiBlendEnabled is now false.
	// A COLD START is right rather than a resume: 7.0's weights were trained for 2.5B steps
	// against drowned advantages, so they carry the damage the fix removes -- and nothing
	// shape-breaks (epiBlend is a blend weight, not a model), so a stale folder would silently
	// resume exactly those weights. checkpoints_7.0 stays intact for post-mortem.
	// NOTE the entropy settings below are MINE, not the branch defaults: vdagEntGateCap 1.5
	// instead of 3.0, set while chasing 7.0's entropy decline before the epi blend was known to
	// be the cause. 6.1b and 6.2 both ran cap 3.0. Worth reverting to the default if 7.1's
	// entropy misbehaves -- it is an unvalidated middle rung, not a measured optimum.
	// 7.2 (2026-08-08, user-directed cold start on the audit-conformance config: geo trio
	// retired, entropy gate off; the epi blend was already off from the 7.0 post-mortem).
	// NOTE the incoming branch carried `checkpoints_6.1b` here — a stale value from before the
	// 7.0/7.1 lineages existed, not an intent to write there. Left as-is it would have been the
	// worst outcome available: 6.1b is the BEST SURVIVING MODEL (12.63B, probe-verified, the
	// only mature lineage left after 6.2 was destroyed on 2026-08-07), and nothing in this
	// config shape-breaks — so the loader would have cleanly resumed it under a materially
	// different critic (geo off, gate off, blend off) and then rotated its good checkpoints out
	// of the window. A silent semantic resume onto the one model that cannot be replaced.
	cfg.checkpointFolder = "checkpoints_7.2";
	cfg.metricsRunName = "7.2-vc";

	// A smoke MUST NOT be able to masquerade as the real run in wandb. Three sandbox smokes on
	// 2026-07-25 landed in the shared project under this exact display name, indistinguishable
	// at a glance from the live lineage. Prefer WANDB_MODE=offline too; this is the backstop for
	// when that is forgotten. Lives here, after the production assignment, per the rule above.
	if (const char* s = std::getenv("GGL_SMOKE"); s && s[0] && std::string(s) != "0")
		cfg.metricsRunName = "SMOKE-" + cfg.metricsRunName;

	// 1M default => a save every ~6s at ~170k SPS, making the 8-deep rotation window ~50
	// SECONDS wide - which is why the 2026-07-13 GPU lockup poisoned EVERY checkpoint in
	// it. 25M = a save every ~2.5 min, window ~20 min, and far less IO. Worst-case crash
	// loss rises from ~6s to ~2.5min of training - the wrapper restart costs more anyway.
	cfg.tsPerSave = 25'000'000;
	// GGL_SMOKE: force a real checkpoint round-trip within a few CPU iterations, so the smoke
	// actually exercises SaveVersions/SaveReferences and the reference_goals persistence rather
	// than only the in-memory path. MUST live here, AFTER the production assignment above - an
	// earlier override is silently clobbered (this exact mistake cost a smoke cycle on
	// 2026-07-25, which is why the rule is written down in three places). Never on the trainer.
	if (const char* s = std::getenv("GGL_SMOKE"); s && s[0] && std::string(s) != "0")
		cfg.tsPerSave = 100'000;
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

		// DETERMINISTIC (argmax) ACTIONS IN THE VIEWER (2026-08-04, user-directed).
		// The viewer samples from the policy like training does, and at ts1 that is actively
		// misleading: a decision every 8.3ms means a stochastic policy re-rolls its jump/dodge
		// bits 120x/second, so ~0.5 nats of entropy renders as continuous dithering rather than
		// as the behaviour the policy actually intends. That exact symptom is what convicted the
		// entropy coefficient earlier in this run ("the viewer showed cars flipping
		// continuously") — i.e. the viewer has been showing exploration noise, not skill.
		// Argmax shows what the policy would DO, which is also what the RLBot real-match path
		// already does, so the viewer and a real game now agree.
		// SAFE HERE AND ONLY HERE: PPOLearnerConfig warns that a LEARN iteration under
		// deterministic mode throws — render mode never calls Learn(), and this assignment is
		// inside the renderMode branch, so it cannot reach the trainer.
		// GGL_RENDER_SAMPLE=1 restores stochastic actions for watching exploration itself.
		cfg.ppo.deterministic = true;
		if (const char* d = std::getenv("GGL_RENDER_SAMPLE"); d && d[0] && std::string(d) != "0") {
			cfg.ppo.deterministic = false;
			RG_LOG("Render mode: GGL_RENDER_SAMPLE set - sampling actions (exploration visible)");
		} else {
			RG_LOG("Render mode: deterministic (argmax) actions");
		}

#ifdef GGL_VIZ_RLBOT
		// Let the viewer hand the other team to a real RLBot bot playing in OUR arena, so
		// pause/rewind/editing keep working against it. Built only with
		// -DGGL_VIZ_RLBOT=ON; the trainer links none of this.
		{
			// Harness roots to scan, most-preferred first. The first is relative to the
			// build dir the viewer runs from (build-viz/), which is where the rest of this
			// file's relative paths are anchored too.
			//
			// The second is the SIBLING checkout, and it is not an accident: Element lives
			// there with the Python venv both it and Nexto are already launched from
			// (see nexto/bot.toml's run_command_linux), so copying it into this tree would
			// fork a working install to no purpose. A root that doesn't exist is skipped,
			// so this stays harmless on a machine that only has one checkout.
			static const std::vector<std::string> botRoots = {
				std::filesystem::absolute("../rlbot-run").string(),
				"/home/luca/Projects/pulsar2/rlbot-run",
			};
			static GGL::VizRLBotServer rlbotServer;

			cfg.vizBotFinder = []() { return GGL::VizRLBotServer::FindBotConfigs(botRoots); };

			cfg.externalControlSource = [](
				const std::string& spec, Team team, const RLGC::GameState& state,
				const std::vector<int>& indices, std::vector<RLGC::Action>& out,
				std::vector<uint8_t>& outValid,
				LearnerConfig::ExternalControlStatus& status) -> bool {

				// The spec already carries an absolute path — VizControl accepted it only
				// because it exactly matched one the finder produced, so there is nothing
				// left to join and no browser-composed path fragment to trust.
				const std::string prefix = "rlbot:";
				std::string wantConfig;
				if (spec.rfind(prefix, 0) == 0)
					wantConfig = spec.substr(prefix.size());

				// What is actually running, so a change of bot, side, or car count
				// relaunches, and deselecting shuts the bot down rather than leaving it
				// holding a car nobody is watching.
				static std::string liveConfig;
				static Team liveTeam = Team::ORANGE;
				static size_t liveCars = 0;

				if (wantConfig.empty()) {
					if (!liveConfig.empty()) {
						rlbotServer.StopBot();
						liveConfig.clear();
					}
					return false;
				}

				if (wantConfig != liveConfig || team != liveTeam || indices.size() != liveCars) {
					if (rlbotServer.Listen(GGL::VizRLBotServer::DEFAULT_PORT)
						&& rlbotServer.LaunchBot(wantConfig, team, (int)indices.size())) {
						liveConfig = wantConfig;
						liveTeam = team;
						liveCars = indices.size();
					} else {
						status.error = rlbotServer.LastError();
						liveConfig.clear();
						return false;
					}
				}

				rlbotServer.Poll(state, indices);
				status.running = rlbotServer.BotRunning();
				status.connected = rlbotServer.Connected();
				status.name = rlbotServer.BotName();
				const bool got = rlbotServer.GetControls(out, outValid);
				status.controlling = got;
				return got;
			};
		}
#endif
	}
	// PSD (Basin-Racing) was REMOVED 2026-07-25. It had been disabled by a pre-registered
	// verdict since 2026-07-13 and had drifted architecturally incompatible with the live net
	// (Ladder wire columns, the arena pool, the critic family). Its two genuinely useful
	// signals - effective rank and dead-unit fraction - were PROMOTED to unconditional
	// telemetry instead of deleted (Plasticity/* panels): the residual cold start was
	// justified by effective-rank decay, so the project needs to keep measuring it.
	// History: git log -- GigaLearnCPP/src/private/GigaLearnCPP/PSD

	// ===== QD LEAGUE: REMOVED WHOLESALE 2026-07-25 =====
	// The MAP-Elites opponent archive is gone. It did not do the job it existed for, and the
	// only thing about it that was ever measured was its cost (docs/LEAGUE_RECON.md: 63 defect
	// claims, 59 survived adversarial verification).
	//   * It COLLAPSED on every lineage that learned: 17o01ku8 94 members -> 3 (coverage 0.426
	//     -> 0.005), bfl8mbw4 3 members in 93.9% of blocks, tpgyf8de 129 -> 11. Not via the
	//     documented competenceFloor cull - that code is UNREACHABLE, because Cull() protects
	//     every cell elite and dedup guarantees one elite per cell. The real engine was
	//     DedupCells at each quantile rebin.
	//   * What it actually served were noisy copies of the UNTRAINED BIRTH NETWORK. All members
	//     were lineage 0 (both reseeds rejected: TryInsert needs strictly better fitness, and a
	//     fresh snapshot scores ~0 against itself while 31 incumbents sat at exactly 0), measuring
	//     0.039-0.064 relative-L2 from policy_versions/0 and 0.73-0.78 from the current main.
	//   * Cost: ~3.6% of wall clock in the evolve barrier, a serving tax, and 2.7 GB rewritten
	//     every save into an index-keyed store that was already desynced (141 .pt files for a
	//     121-member archive).
	// Replaced by: archived past selves (trainAgainstOldVersions above) + Nexto for training
	// diversity, and the permanent reference set for honest measurement. History:
	// git log -- GigaLearnCPP/src/private/GigaLearnCPP/League ; tag pre-league-strip-20260725

	// ===== STEERING: REMOVED WHOLESALE 2026-07-25 =====
	// The activation-steering research programme is finished. Its actuation went inert when the
	// optimism work superseded it, and the remainder - possession-outcome labeling, the frontier
	// reset pool, FrontierDrillState, the practice/control arena split, the census and the
	// learning-progress miner - is removed here because it CONTRADICTS the constraint set of
	// research/reports/COMPOSITION_CRITIC.md, which the trainer now implements:
	//   C1 (homogeneity)      all environment instances identical; no dedicated drill/reset
	//                         instances. The practice/control split violated this.
	//   C2 (no env design)    no banked reset states, no success-state memories that define the
	//                         objective retrospectively. FrontierPool was exactly that, and the
	//                         paper's section 4.1 criticises the design by name.
	// Keeping it would mean any future deployment measured "architecture PLUS an environment-side
	// drill curriculum" - confounding the paper's central claim. Section 6.1 already reports the
	// spawn curriculum separately as a skyline that violates C2, for the same reason.
	//
	// The reset mix below is NOT a curriculum: it is fixed, identical in every arena, and does
	// not adapt. That is what C1 asks for.
	//
	// The lessons survive in research/reports/ (STEERING.md, STEERED_PRACTICE.md, FEAR_MINE.md,
	// KNOWING_DOING.md) and the code is one `git log -- <path>` away from tag
	// pre-strip-20260725. Two of them still bind and are cited from this file: the CLIPPING
	// RATCHET (a push large enough to move the IS ratio outside the clip window makes PPO keep
	// positive-advantage gradients and discard negative ones) and CRITIC ALIASING at episode
	// boundaries (AttemptResolutionCondition, two Elo collapses - removed with the rest).

	// ===== OPTIMISM: the composition critic (research/reports/COMPOSITION_CRITIC.md) =====
	// The live stack is exactly the paper's: V_real (the critic) -> V_exp (return-level
	// expectile, measurement) -> twin composition critics V-dagger, actuated by ONE potential
	// term, Phi = +H with H = relu(min(V1,V2) - V_real). See PPOLearnerConfig::vdagEnabled /
	// vdagTau / vdagSeekBeta - the mechanism lives with the headroom critic, not here.
	//
	// REMOVED 2026-07-25 to conform to the paper: RND novelty, the quasimetric map, the
	// goal/concede banks, V_metric, gap_PK, the closure drive Phi = -(gap_KD + gap_PK), the
	// 5-column policy-head wire (policy input returns to trunk width), and the
	// impossible-control falsification family that scored gap_PK. The paper's section 4.5
	// measured the closure sign as CATASTROPHIC on a peaked field (touch 0.014/0.027 vs 0.244
	// for plain PPO), and section 6.4 rejects the quantile variant; section 5 states plainly that
	// the policy never consumes H, which is what retires the wire.
	// History: git log -- docs/LADDER.md docs/EMERGENCE.md
	if (!cfg.renderMode) {
		// V_exp: trained every iteration on the critic's own extrinsic targets through a
		// DETACHED trunk read. Publishes Gap/* - nothing injects from it.
		cfg.gapSensor.enabled = true;

		// EXTERNAL OPPONENT: NEXTO (2026-07-20, user-directed "play better bots
		// to force the aerials"). On serveFrac of iterations the whole fleet's
		// non-self team is Nexto (frozen public SSL-level RLGym bot, tick_skip 8
		// like this run; adapter + full rationale in NextoOpponent.h). Two jobs:
		// (1) exposure - contesting an opponent that lives in the air puts the
		// high-ball states our self-play never produces into the buffer; (2) the
		// Nexto/Goals For/Against panels are a FIXED external yardstick immune to
		// the pool inflation measured in H2_TRUNCATION.md. Rows excluded from
		// training like all opponent sources; eval paths untouched (Rating
		// semantics unchanged). No automatic guard (the latch was removed
		// 2026-07-25) - watch the Nexto panels. Expect to LOSE heavily at first -
		// the goal-diff SLOPE is the signal, not the level. Revert = false.
		cfg.externalOpponent.enabled = true;
		cfg.externalOpponent.modelPath =
			"/home/luca/Projects/pulsar2-3.0/rlbot-run/nexto/nexto-model.pt";
		cfg.externalOpponent.serveFrac = 0.15f;
		// GGL_SMOKE: serve on most iterations so a short smoke exercises the
		// adapter (obs port + action map + goal telemetry) deterministically.
		// Lives HERE, after the production assignment - the smoke block up top
		// runs first and would be clobbered. Never set on the trainer.
		if (const char* s2 = std::getenv("GGL_SMOKE"); s2 && s2[0] && std::string(s2) != "0")
			cfg.externalOpponent.serveFrac = 0.75f;
	} else {
		// Render must build the SAME architecture as the trainer or it loads the wrong
		// shapes (the ba4f33a bug). With the wire gone the policy head is plain trunk
		// width again, so this is now only about gapSensor's own head; it stays inert
		// in render because Learn() never runs there.
		cfg.gapSensor.enabled = true; // arch parity only; Learn() never runs in render
	}

	// Team-mode arena split, decided by the lineage-scoped phase marker (see the
	// PHASE_B_RATING_TRIGGER comment). Must be set before the Learner is built -
	// EnvCreateFunc reads these.
	//
	// RESTORED 2026-07-26. This block was deleted as collateral in 0fac55f ("Remove steering
	// wholesale"): it sat between the g_NumPracticeArenas and g_FrontierPool steering blocks
	// that commit was legitimately removing. The marker's WRITE side (the iteration callback
	// below) survived, the READ side did not, so g_PhaseB was pinned false and the trigger
	// re-fired forever - the trainer wrote the marker, exited 99, relaunched into PHASE A,
	// and repeated every ~6 minutes for 70M+ steps (five loops in one log, ts 3.695B-3.768B).
	// If the marker read ever disappears again the symptom is that exact restart loop.
	// PHASE_B_ENABLED guards the READ as well as the trigger, so an inherited or hand-copied
	// marker cannot engage team modes behind your back.
	g_PhaseB = PHASE_B_ENABLED && std::filesystem::exists(cfg.checkpointFolder / PHASE_B_MARKER);
	TEAM_SPIRIT = g_PhaseB ? 0.6f : 0.3f; // 5.0 spirit schedule (see the declaration)
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
	// take some evals to leave their initial value. Without the envCreateFn assignment
	// SkillEnvCreateFunc is orphaned and the tracker clones the TRAINING create-func over its
	// low indices, i.e. an all-1v1 eval fleet - the bug this function exists to prevent.
	g_SkillNumArenas = cfg.skillTracker.numArenas;
	g_SkillArenas2v2 = g_PhaseB ? RS_MAX(1, (int)(g_SkillNumArenas * PHASE_B_FRAC_2V2)) : 0;
	g_SkillArenas3v3 = g_PhaseB ? RS_MAX(1, (int)(g_SkillNumArenas * PHASE_B_FRAC_3V3)) : 0;
	cfg.skillTracker.envCreateFn = SkillEnvCreateFunc;
	if (cfg.skillTracker.enabled)
		RG_LOG("Skill tracker eval fleet: "
			<< (g_SkillNumArenas - g_SkillArenas2v2 - g_SkillArenas3v3) << " 1v1 / "
			<< g_SkillArenas2v2 << " 2v2 / " << g_SkillArenas3v3 << " 3v3 arenas");

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
			// Team modes disabled for this lineage (PHASE_B_ENABLED) — never arm the streak,
			// never write the marker, never exit(99). The panels above keep publishing so the
			// series does not vanish mid-history; they just stay pinned at 0.
			if (!PHASE_B_ENABLED)
				return;
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
