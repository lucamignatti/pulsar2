#include "TestFramework.h"

#include <RLGymCPP/Rewards/CommonRewards.h>
#include <RLGymCPP/Rewards/ScheduledScaleReward.h>
#include <RLGymCPP/Rewards/ZeroSumReward.h>
#include <RLGymCPP/Gamestates/GameState.h>

#include <atomic>
#include <cmath>

using namespace RLGC;

// Coverage for the 2026-08-17 reward changes:
//   * ScheduledScaleReward - the scaffold-anneal wrapper. The load-bearing claims are
//     (a) the scale is latched per EPISODE so a wrapped PBRS term stays an exact
//     potential while the schedule ramps, and (b) it forwards GetAllRewards so
//     team-aware children are not silently downgraded to their per-player path.
//   * KickoffRaceReward - the window re-derivation. This reward had never fired in any
//     lineage (2 nonzero wandb samples ever) because its window closed at 2.0s, below
//     any physically realistic kickoff touch. These tests pin the payout at the touch
//     times actually observed in this bot rather than at the old unreachable ones.
//   * AerialTouchReward - the refire cooldown, which is the term's anti-farm mechanism
//     and was measured being outrun by a juggle at 0.8s.

// ---------------------------------------------------------------------------
// Frame runner: N players, explicit deltaTime, stable prev pointers.
// Frame 0 is the post-Reset state and sees a null prev, matching the live loop.
// PreStep is driven exactly where EnvSet drives it: once per state, before the
// reward pass (EnvSet.cpp - PreStep at the top of the step, GetAllRewards after).
// ---------------------------------------------------------------------------
struct PlayerFrame {
	Vec pos = Vec(0, 0, 17);
	Vec vel = Vec(0, 0, 0);
	Team team = Team::BLUE;
	bool onGround = true;
	bool touched = false;
	bool demoed = false;
	float airTime = 0;
};

struct WorldFrame {
	Vec ballPos = Vec(0, 0, 93);
	Vec ballVel = Vec(0, 0, 0);
	float deltaTime = 1.f / 15.f; // tickSkip 8
	std::vector<PlayerFrame> players;
};

// Returns per-frame, per-player rewards: out[frame][player].
static std::vector<std::vector<float>> Run(Reward& r, const std::vector<WorldFrame>& frames) {
	std::vector<GameState> states;
	states.reserve(frames.size());
	for (const WorldFrame& wf : frames) {
		GameState gs;
		gs.ball.pos = wf.ballPos;
		gs.ball.vel = wf.ballVel;
		gs.deltaTime = wf.deltaTime;
		for (size_t i = 0; i < wf.players.size(); i++) {
			const PlayerFrame& pf = wf.players[i];
			Player p = {};
			p.index = (int)i;
			p.carId = (uint32_t)(i + 1);
			p.team = pf.team;
			p.pos = pf.pos;
			p.vel = pf.vel;
			p.isOnGround = pf.onGround;
			p.ballTouchedStep = pf.touched;
			p.isDemoed = pf.demoed;
			p.airTime = pf.airTime;
			gs.players.push_back(p);
		}
		states.push_back(std::move(gs));
	}
	for (size_t i = 1; i < states.size(); i++) {
		states[i].prev = &states[i - 1];
		for (size_t j = 0; j < states[i].players.size(); j++)
			states[i].players[j].prev = &states[i - 1].players[j];
	}

	r.Reset(states[0]);
	std::vector<std::vector<float>> out;
	for (size_t i = 0; i < states.size(); i++) {
		r.PreStep(states[i]);
		out.push_back(r.GetAllRewards(states[i], false));
	}
	return out;
}

// Single-player convenience.
static std::vector<float> RunOne(Reward& r, const std::vector<WorldFrame>& frames) {
	std::vector<float> out;
	for (auto& row : Run(r, frames))
		out.push_back(row.at(0));
	return out;
}

static float Sum(const std::vector<float>& v) {
	float s = 0;
	for (float x : v)
		s += x;
	return s;
}

// ---------------------------------------------------------------------------
// Stubs
// ---------------------------------------------------------------------------

// An exact potential: r = gamma*Phi(s') - Phi(s), Phi = ballZ/1000. At gamma == 1 any
// trajectory telescopes to Phi(end) - Phi(start), so a closed loop nets exactly 0.
class StubPotential : public Reward {
public:
	float gamma;
	explicit StubPotential(float gamma) : gamma(gamma) {}
	static float Phi(const GameState& s) { return s.ball.pos.z / 1000.f; }
	virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
		if (!state.prev)
			return 0;
		return gamma * Phi(state) - Phi(*state.prev);
	}
};

// Implements its real behaviour ONLY in GetAllRewards (like the team-closest
// BallProximityPotentialReward). A wrapper that forwards just GetReward returns the
// sentinel instead - that is the bug this stub is here to catch.
class StubTeamAware : public Reward {
public:
	constexpr static float TEAM_VALUE = 7.f;
	constexpr static float PER_PLAYER_SENTINEL = -99.f;
	virtual std::vector<float> GetAllRewards(const GameState& state, bool isFinal) override {
		return std::vector<float>(state.players.size(), TEAM_VALUE);
	}
	virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
		return PER_PLAYER_SENTINEL;
	}
};

static WorldFrame OnePlayerAt(float ballZ) {
	WorldFrame wf;
	wf.ballPos = Vec(0, 0, ballZ);
	wf.players.push_back(PlayerFrame{});
	return wf;
}

// ---------------------------------------------------------------------------
// ScheduledScaleReward
// ---------------------------------------------------------------------------

// The latch: a scale published mid-episode must NOT take effect until the next Reset.
// This is what keeps a wrapped potential a potential (see below) - if the scale applied
// immediately, k would differ between Phi(s) and Phi(s') within one transition.
TEST(ScheduledScale_latches_at_reset_not_mid_episode) {
	std::atomic<float> scale{ 1.0f };
	ScheduledScaleReward r(new StubTeamAware(), &scale);

	std::vector<WorldFrame> frames = { OnePlayerAt(93), OnePlayerAt(93), OnePlayerAt(93) };

	std::vector<GameState> states(1);
	states[0].players.push_back(Player{});
	r.Reset(states[0]); // latch 1.0

	GameState gs;
	Player p = {};
	p.index = 0;
	gs.players.push_back(p);

	CHECK_NEAR(r.GetAllRewards(gs, false).at(0), StubTeamAware::TEAM_VALUE, 1e-5f);

	scale.store(0.25f); // learner publishes a new scale mid-episode
	CHECK_NEAR(r.GetAllRewards(gs, false).at(0), StubTeamAware::TEAM_VALUE, 1e-5f);

	r.Reset(states[0]); // next episode picks it up
	CHECK_NEAR(r.GetAllRewards(gs, false).at(0), StubTeamAware::TEAM_VALUE * 0.25f, 1e-5f);
}

// The PBRS guarantee. A wrapped exact potential must still telescope to ~0 over a closed
// loop at gamma == 1, at ANY point on the anneal ramp - including after the schedule has
// moved between episodes. If the scale were read per-step this sum would be nonzero,
// which is the "free reward from a grind" failure the potential terms exist to preclude.
TEST(ScheduledScale_preserves_pbrs_telescoping_across_ramp) {
	std::atomic<float> scale{ 1.0f };
	ScheduledScaleReward r(new StubPotential(1.0f), &scale);

	// Closed loop: ball rises then returns to exactly its starting height.
	std::vector<WorldFrame> loop = {
		OnePlayerAt(93), OnePlayerAt(500), OnePlayerAt(1200), OnePlayerAt(500), OnePlayerAt(93)
	};

	for (float k : { 1.0f, 0.7f, 0.4f }) {
		scale.store(k);
		float net = Sum(RunOne(r, loop)); // Run() calls Reset first, so k is latched here
		CHECK_NEAR(net, 0.f, 1e-5f);
	}
}

// Scaling is linear, so the per-frame credit shape is preserved exactly - annealing
// changes the term's magnitude and nothing about where it pays.
TEST(ScheduledScale_is_exactly_linear_in_scale) {
	std::atomic<float> scale{ 1.0f };
	ScheduledScaleReward r(new StubPotential(0.9969f), &scale);

	std::vector<WorldFrame> traj = { OnePlayerAt(93), OnePlayerAt(700), OnePlayerAt(300) };

	scale.store(1.0f);
	std::vector<float> full = RunOne(r, traj);
	scale.store(0.4f);
	std::vector<float> annealed = RunOne(r, traj);

	CHECK(full.size() == annealed.size());
	for (size_t i = 0; i < full.size(); i++)
		CHECK_NEAR(annealed[i], full[i] * 0.4f, 1e-5f);
}

// The wrapper must route through the child's GetAllRewards. RewardWrapper does NOT
// forward it, so this is the trap any new wrapper in this tree can fall into.
TEST(ScheduledScale_forwards_get_all_rewards_for_team_aware_child) {
	std::atomic<float> scale{ 0.5f };
	ScheduledScaleReward r(new StubTeamAware(), &scale);

	WorldFrame wf;
	wf.players.push_back(PlayerFrame{ Vec(0, -500, 17), Vec(), Team::BLUE });
	wf.players.push_back(PlayerFrame{ Vec(0, 500, 17), Vec(), Team::ORANGE });

	auto out = Run(r, { wf });
	CHECK(out.at(0).size() == 2);
	for (float v : out.at(0))
		CHECK_NEAR(v, StubTeamAware::TEAM_VALUE * 0.5f, 1e-5f);
}

// Placement check: ZeroSum(Scaffold(child)) must equal scale * ZeroSum(child), which is
// what makes it safe to scale INSIDE the wrapper (and so keep EnvSet's cached
// dynamic_cast, and the term's wandb panel, intact).
TEST(ScheduledScale_inside_zerosum_equals_scaled_zerosum) {
	WorldFrame wf;
	wf.players.push_back(PlayerFrame{ Vec(0, -500, 17), Vec(), Team::BLUE });
	wf.players.push_back(PlayerFrame{ Vec(0, 500, 17), Vec(), Team::ORANGE });
	wf.players[0].touched = true;
	wf.ballVel = Vec(0, 1000, 0);

	std::vector<WorldFrame> traj = { wf, wf };

	ZeroSumReward plain(new TouchAccelReward(), 0.3f);
	auto unscaled = Run(plain, traj);

	std::atomic<float> scale{ 0.4f };
	ZeroSumReward scaled(new ScheduledScaleReward(new TouchAccelReward(), &scale), 0.3f);
	auto out = Run(scaled, traj);

	for (size_t f = 0; f < unscaled.size(); f++)
		for (size_t p = 0; p < unscaled[f].size(); p++)
			CHECK_NEAR(out[f][p], unscaled[f][p] * 0.4f, 1e-5f);
}

// ---------------------------------------------------------------------------
// KickoffRaceReward - the window re-derivation
// ---------------------------------------------------------------------------

// Kickoff-like world: ball at centre at rest, one blue toucher, one contesting orange.
static std::vector<WorldFrame> KickoffFrames(float touchAtSecs, float totalSecs, bool contested) {
	const float dt = 1.f / 15.f;
	int nFrames = (int)std::lround(totalSecs / dt) + 1;
	int touchFrame = (int)std::lround(touchAtSecs / dt);

	std::vector<WorldFrame> frames;
	for (int i = 0; i < nFrames; i++) {
		WorldFrame wf;
		wf.deltaTime = dt;
		wf.ballPos = Vec(0, 0, 93);
		wf.ballVel = Vec(0, 0, 0);

		PlayerFrame blue;
		blue.team = Team::BLUE;
		blue.pos = Vec(0, -3280, 17); // corner spawn distance to a resting ball
		blue.touched = (i == touchFrame);

		PlayerFrame orange;
		orange.team = Team::ORANGE;
		// Contesting: inside CONTEST_DIST of the ball. Delay kickoff: parked far away and
		// not closing, so Contested() is false and the term must pay nothing.
		orange.pos = contested ? Vec(0, 1000, 17) : Vec(0, 4500, 17);
		orange.vel = Vec(0, 0, 0);

		wf.players.push_back(blue);
		wf.players.push_back(orange);
		frames.push_back(wf);
	}
	return frames;
}

// THE REGRESSION TEST. The bot's healthy median kickoff first touch is ~3.4s; under the
// old 2.0s window this paid exactly 0, which is why the term had 2 nonzero samples across
// every run since it shipped. It must now pay real credit here.
TEST(KickoffRace_pays_at_the_measured_median_touch_time) {
	KickoffRaceReward r;
	auto out = Run(r, KickoffFrames(3.4f, 6.0f, /*contested*/ true));

	float total = 0;
	for (auto& row : out)
		total += row.at(0);

	// 3.4s sits on the ramp between FULL_CREDIT_SECS (2s) and WINDOW_SECS (5s).
	// The touch lands on frame round(3.4*15) = 51, and `elapsed` is one step ahead of
	// frame*dt because PreStep fires on the first post-reset step too - by then one
	// tickSkip of sim time has genuinely passed, which is the live ordering (EnvSet
	// steps, then PreStep, then the reward pass). So elapsed = 52/15 = 3.467s and the
	// payout is 1 - (3.467-2)/(5-2) = 0.511.
	CHECK_NEAR(total, 0.511f, 0.01f);
	CHECK(total > 0.f);
}

// A near-optimal kickoff should not be nickel-and-dimed for not being instantaneous:
// anything inside the grace window pays in full.
TEST(KickoffRace_full_credit_inside_grace_window) {
	KickoffRaceReward r;
	auto out = Run(r, KickoffFrames(1.5f, 6.0f, true));
	float total = 0;
	for (auto& row : out)
		total += row.at(0);
	CHECK_NEAR(total, 1.0f, 1e-3f);
}

// Past the window the race is over and nothing pays - a dawdled touch is not a race win.
TEST(KickoffRace_zero_past_window) {
	KickoffRaceReward r;
	auto out = Run(r, KickoffFrames(5.6f, 8.0f, true));
	float total = 0;
	for (auto& row : out)
		total += row.at(0);
	CHECK_NEAR(total, 0.f, 1e-5f);
}

// The contested gate is the safeguard against being farmed by a delay kickoff, and the
// window change must not have weakened it.
TEST(KickoffRace_uncontested_delay_kickoff_still_pays_zero) {
	KickoffRaceReward r;
	auto out = Run(r, KickoffFrames(3.0f, 6.0f, /*contested*/ false));
	float total = 0;
	for (auto& row : out)
		total += row.at(0);
	CHECK_NEAR(total, 0.f, 1e-5f);
}

// Non-kickoff episodes must stay completely inert (the ball is not at centre at rest).
TEST(KickoffRace_inert_when_episode_is_not_a_kickoff) {
	KickoffRaceReward r;
	auto frames = KickoffFrames(3.0f, 6.0f, true);
	for (auto& wf : frames)
		wf.ballPos = Vec(2000, 1500, 93); // mid-play ball
	auto out = Run(r, frames);
	float total = 0;
	for (auto& row : out)
		total += row.at(0);
	CHECK_NEAR(total, 0.f, 1e-5f);
}

// ---------------------------------------------------------------------------
// AerialTouchReward - the refire cooldown
// ---------------------------------------------------------------------------

// A qualifying aerial strike: airborne, sustained flight, high ball, real impulse.
static std::vector<WorldFrame> JuggleFrames(float touchIntervalSecs, float totalSecs) {
	const float dt = 1.f / 15.f;
	int nFrames = (int)std::lround(totalSecs / dt) + 1;
	int stride = (int)std::lround(touchIntervalSecs / dt);

	std::vector<WorldFrame> frames;
	for (int i = 0; i < nFrames; i++) {
		WorldFrame wf;
		wf.deltaTime = dt;
		wf.ballPos = Vec(0, 0, 1450); // full height credit
		bool touch = (i > 0) && (i % stride == 0);
		// Ball delta-v of 600 uu/s on touch frames (>= FULL_CREDIT_DELTA_V 500).
		wf.ballVel = touch ? Vec(0, 600, 0) : Vec(0, 0, 0);

		PlayerFrame p;
		p.team = Team::BLUE;
		p.pos = Vec(0, 0, 1300);
		p.onGround = false;
		p.airTime = 2.0f; // >= MAX_CREDIT_AIR_TIME 1.75
		p.touched = touch;

		wf.players.push_back(p);
		frames.push_back(wf);
	}
	return frames;
}

// THE ANTI-FARM PROPERTY. A juggle re-touching every 1s used to clear the 0.8s cooldown
// on every single touch, so the farm ran at the full payout ceiling - it was answered
// twice by cutting the weight instead. At 5s the same juggle collects at most once per
// cooldown, which is the ~6x income-ceiling reduction the change is for.
TEST(AerialTouch_cooldown_rate_limits_a_one_second_juggle) {
	AerialTouchReward r;
	// 10s of juggling, a qualifying touch every 1s => 10 touches offered.
	auto out = RunOne(r, JuggleFrames(1.0f, 10.0f));

	int payouts = 0;
	for (float v : out)
		if (v > 0)
			payouts++;

	// 10s at a 5s cooldown: the first touch pays, then one more per 5s elapsed.
	CHECK(payouts <= 3);
	CHECK(payouts >= 1);
}

// ...while a legitimate aerial cadence (well above the cooldown) is untouched: every
// strike still pays in full. This asymmetry is the entire justification for fixing the
// farm with the cooldown rather than with the weight.
TEST(AerialTouch_legitimate_cadence_pays_every_strike) {
	AerialTouchReward r;
	// A strike every 6s over 24s => 4 offered, all outside the 5s cooldown.
	auto out = RunOne(r, JuggleFrames(6.0f, 24.0f));

	int payouts = 0;
	for (float v : out)
		if (v > 0)
			payouts++;

	CHECK(payouts == 4);
}

// The cooldown is now wall-clock, so the same juggle is rate-limited identically at any
// tickSkip. The old step-count constant claimed this and did not have it: 12 steps was
// 0.8s at tickSkip 8 and 0.1s on the ts1 lineages.
TEST(AerialTouch_cooldown_is_tickskip_invariant) {
	// Re-times the same 10s / touch-every-1s trajectory onto a different decision rate.
	auto countPayouts = [](float dt) {
		const WorldFrame proto = JuggleFrames(1.0f, 10.0f).front();
		int nFrames = (int)std::lround(10.0f / dt) + 1;
		int stride = (int)std::lround(1.0f / dt);

		std::vector<WorldFrame> retimed;
		for (int i = 0; i < nFrames; i++) {
			WorldFrame wf = proto;
			wf.deltaTime = dt;
			bool touch = (i > 0) && (i % stride == 0);
			wf.ballVel = touch ? Vec(0, 600, 0) : Vec(0, 0, 0);
			wf.players[0].touched = touch;
			retimed.push_back(wf);
		}

		AerialTouchReward r;
		int n = 0;
		for (float v : RunOne(r, retimed))
			if (v > 0)
				n++;
		return n;
	};

	CHECK(countPayouts(1.f / 15.f) == countPayouts(1.f / 120.f));
}
