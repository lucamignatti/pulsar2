#include "TestFramework.h"

#include <GigaLearnCPP/Util/VizControl.h>

#include <cmath>
#include <filesystem>

using namespace GGL;

// The viz control panel's rewind lets you scrub back, edit, and play forward again.
// That is only honest if restoring a snapshot puts the engine in the state it was
// actually in — and the v3 facade warns that some CarState fields it carries are
// inert (RocketSim.h:46-48), so a state written back may not be the state read out.
//
// These tests pin that down the only way that means anything: restore a snapshot and
// re-simulate the SAME control sequence, then check the physics lands in the same
// place. A field-by-field round-trip check would pass even if the engine kept hidden
// state the snapshot never saw; only re-simulation catches that.

static void EnsureRocketSimViz() {
	static bool inited = false;
	if (inited)
		return;
	for (const char* path : { "collision_meshes", "../collision_meshes", "build/collision_meshes" }) {
		if (std::filesystem::exists(std::filesystem::path(path) / "soccar")) {
			RocketSim::Init(path);
			inited = true;
			return;
		}
	}
	// RocketSim::Init is idempotent-guarded upstream; if another test file already
	// initialized it, creating an arena works without a second Init
	inited = true;
}

// Deterministic control stream, independent of the engine's thread_local RNG (which
// reseeds off the wall clock and would desync the two runs we need to be identical).
struct ControlGen {
	uint64_t s;
	explicit ControlGen(uint64_t seed) : s(seed) {}

	uint32_t Next() {
		s = s * 6364136223846793005ULL + 1442695040888963407ULL;
		return (uint32_t)(s >> 33);
	}
	float UnitFloat() { return (float)(Next() % 2001) / 1000.f - 1.f; } // [-1, 1]
	bool Chance(uint32_t oneIn) { return (Next() % oneIn) == 0; }

	CarControls Make() {
		CarControls c = {};
		c.throttle = UnitFloat();
		c.steer = UnitFloat();
		c.pitch = UnitFloat();
		c.yaw = UnitFloat();
		c.roll = UnitFloat();
		c.boost = Chance(3);
		c.jump = Chance(5);
		c.handbrake = Chance(8);
		return c;
	}
};

// What a drive actually exercised. A test that claims to cover contacts and demos has
// to prove they happened rather than assume it — checking the ball's FINAL position is
// not enough, since over ten seconds it can wander away and bounce back.
struct DriveWitness {
	int demoedCarTicks = 0;
	float maxBallSpeed = 0;
};

// Drive every car for `ticks` with a control stream reproducible from `seed`.
static void DriveArena(Arena* arena, uint64_t seed, int ticks, DriveWitness* witness = NULL) {
	ControlGen gen(seed);
	for (int t = 0; t < ticks; t++) {
		for (Car* car : arena->_cars)
			car->controls = gen.Make();
		arena->Step(1);
		if (witness) {
			for (Car* car : arena->_cars)
				if (car->GetState().isDemoed)
					witness->demoedCarTicks++;
			Vec v = arena->ball->GetState().vel;
			witness->maxBallSpeed = std::max(witness->maxBallSpeed, sqrtf(v.x*v.x + v.y*v.y + v.z*v.z));
		}
	}
}

static Arena* MakeArena(int teamSize) {
	Arena* arena = Arena::Create(GameMode::SOCCAR);
	for (int i = 0; i < teamSize; i++) {
		arena->AddCar(Team::BLUE);
		arena->AddCar(Team::ORANGE);
	}
	arena->ResetToRandomKickoff(/*seed=*/1234);
	return arena;
}

// Engine-agnostic view of the car fields this test compares. Under v3 it includes
// timeSinceBoosted / isBoosting / supersonicGraceTimer / bumpCooldownTimer — the four
// the v2-compat CarState cannot carry, and the reason the snapshot goes through the
// FFI instead. If someone routes ArenaSnapshot back through Car::GetState/SetState,
// these are what catch it.
struct CarFacts {
	Vec pos, vel, angVel, fwd, up;
	float boost, jumpTime, flipTime, airTime, airTimeSinceJump, handbrakeVal, demoRespawnTimer;
	float timeSinceBoosted, boostingTime, supersonicGraceTimer, bumpCooldownTimer;
	int isOnGround, hasJumped, hasDoubleJumped, hasFlipped, isFlipping, isJumping;
	int isDemoed, isBoosting, isSupersonic;
};

static CarFacts Facts(const GGL::VizCarState& s) {
	CarFacts f = {};
#ifdef RG_ROCKETSIM_V3
	auto fnVec = [](const RsfVec3& v) { return Vec(v.x, v.y, v.z); };
	f.pos = fnVec(s.phys.pos);
	f.vel = fnVec(s.phys.vel);
	f.angVel = fnVec(s.phys.angVel);
	f.fwd = fnVec(s.phys.rotMat.forward);
	f.up = fnVec(s.phys.rotMat.up);
	f.timeSinceBoosted = s.timeSinceBoosted;
	f.boostingTime = s.boostingTime;
	f.supersonicGraceTimer = s.supersonicGraceTimer;
	f.bumpCooldownTimer = s.bumpCooldownTimer;
	f.isBoosting = s.isBoosting;
#else
	f.pos = s.pos;
	f.vel = s.vel;
	f.angVel = s.angVel;
	f.fwd = s.rotMat.forward;
	f.up = s.rotMat.up;
	f.boostingTime = s.timeSpentBoosting;
	f.bumpCooldownTimer = s.carContact.cooldownTimer;
#endif
	f.boost = s.boost;
	f.jumpTime = s.jumpTime;
	f.flipTime = s.flipTime;
	f.airTime = s.airTime;
	f.airTimeSinceJump = s.airTimeSinceJump;
	f.handbrakeVal = s.handbrakeVal;
	f.demoRespawnTimer = s.demoRespawnTimer;
	f.isOnGround = s.isOnGround;
	f.hasJumped = s.hasJumped;
	f.hasDoubleJumped = s.hasDoubleJumped;
	f.hasFlipped = s.hasFlipped;
	f.isFlipping = s.isFlipping;
	f.isJumping = s.isJumping;
	f.isDemoed = s.isDemoed;
	f.isSupersonic = s.isSupersonic;
	return f;
}

// Reports WHICH field diverged — a bare boolean here would mean bisecting the engine
// by hand the first time this fails.
static std::string DiffSnapshots(const ArenaSnapshot& a, const ArenaSnapshot& b) {
	std::ostringstream s;

	auto fnVec = [&](const char* what, const Vec& x, const Vec& y) {
		if (x.x != y.x || x.y != y.y || x.z != y.z)
			s << what << " (" << x.x << "," << x.y << "," << x.z << " vs "
			  << y.x << "," << y.y << "," << y.z << "); ";
	};
	auto fnNum = [&](const char* what, double x, double y) {
		if (x != y)
			s << what << " (" << x << " vs " << y << "); ";
	};

	// tickCount is intentionally NOT compared: the engine owns it and the FFI has no
	// setter, so a rewound arena keeps counting forward. It drives episode
	// bookkeeping, not physics — see the ArenaSnapshot header note.
	fnVec("ball.pos", a.ball.pos, b.ball.pos);
	fnVec("ball.vel", a.ball.vel, b.ball.vel);
	fnVec("ball.angVel", a.ball.angVel, b.ball.angVel);

	if (a.cars.size() != b.cars.size()) {
		s << "car count (" << a.cars.size() << " vs " << b.cars.size() << "); ";
		return s.str();
	}
	for (size_t i = 0; i < a.cars.size(); i++) {
		const CarFacts x = Facts(a.cars[i].state);
		const CarFacts y = Facts(b.cars[i].state);
		std::string p = "car" + std::to_string(i) + ".";
		fnVec((p + "pos").c_str(), x.pos, y.pos);
		fnVec((p + "vel").c_str(), x.vel, y.vel);
		fnVec((p + "angVel").c_str(), x.angVel, y.angVel);
		fnVec((p + "rot.forward").c_str(), x.fwd, y.fwd);
		fnVec((p + "rot.up").c_str(), x.up, y.up);
		fnNum((p + "boost").c_str(), x.boost, y.boost);
		fnNum((p + "isOnGround").c_str(), x.isOnGround, y.isOnGround);
		fnNum((p + "hasJumped").c_str(), x.hasJumped, y.hasJumped);
		fnNum((p + "hasDoubleJumped").c_str(), x.hasDoubleJumped, y.hasDoubleJumped);
		fnNum((p + "hasFlipped").c_str(), x.hasFlipped, y.hasFlipped);
		fnNum((p + "jumpTime").c_str(), x.jumpTime, y.jumpTime);
		fnNum((p + "flipTime").c_str(), x.flipTime, y.flipTime);
		fnNum((p + "isFlipping").c_str(), x.isFlipping, y.isFlipping);
		fnNum((p + "isJumping").c_str(), x.isJumping, y.isJumping);
		fnNum((p + "airTime").c_str(), x.airTime, y.airTime);
		fnNum((p + "airTimeSinceJump").c_str(), x.airTimeSinceJump, y.airTimeSinceJump);
		fnNum((p + "isDemoed").c_str(), x.isDemoed, y.isDemoed);
		fnNum((p + "demoRespawnTimer").c_str(), x.demoRespawnTimer, y.demoRespawnTimer);
		fnNum((p + "handbrakeVal").c_str(), x.handbrakeVal, y.handbrakeVal);
		// The four the compat facade drops — the actual bug this suite exists for.
		fnNum((p + "timeSinceBoosted").c_str(), x.timeSinceBoosted, y.timeSinceBoosted);
		fnNum((p + "isBoosting").c_str(), x.isBoosting, y.isBoosting);
		fnNum((p + "boostingTime").c_str(), x.boostingTime, y.boostingTime);
		fnNum((p + "supersonicGraceTimer").c_str(), x.supersonicGraceTimer, y.supersonicGraceTimer);
		fnNum((p + "bumpCooldownTimer").c_str(), x.bumpCooldownTimer, y.bumpCooldownTimer);
	}

	if (a.pads.size() != b.pads.size()) {
		s << "pad count; ";
		return s.str();
	}
	for (size_t i = 0; i < a.pads.size(); i++) {
		std::string p = "pad" + std::to_string(i) + ".";
		fnNum((p + "isActive").c_str(), a.pads[i].isActive, b.pads[i].isActive);
		fnNum((p + "cooldown").c_str(), a.pads[i].cooldown, b.pads[i].cooldown);
	}

	return s.str();
}

// THE test the panel's rewind rests on: snapshot mid-play, run forward, rewind, run
// the identical inputs forward again, and require the same outcome.
TEST(VizSnapshot_RewindReproducesTheSameFuture) {
	EnsureRocketSimViz();
	Arena* arena = MakeArena(3);

	// Get well away from the kickoff pose first: cars airborne, mid-flip, boost
	// spent, pads on cooldown — the state that would expose a lossy snapshot.
	DriveArena(arena, /*seed=*/99, /*ticks=*/240);

	ArenaSnapshot mark = {};
	mark.CaptureFrom(arena);

	DriveArena(arena, /*seed=*/7, /*ticks=*/300);
	ArenaSnapshot firstRun = {};
	firstRun.CaptureFrom(arena);

	CHECK(mark.ApplyTo(arena));

	DriveArena(arena, /*seed=*/7, /*ticks=*/300);
	ArenaSnapshot secondRun = {};
	secondRun.CaptureFrom(arena);

	std::string diff = DiffSnapshots(firstRun, secondRun);
	if (!diff.empty())
		throw TestFailure{ "rewind diverged after 300 ticks: " + diff };

	delete arena;
}

// Contacts (car-ball, car-car, demos) reach engine state that plain physics does not:
// the contact tracker, bump cooldowns, and — on a demo respawn — the arena RNG. A
// rewind over a real play crosses all of them, so pin a long window that actually
// contains them rather than trusting the flailing-cars case above.
TEST(VizSnapshot_RewindIsExactAcrossContactsAndDemos) {
	EnsureRocketSimViz();
	Arena* arena = MakeArena(3);

	// Fire cars point-blank into a resting ball so a hard touch is certain: aiming them
	// from a distance and letting random inputs take over mostly misses.
	{
		BallState bs = arena->ball->GetState();
		bs.pos = Vec(0, 0, RLConst::BALL_REST_Z);
		bs.vel = Vec(0, 0, 0);
		bs.angVel = Vec(0, 0, 0);
		arena->ball->SetState(bs);

		int i = 0;
		for (Car* car : arena->_cars) {
			CarState s = car->GetState();
			float theta = (float)i * (2.f * (float)M_PI / 6.f);
			Vec dir = Vec(cosf(theta), sinf(theta), 0);
			s.pos = bs.pos + dir * 350.f;
			s.pos.z = 17.f;
			s.vel = dir * -1800.f; // straight at the ball
			s.rotMat = RotMat::LookAt(dir * -1, Vec(0, 0, 1));
			s.boost = 100;
			s.isOnGround = true;
			car->SetState(s);
			i++;
		}
	}
	// Full throttle + boost into the pile: no randomness in the contact phase.
	for (int t = 0; t < 40; t++) {
		for (Car* car : arena->_cars) {
			CarControls c = {};
			c.throttle = 1;
			c.boost = true;
			car->controls = c;
		}
		arena->Step(1);
	}
	DriveArena(arena, /*seed=*/1234567, /*ticks=*/110);

	// Bump-demos need a supersonic hit landing just right, which random inputs do not
	// reliably produce (an earlier version of this test never demoed anyone and passed
	// vacuously). Put two cars into the demoed state directly with staggered timers, so
	// their RNG-driven respawns land INSIDE the replay window below.
	{
		float timer = 0.6f;
		for (int i = 0; i < 2; i++) {
			CarState s = arena->_cars[i]->GetState();
			s.isDemoed = true;
			s.demoRespawnTimer = timer;
			arena->_cars[i]->SetState(s);
			timer += 0.5f;
		}
	}

	ArenaSnapshot mark = {};
	mark.CaptureFrom(arena);

	// 10 seconds — long enough for demoed cars to respawn inside the replay window.
	const int TICKS = 1200;
	DriveWitness witness = {};
	DriveArena(arena, /*seed=*/2024, /*ticks=*/TICKS, &witness);
	ArenaSnapshot firstRun = {};
	firstRun.CaptureFrom(arena);

	CHECK(mark.ApplyTo(arena));
	DriveArena(arena, /*seed=*/2024, /*ticks=*/TICKS);
	ArenaSnapshot secondRun = {};
	secondRun.CaptureFrom(arena);

	// Both paths must actually have been crossed, or this proves nothing about either.
	CHECK(witness.maxBallSpeed > 300.f);
	CHECK(witness.demoedCarTicks > 0);

	std::string diff = DiffSnapshots(firstRun, secondRun);
	if (!diff.empty())
		throw TestFailure{ "rewind diverged across contacts: " + diff };

	delete arena;
}

// A restore must land exactly on the recorded state, not merely converge to it.
TEST(VizSnapshot_RestoreIsExactImmediately) {
	EnsureRocketSimViz();
	Arena* arena = MakeArena(2);
	DriveArena(arena, /*seed=*/5150, /*ticks=*/180);

	ArenaSnapshot mark = {};
	mark.CaptureFrom(arena);

	DriveArena(arena, /*seed=*/4, /*ticks=*/60);
	CHECK(mark.ApplyTo(arena));

	ArenaSnapshot readBack = {};
	readBack.CaptureFrom(arena);

	std::string diff = DiffSnapshots(mark, readBack);
	if (!diff.empty())
		throw TestFailure{ "snapshot did not round-trip: " + diff };

	delete arena;
}

// Boost pads are engine state the ball and cars don't carry: rewinding into a window
// where a big pad was still on cooldown has to give that cooldown back.
TEST(VizSnapshot_RestoresBoostPadCooldowns) {
	EnsureRocketSimViz();
	Arena* arena = MakeArena(3);

	// Knock pads down explicitly rather than driving randomly and hoping a car rolls
	// over one: an earlier version of this test drove for 400 ticks, picked up nothing,
	// and would have passed vacuously forever.
	CHECK(arena->_boostPads.size() > 4);
	for (size_t i = 0; i < arena->_boostPads.size(); i += 3) {
		BoostPadState ps = {};
		ps.isActive = false;
		ps.cooldown = 1.5f + 0.25f * (float)i;
		arena->_boostPads[i]->SetState(ps);
	}

	ArenaSnapshot mark = {};
	mark.CaptureFrom(arena);

	int inactiveAtMark = 0;
	for (const auto& pad : mark.pads)
		if (!pad.isActive)
			inactiveAtMark++;
	CHECK(inactiveAtMark > 0);

	// Let every cooldown expire, then rewind back into the middle of them.
	DriveArena(arena, /*seed=*/8, /*ticks=*/1500);
	for (const auto& pad : arena->_boostPads)
		CHECK(pad->GetState().isActive);

	CHECK(mark.ApplyTo(arena));

	ArenaSnapshot readBack = {};
	readBack.CaptureFrom(arena);
	for (size_t i = 0; i < mark.pads.size(); i++) {
		CHECK_EQ(readBack.pads[i].isActive, mark.pads[i].isActive);
		CHECK_NEAR(readBack.pads[i].cooldown, mark.pads[i].cooldown, 1e-6f);
	}

	delete arena;
}

TEST(VizSnapshot_RingEvictsOldestAndFindsByFrameId) {
	EnsureRocketSimViz();
	Arena* arena = MakeArena(1);

	SnapshotRing ring(/*capacity=*/8);
	CHECK(ring.Empty());
	CHECK(ring.Find(0) == NULL);

	std::vector<int64_t> ids;
	for (int i = 0; i < 20; i++) {
		DriveArena(arena, /*seed=*/(uint64_t)i + 1, /*ticks=*/8);
		ids.push_back(ring.Push(arena));
	}

	CHECK_EQ(ring.frames.size(), (size_t)8);
	CHECK_EQ(ring.NewestId(), ids.back());
	CHECK_EQ(ring.OldestId(), ids[12]);

	// Rotated past: not found rather than silently clamped to the oldest kept frame.
	CHECK(ring.Find(ids[0]) == NULL);

	const ArenaSnapshot* found = ring.Find(ids[15]);
	CHECK(found != NULL);
	CHECK_EQ(found->frameId, ids[15]);

	// Ahead of the newest resolves to the newest (the browser can ask for "now").
	const ArenaSnapshot* latest = ring.Find(ids.back() + 100);
	CHECK(latest != NULL);
	CHECK_EQ(latest->frameId, ring.NewestId());

	delete arena;
}

TEST(VizSnapshot_TruncateDropsTheDiscardedFuture) {
	EnsureRocketSimViz();
	Arena* arena = MakeArena(1);

	SnapshotRing ring(/*capacity=*/64);
	std::vector<int64_t> ids;
	for (int i = 0; i < 10; i++) {
		DriveArena(arena, /*seed=*/(uint64_t)i + 100, /*ticks=*/8);
		ids.push_back(ring.Push(arena));
	}

	ring.TruncateAfter(ids[4]);
	CHECK_EQ(ring.frames.size(), (size_t)5);
	CHECK_EQ(ring.NewestId(), ids[4]);
	CHECK(ring.Find(ids[9]) != NULL);            // clamps to newest kept
	CHECK_EQ(ring.Find(ids[9])->frameId, ids[4]);

	// Branching keeps ids increasing, so a stale browser reference to a discarded
	// frame can never be mistaken for the new frame that took its place.
	DriveArena(arena, /*seed=*/999, /*ticks=*/8);
	int64_t afterBranch = ring.Push(arena);
	CHECK_EQ(afterBranch, ids[4] + 1);
	CHECK(afterBranch < ids[9]);

	delete arena;
}

// Regression: truncating to a frame the ring doesn't hold used to pop every frame and
// reset ids to zero. It was reachable from the panel — resuming play after a seek
// truncated at the "live" sentinel (-1) and silently destroyed the whole rewind
// history mid-session, which read as the viewer having restarted.
TEST(VizSnapshot_TruncateRefusesFramesTheRingDoesNotHold) {
	EnsureRocketSimViz();
	Arena* arena = MakeArena(1);

	SnapshotRing ring(/*capacity=*/4);
	std::vector<int64_t> ids;
	for (int i = 0; i < 10; i++) {
		DriveArena(arena, /*seed=*/(uint64_t)i + 7, /*ticks=*/4);
		ids.push_back(ring.Push(arena));
	}
	const size_t sizeBefore = ring.frames.size();
	const int64_t newestBefore = ring.NewestId();

	ring.TruncateAfter(-1);            // the "live" sentinel
	CHECK_EQ(ring.frames.size(), sizeBefore);
	CHECK_EQ(ring.NewestId(), newestBefore);

	ring.TruncateAfter(ids[0]);        // rotated out of a capacity-4 ring long ago
	CHECK_EQ(ring.frames.size(), sizeBefore);
	CHECK_EQ(ring.NewestId(), newestBefore);

	// Ids must keep climbing afterwards, not restart from zero.
	DriveArena(arena, /*seed=*/77, /*ticks=*/4);
	CHECK_EQ(ring.Push(arena), newestBefore + 1);

	delete arena;
}
