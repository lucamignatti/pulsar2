#include "TestFramework.h"

#include <RLGymCPP/StateSetters/FrontierDrillState.h>
#include <RLGymCPP/StateSetters/KickoffState.h>

#include <filesystem>

using namespace RLGC;

// FrontierPool + FrontierDrillState (STEERING_ROADMAP phase 3): banked frontier states
// reset practice arenas into (optionally perturbed) copies; empty/stale pools and
// mode mismatches fall back to the wrapped setter.

static void EnsureRocketSimFrontier() {
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
	inited = true; // another test file already initialized
}

static FrontierPool::Entry MakeEntry() {
	FrontierPool::Entry e;
	e.ball.pos = Vec(500, -1200, 800);
	e.ball.vel = Vec(300, 400, -100);
	e.ball.angVel = Vec(1, 0, 0);
	FrontierPool::CarSpawn blue = {};
	blue.pos = Vec(-900, -2500, 17);
	blue.vel = Vec(200, 800, 0);
	blue.angVel = Vec(0, 0, 0.2f);
	blue.forward = Vec(0.6f, 0.8f, 0).Normalized();
	blue.up = Vec(0, 0, 1);
	blue.boost = 63;
	FrontierPool::CarSpawn orange = blue;
	orange.pos = Vec(1100, 2300, 17);
	orange.boost = 27;
	e.blueCars = { blue };
	e.orangeCars = { orange };
	return e;
}

TEST(FrontierPool_sample_and_staleness) {
	FrontierPool pool;
	pool.Advance(10);
	pool.Fill(0, { MakeEntry() });

	FrontierPool::Entry out;
	CHECK(pool.Sample(0, out));
	CHECK_NEAR(out.ball.pos.x, 500.f, 1e-4f);
	CHECK(!pool.Sample(1, out)); // no 2v2 entries

	pool.Advance(11);
	CHECK(pool.Sample(0, out)); // within maxAgeIters (2)
	pool.Advance(13);
	CHECK(!pool.Sample(0, out)); // stale now
}

TEST(FrontierDrill_reset_reproduces_entry) {
	EnsureRocketSimFrontier();
	auto arena = Arena::Create(GameMode::SOCCAR);
	Car* blue = arena->AddCar(Team::BLUE);
	Car* orange = arena->AddCar(Team::ORANGE);

	auto pool = std::make_shared<FrontierPool>();
	pool->Advance(1);
	pool->Fill(0, { MakeEntry() });

	// useFrac 1 + zero noise: the reset must reproduce the entry (modulo clamps)
	FrontierDrillState setter(pool, new KickoffState(), 1.0f, 0.f, 0.f);
	setter.ResetArena(arena);

	auto bs = arena->ball->GetState();
	CHECK_NEAR(bs.pos.x, 500.f, 1.f);
	CHECK_NEAR(bs.pos.y, -1200.f, 1.f);
	CHECK_NEAR(bs.pos.z, 800.f, 1.f);

	auto cb = blue->GetState();
	CHECK_NEAR(cb.pos.x, -900.f, 1.f);
	CHECK_NEAR(cb.boost, 63.f, 0.5f);
	CHECK(cb.rotMat.forward.Dot(Vec(0.6f, 0.8f, 0).Normalized()) > 0.999f);
	auto co = orange->GetState();
	CHECK_NEAR(co.pos.x, 1100.f, 1.f);
	CHECK_NEAR(co.boost, 27.f, 0.5f);

	delete arena;
}

TEST(FrontierDrill_falls_back_when_pool_stale_or_wrong_mode) {
	EnsureRocketSimFrontier();
	auto arena = Arena::Create(GameMode::SOCCAR);
	arena->AddCar(Team::BLUE);
	arena->AddCar(Team::ORANGE);

	auto pool = std::make_shared<FrontierPool>();
	pool->Advance(1);
	pool->Fill(0, { MakeEntry() });
	pool->Advance(10); // stale

	FrontierDrillState setter(pool, new KickoffState(), 1.0f, 0.f, 0.f);
	setter.ResetArena(arena);
	// KickoffState fallback: ball at center
	auto bs = arena->ball->GetState();
	CHECK_NEAR(bs.pos.x, 0.f, 1.f);
	CHECK_NEAR(bs.pos.y, 0.f, 1.f);

	// 2v2 arena vs 1v1-only entries: also falls back
	auto arena2 = Arena::Create(GameMode::SOCCAR);
	for (int i = 0; i < 2; i++) {
		arena2->AddCar(Team::BLUE);
		arena2->AddCar(Team::ORANGE);
	}
	pool->Advance(11);
	pool->Fill(0, { MakeEntry() });
	FrontierDrillState setter2(pool, new KickoffState(), 1.0f, 0.f, 0.f);
	setter2.ResetArena(arena2);
	auto bs2 = arena2->ball->GetState();
	CHECK_NEAR(bs2.pos.x, 0.f, 1.f);

	delete arena;
	delete arena2;
}
