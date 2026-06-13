#include "FrontierState.h"

#include "../Math.h"
#include "../CommonValues.h"

using namespace RLGC;

RLGC::FrontierStateBuffer::FrontierStateBuffer(int capacity) : capacity((size_t)RS_MAX(capacity, 1)) {
	ring.resize(this->capacity);
}

void RLGC::FrontierStateBuffer::Insert(const FrontierSnapshot& snap) {
	std::lock_guard<std::mutex> lock(mutex);
	ring[writeIdx] = snap;
	writeIdx = (writeIdx + 1) % capacity;
	count = RS_MIN(count + 1, capacity);
}

bool RLGC::FrontierStateBuffer::Sample(FrontierSnapshot& out) const {
	std::lock_guard<std::mutex> lock(mutex);
	if (count == 0)
		return false;
	out = ring[RocketSim::Math::RandInt(0, (int)count)];
	return true;
}

size_t RLGC::FrontierStateBuffer::Size() const {
	std::lock_guard<std::mutex> lock(mutex);
	return count;
}

bool RLGC::FrontierStateBuffer::PassesSanityFilter(const FrontierSnapshot& snap) {
	using namespace CommonValues;

	auto fnFinite = [](const Vec& v) {
		return isfinite(v.x) && isfinite(v.y) && isfinite(v.z);
	};

	// Ball: in play (not inside a goal), inside the arena shell, sane speed
	const BallState& bs = snap.ball;
	if (!fnFinite(bs.pos) || !fnFinite(bs.vel) || !fnFinite(bs.angVel))
		return false;
	if (abs(bs.pos.x) > SIDE_WALL_X - BALL_RADIUS || abs(bs.pos.y) > BACK_WALL_Y - BALL_RADIUS)
		return false;
	if (bs.pos.z < BALL_RADIUS * 0.9f || bs.pos.z > CEILING_Z - BALL_RADIUS)
		return false;
	if (bs.vel.Length() > BALL_MAX_SPEED)
		return false;

	// Cars: in bounds at sane velocity (demoed cars were already skipped at capture)
	for (const FrontierCarSnapshot& cs : snap.cars) {
		if (!fnFinite(cs.pos) || !fnFinite(cs.vel) || !fnFinite(cs.angVel))
			return false;
		if (abs(cs.pos.x) > SIDE_WALL_X || abs(cs.pos.y) > BACK_WALL_Y)
			return false;
		if (cs.pos.z < 0 || cs.pos.z > CEILING_Z)
			return false;
		if (cs.vel.Length() > CAR_MAX_SPEED * 1.05f)
			return false;
		if (cs.angVel.Length() > CAR_MAX_ANG_VEL * 1.05f)
			return false;
	}

	return true;
}

void RLGC::FrontierStateBuffer::TagArenaReset(const Arena* arena, uint8_t tag) {
	std::lock_guard<std::mutex> lock(mutex);
	arenaTags[arena] = tag;
}

uint8_t RLGC::FrontierStateBuffer::GetArenaTag(const Arena* arena) const {
	std::lock_guard<std::mutex> lock(mutex);
	auto itr = arenaTags.find(arena);
	return (itr != arenaTags.end()) ? itr->second : (uint8_t)TAG_NONE;
}

// BOOST_LOCATIONS index -> arena pad index, same matching as GameState.cpp's
// _BuildBoostPadIndexMap (which is file-static there, so rebuilt here once).
// Pad layout is identical across same-mode arenas, so one global map suffices.
static int g_PadIndexMap[CommonValues::BOOST_LOCATIONS_AMOUNT] = {};
static bool g_PadIndexMapBuilt = false;
static std::mutex g_PadIndexMapMutex = {};
static bool _BuildPadIndexMap(Arena* arena) {
	if (arena->_boostPads.size() != CommonValues::BOOST_LOCATIONS_AMOUNT)
		return false;

	for (int i = 0; i < CommonValues::BOOST_LOCATIONS_AMOUNT; i++) {
		Vec targetPos = CommonValues::BOOST_LOCATIONS[i];
		bool found = false;
		for (int j = 0; j < (int)arena->_boostPads.size(); j++) {
			if (arena->_boostPads[j]->config.pos.DistSq2D(targetPos) < 10) {
				g_PadIndexMap[i] = j;
				found = true;
				break;
			}
		}
		if (!found)
			return false;
	}
	return true;
}

void RLGC::FrontierState::ResetArena(Arena* arena) {

	auto fnFallback = [&](void) {
		fallback->ResetArena(arena);
		if (buffer) {
			buffer->fallbackResets.fetch_add(1, std::memory_order_relaxed);
			buffer->TagArenaReset(arena, FrontierStateBuffer::TAG_FALLBACK);
		}
	};

	if (!buffer || (int)buffer->Size() < minFill) {
		fnFallback();
		return;
	}

	// Sample until a snapshot is compatible with this arena (mode + per-team car
	// counts); snapshots from a different phase's mode mix just get re-rolled.
	constexpr int MAX_SAMPLE_ATTEMPTS = 4;
	FrontierSnapshot snap;
	bool found = false;
	for (int attempt = 0; attempt < MAX_SAMPLE_ATTEMPTS && !found; attempt++) {
		if (!buffer->Sample(snap))
			break;

		if (snap.gameMode != (int)arena->gameMode)
			continue;

		int snapTeamCounts[2] = { 0, 0 };
		for (auto& car : snap.cars)
			snapTeamCounts[(int)car.team]++;
		int arenaTeamCounts[2] = { 0, 0 };
		for (Car* car : arena->_cars)
			arenaTeamCounts[(int)car->team]++;

		found = (snapTeamCounts[0] == arenaTeamCounts[0]) && (snapTeamCounts[1] == arenaTeamCounts[1]);
	}

	if (!found) {
		fnFallback();
		return;
	}

	// Clean pad/car baseline first, exactly like RandomState
	arena->ResetToRandomKickoff();

	arena->ball->SetState(snap.ball);

	// Assign snapshot cars to arena cars by team, in order. Within a team the pairing is
	// arbitrary (arena->_cars iteration order differs between arenas), which is fine — the
	// curriculum only needs the team formation, not car identity. Fresh CarState like
	// RandomState: jump/flip/contact state intentionally resets.
	int teamCursor[2] = { 0, 0 };
	for (Car* car : arena->_cars) {
		int teamIdx = (int)car->team;
		int snapIdx = -1;
		for (int i = teamCursor[teamIdx]; i < (int)snap.cars.size(); i++) {
			if ((int)snap.cars[i].team == teamIdx) {
				snapIdx = i;
				break;
			}
		}
		if (snapIdx == -1) // Can't happen after the team-count check; be safe anyway
			continue;
		teamCursor[teamIdx] = snapIdx + 1;

		const FrontierCarSnapshot& scs = snap.cars[snapIdx];
		CarState cs = {};
		cs.pos = scs.pos;
		cs.rotMat = scs.rotMat;
		cs.vel = scs.vel;
		cs.angVel = scs.angVel;
		cs.boost = scs.boost;
		car->SetState(cs);
	}

	// Restore boost pads (BOOST_LOCATIONS order -> arena pad order)
	if (!snap.padsActive.empty() && snap.padsActive.size() == snap.padTimers.size()) {
		if (!g_PadIndexMapBuilt) {
			std::lock_guard<std::mutex> lock(g_PadIndexMapMutex);
			if (!g_PadIndexMapBuilt)
				g_PadIndexMapBuilt = _BuildPadIndexMap(arena);
		}
		if (g_PadIndexMapBuilt &&
			snap.padsActive.size() == (size_t)CommonValues::BOOST_LOCATIONS_AMOUNT &&
			arena->_boostPads.size() == (size_t)CommonValues::BOOST_LOCATIONS_AMOUNT) {
			for (int i = 0; i < CommonValues::BOOST_LOCATIONS_AMOUNT; i++) {
				BoostPadState ps = {};
				ps.isActive = snap.padsActive[i];
				ps.cooldown = snap.padTimers[i];
				arena->_boostPads[g_PadIndexMap[i]]->SetState(ps);
			}
		}
	}

	buffer->frontierResets.fetch_add(1, std::memory_order_relaxed);
	buffer->TagArenaReset(arena, FrontierStateBuffer::TAG_FRONTIER);
}
