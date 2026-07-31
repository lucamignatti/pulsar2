#include "VizControl.h"

#include <nlohmann/json.hpp>

#include <algorithm>

#ifdef _MSC_VER
#include <winsock2.h>
#pragma comment(lib, "ws2_32.lib")
#define RG_CLOSE_SOCKET closesocket
#else
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#define RG_CLOSE_SOCKET close
#endif

using namespace GGL;

// Engine-native car state access. Under v3 this bypasses Car::GetState/SetState,
// which cannot round-trip four simulated fields (see the VizCarState note in the
// header); under v2 the facade type IS the engine type, so the plain calls are exact.
static void CaptureCar(Car* car, VizCarState& out, std::vector<uint8_t>& extraOut) {
#ifdef RG_ROCKETSIM_V3
	rsf_arena_get_car_state(car->_arena->_rs, car->_idx, &out);
	extraOut.resize(rsf_car_extra_state_size());
	rsf_arena_get_car_extra_state(car->_arena->_rs, car->_idx, extraOut.data());
#else
	out = car->GetState();
	extraOut.clear();
#endif
}

static void ApplyCar(Car* car, const VizCarState& in, const std::vector<uint8_t>& extra) {
#ifdef RG_ROCKETSIM_V3
	rsf_arena_set_car_state(car->_arena->_rs, car->_idx, &in);
	// Order matters: set_car_state zeroes the pending bump impulse, so the extras must
	// land after it, not before.
	if (extra.size() == rsf_car_extra_state_size())
		rsf_arena_set_car_extra_state(car->_arena->_rs, car->_idx, extra.data());
#else
	car->SetState(in);
	(void)extra;
#endif
}

void ArenaSnapshot::CaptureFrom(Arena* arena) {
	tickCount = arena->tickCount;
	ball = arena->ball->GetState();
#ifdef RG_ROCKETSIM_V3
	rngState = rsf_arena_get_rng_state(arena->_rs);
#endif

	cars.clear();
	cars.reserve(arena->_cars.size());
	for (Car* car : arena->_cars) {
		CarSnapshot cs = {};
		cs.id = car->id;
		cs.team = car->team;
		CaptureCar(car, cs.state, cs.engineExtra);
		cs.controls = car->controls;
		cs.ballHitInfo = car->_ballHitInfo;
		cars.push_back(cs);
	}

	pads.clear();
	pads.reserve(arena->_boostPads.size());
	for (BoostPad* pad : arena->_boostPads) {
		BoostPadState ps = pad->GetState();
		PadSnapshot snap = {};
		snap.isActive = ps.isActive;
		snap.cooldown = ps.cooldown;
		snap.lockedCarId = ps.curLockedCar ? ps.curLockedCar->id : 0;
		pads.push_back(snap);
	}
}

bool ArenaSnapshot::ApplyTo(Arena* arena) const {
	if (cars.size() != arena->_cars.size() || pads.size() != arena->_boostPads.size())
		return false;

	// tickCount is deliberately not restored — the engine owns it and exposes no
	// setter (see the header). Everything below is authoritative.
	arena->ball->SetState(ball);
#ifdef RG_ROCKETSIM_V3
	rsf_arena_set_rng_state(arena->_rs, rngState);
#endif

	for (size_t i = 0; i < cars.size(); i++) {
		Car* car = arena->_cars[i];
		// Cars are never added or removed while the viewer runs, so index order is
		// stable; the id check is a cheap guard against that assumption changing.
		if (car->id != cars[i].id)
			return false;
		// Controls first: the v2 path's SetState folds the car's *pending* controls
		// into the engine state, so assigning them afterwards would apply the snapshot
		// on top of whatever action the pre-seek policy had queued.
		car->controls = cars[i].controls;
		ApplyCar(car, cars[i].state, cars[i].engineExtra);
		car->_ballHitInfo = cars[i].ballHitInfo;
	}

	for (size_t i = 0; i < pads.size(); i++) {
		BoostPadState ps = {};
		ps.isActive = pads[i].isActive;
		ps.cooldown = pads[i].cooldown;
		ps.curLockedCar = NULL;
		if (pads[i].lockedCarId) {
			for (Car* car : arena->_cars) {
				if (car->id == pads[i].lockedCarId) {
					ps.curLockedCar = car;
					break;
				}
			}
		}
		arena->_boostPads[i]->SetState(ps);
	}

	return true;
}

int64_t SnapshotRing::Push(Arena* arena) {
	// Reuse the evicted frame's storage rather than allocating a fresh one: at 15 Hz
	// with a full ring this runs forever, and the car/pad vectors are fixed-size.
	ArenaSnapshot recycled = {};
	if (frames.size() >= capacity) {
		recycled = std::move(frames.front());
		frames.pop_front();
	}

	recycled.frameId = nextFrameId++;
	recycled.CaptureFrom(arena);
	frames.push_back(std::move(recycled));
	return frames.back().frameId;
}

const ArenaSnapshot* SnapshotRing::Find(int64_t frameId) const {
	if (frames.empty() || frameId < frames.front().frameId)
		return NULL;

	// Ids are assigned in push order and frames are never reordered, so the deque is
	// sorted by frameId and partition_point gives the last frame at or before it.
	auto it = std::partition_point(frames.begin(), frames.end(),
		[frameId](const ArenaSnapshot& f) { return f.frameId <= frameId; });
	return &*(it - 1);
}

bool SnapshotRing::ReplaceFrame(int64_t frameId, Arena* arena) {
	// Located by binary search on the iterator rather than via Find(): frames live in a
	// deque, so differencing an element pointer against &frames.front() to recover an
	// index is not valid pointer arithmetic.
	auto it = std::partition_point(frames.begin(), frames.end(),
		[frameId](const ArenaSnapshot& f) { return f.frameId < frameId; });
	if (it == frames.end() || it->frameId != frameId)
		return false;

	const int64_t keepId = it->frameId; // CaptureFrom overwrites everything but the id
	it->CaptureFrom(arena);
	it->frameId = keepId;
	return true;
}

void SnapshotRing::TruncateAfter(int64_t frameId) {
	// Branching from a frame the ring no longer holds would pop every frame and leave
	// an empty history — a silent, total loss of rewind depth from what looks like a
	// bookkeeping call. There is no meaningful truncation to perform in that case, so
	// refuse rather than approximate.
	if (frames.empty() || frameId < frames.front().frameId)
		return;

	while (!frames.empty() && frames.back().frameId > frameId)
		frames.pop_back();

	// Ids must keep increasing across a branch, so the browser can't confuse a frame
	// from the discarded future with the one replacing it.
	nextFrameId = frames.empty() ? 0 : frames.back().frameId + 1;
}

/////////////// VizControl ///////////////

VizControl::VizControl(int cmdPort, size_t historyFrames) : history(historyFrames) {
#ifdef _MSC_VER
	{
		static bool wsaStarted = false;
		if (!wsaStarted) {
			WSADATA wsa;
			WSAStartup(MAKEWORD(2, 2), &wsa);
			wsaStarted = true;
		}
	}
#endif

	sock = (int)socket(AF_INET, SOCK_DGRAM, 0);
	if (sock < 0) {
		RG_LOG("[viz] WARNING: could not create the control socket; the viewer will run uncontrolled.");
		return;
	}

	// Non-blocking: the render loop polls once per step and must never stall waiting
	// for a browser that may not even be open.
#ifdef _MSC_VER
	u_long nonBlocking = 1;
	ioctlsocket(sock, FIONBIO, &nonBlocking);
#else
	fcntl(sock, F_SETFL, fcntl(sock, F_GETFL, 0) | O_NONBLOCK);
#endif

	sockaddr_in addr = {};
	addr.sin_family = AF_INET;
	addr.sin_port = htons((uint16_t)cmdPort);
	// Loopback only. The viz stack is reachable off-box through the tailnet proxy, but
	// nothing that can teleport cars should be listening on a public interface.
	addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

	if (bind(sock, (sockaddr*)&addr, sizeof(addr)) < 0) {
		// Almost always a second viewer already holding the port. Not fatal: this one
		// still renders, it just isn't the one the panel drives.
		RG_LOG("[viz] WARNING: control port " << cmdPort << " is unavailable (another viewer?); "
			<< "this viewer will run uncontrolled.");
		RG_CLOSE_SOCKET(sock);
		sock = -1;
		return;
	}

	RG_LOG("[viz] control channel listening on udp://127.0.0.1:" << cmdPort);
}

VizControl::~VizControl() {
	if (sock >= 0)
		RG_CLOSE_SOCKET(sock);
}

void VizControl::Poll() {
	if (sock < 0)
		return;

	// Datagrams are small JSON objects; anything larger is not ours.
	char buf[8192];
	for (int drained = 0; drained < 256; drained++) {
		auto received = recvfrom(sock, buf, sizeof(buf) - 1, 0, NULL, NULL);
		if (received <= 0)
			break; // EWOULDBLOCK on an empty queue

		buf[received] = '\0';
		HandleCommand(std::string(buf, (size_t)received));
	}
}

void VizControl::HandleCommand(const std::string& text) {
	nlohmann::json j;
	try {
		j = nlohmann::json::parse(text);
	} catch (const std::exception&) {
		return; // not JSON, not ours
	}
	if (!j.is_object() || !j.contains("cmd") || !j["cmd"].is_string())
		return;

	const std::string cmd = j["cmd"].get<std::string>();

	// Small helper: a number field with a fallback, tolerant of the wrong type
	// arriving (the page is hand-written JS, and a NaN slider must not wedge the sim).
	auto fnNum = [&](const char* key, double fallback) -> double {
		if (!j.contains(key) || !j[key].is_number())
			return fallback;
		double v = j[key].get<double>();
		return std::isfinite(v) ? v : fallback;
	};

	if (cmd == "pause") {
		paused = true;
	} else if (cmd == "play") {
		paused = false;
		stepsQueued = 0;
		viewFrameId = -1;
	} else if (cmd == "step") {
		// Stepping implies paused: the button means "advance exactly this much".
		paused = true;
		// Capped: a key held down, or a page that got excited, must not be able to bank
		// an unbounded run of steps that the viewer then works through long after the
		// user stopped asking for them.
		constexpr int MAX_QUEUED_STEPS = 240; // 16s at 15 Hz
		stepsQueued = RS_MIN(MAX_QUEUED_STEPS, stepsQueued + RS_MAX(1, (int)fnNum("n", 1)));
	} else if (cmd == "agent") {
		// {"cmd":"agent","team":0|1,"spec":"..."} — sets ONE side. The panel sends one of
		// these per dropdown, so the two teams are configured independently.
		Team team = (fnNum("team", 1) != 0) ? Team::ORANGE : Team::BLUE;
		std::string want;
		if (j.contains("spec") && j["spec"].is_string())
			want = j["spec"].get<std::string>();

		// This arrives from a browser and is about to be joined onto a filesystem path
		// (and, for a bot, handed to fork/exec). Accept only the shapes the panel offers.
		if (IsRLBot(want)) {
			std::string cfg = RLBotConfig(want);
			if (std::find(availableBots.begin(), availableBots.end(), cfg) == availableBots.end()) {
				RG_LOG("[viz] refusing unknown bot config: " << cfg);
				opponentError = "unknown bot config";
				return;
			}
			// One external agent at a time; selecting a bot for one side clears any bot
			// on the other rather than silently running two fleets.
			Team other = (team == Team::BLUE) ? Team::ORANGE : Team::BLUE;
			std::string& otherSpec = (other == Team::BLUE) ? blueAgent : orangeAgent;
			if (IsRLBot(otherSpec))
				otherSpec.clear();
		} else if (!want.empty()) {
			const std::string refPrefix = "policy_versions/";
			std::string leaf = want;
			if (leaf.rfind(refPrefix, 0) == 0)
				leaf = leaf.substr(refPrefix.size());
			bool ok = true;
			for (char c : leaf)
				ok &= (isalnum((unsigned char)c) || c == '_' || c == '-');
			if (!ok || leaf.empty()) {
				RG_LOG("[viz] refusing agent path: " << want);
				opponentError = "rejected path";
				return;
			}
		}

		std::string& slot = (team == Team::BLUE) ? blueAgent : orangeAgent;
		if (slot != want) {
			slot = want;
			opponentDirty = true;
		}
	} else if (cmd == "reset_record") {
		ResetRecord();
	} else if (cmd == "deterministic") {
		deterministic = !j.contains("value") || !j["value"].is_boolean() || j["value"].get<bool>();
	} else if (cmd == "speed") {
		speed = (float)RS_CLAMP(fnNum("value", 1.0), 0.05, 8.0);
	} else if (cmd == "set_ball" || cmd == "set_car") {
		// Editing is only meaningful on a parked sim — otherwise the next step
		// overwrites the edit before anyone sees it. Park rather than reject, so a drag
		// on a running viewer does the obvious thing instead of nothing.
		paused = true;

		PendingEdit edit = {};
		edit.isBall = (cmd == "set_ball");
		if (!edit.isBall)
			edit.carId = (uint32_t)fnNum("id", 0);

		auto fnVec = [&](const char* key, Vec& out) -> bool {
			if (!j.contains(key) || !j[key].is_array() || j[key].size() != 3)
				return false;
			double v[3];
			for (int i = 0; i < 3; i++) {
				if (!j[key][i].is_number())
					return false;
				v[i] = j[key][i].get<double>();
				if (!std::isfinite(v[i]))
					return false;
			}
			out = Vec((float)v[0], (float)v[1], (float)v[2]);
			return true;
		};

		edit.hasPos = fnVec("pos", edit.pos);
		edit.hasVel = fnVec("vel", edit.vel);
		edit.hasAngVel = fnVec("angVel", edit.angVel);
		// A rotation needs BOTH basis vectors to be meaningful; half of one would
		// produce a degenerate matrix the physics would then have to live with.
		Vec fwd = {}, up = {};
		if (fnVec("forward", fwd) && fnVec("up", up)) {
			edit.hasRot = true;
			edit.forward = fwd;
			edit.up = up;
		}
		if (j.contains("boost") && j["boost"].is_number()) {
			edit.hasBoost = true;
			edit.boost = (float)RS_CLAMP(fnNum("boost", 0), 0, 100);
		}
		if (j.contains("demoed") && j["demoed"].is_boolean()) {
			edit.hasDemoed = true;
			edit.demoed = j["demoed"].get<bool>();
		}

		pendingEdits.push_back(edit);
	} else if (cmd == "seek") {
		// Scrubbing is inspection, so it always parks the sim; the page has to ask for
		// play explicitly to commit to a branch.
		paused = true;
		stepsQueued = 0;
		seekPending = true;
		seekTarget = (int64_t)fnNum("frame", (double)history.NewestId());
	}
}

bool VizControl::ConsumeHalt() {
	if (!paused)
		return false;
	if (stepsQueued > 0) {
		stepsQueued--;
		return false;
	}
	return true;
}

bool VizControl::ApplySeek(Arena* arena) {
	if (!seekPending)
		return false;
	seekPending = false;

	const ArenaSnapshot* snap = history.Find(seekTarget);
	if (!snap)
		return false; // rotated out of the ring; leave the arena alone

	if (!snap->ApplyTo(arena))
		return false;

	viewFrameId = snap->frameId;
	branchFrom = snap->frameId;
	branchPending = true;
	return true;
}

void VizControl::TrackScore(const RLGC::GameState& state) {
	// goalScored stays true for as long as the ball sits in the net, so count only the
	// transition — otherwise one goal reads as dozens.
	if (!state.goalScored) {
		goalLatched = false;
		return;
	}
	if (goalLatched)
		return;
	goalLatched = true;

	// RS_TEAM_FROM_Y: negative y is BLUE's half, so a ball in the +y net was put there
	// by BLUE.
	Record& rec = records[MatchupKey()];
	if (state.ball.pos.y > 0)
		rec.blueGoals++;
	else
		rec.orangeGoals++;
}

// The tally belongs to the MATCHUP, so changing either side starts a fresh scoreline
// instead of pooling results from different opponents.
std::string VizControl::MatchupKey() const {
	return blueAgent + "|" + orangeAgent;
}

void VizControl::ResetRecord() {
	records[MatchupKey()] = Record{};
}

bool VizControl::ApplyEdits(Arena* arena) {
	if (pendingEdits.empty())
		return false;

	for (const PendingEdit& edit : pendingEdits) {
		if (edit.isBall) {
			// Read-modify-write: only the named fields move.
			BallState bs = arena->ball->GetState();
			if (edit.hasPos) bs.pos = edit.pos;
			if (edit.hasVel) bs.vel = edit.vel;
			if (edit.hasAngVel) bs.angVel = edit.angVel;

			// The engine SLEEPS the ball whenever its linear and angular velocity are
			// both exactly zero, and re-evaluates that every tick from the velocity
			// alone (rocketsim sim/arena/base.rs, "Update ball activation"). So a ball
			// dragged into the air at rest does not fall — it hangs there until
			// something hits it, which is a baffling thing to watch and the first thing
			// anyone tries with this editor.
			//
			// Any nonzero velocity keeps it awake long enough for gravity to take over
			// from the next tick. It has to clear the engine's velocity QUANTIZER
			// though, which runs first and truncates: `(v * VEL_SCALE) as i32` with
			// VEL_SCALE 100, so anything under 0.01 uu/s becomes exactly zero and the
			// ball goes right back to sleep. 0.02 uu/s survives that and is still
			// 0.0013 uu per displayed frame — far below anything visible.
			// Harmless for a ball placed on the ground: the floor stops it and the
			// engine puts it straight back to sleep.
			constexpr float WAKE_VEL = 0.02f;
			if (edit.hasPos && bs.vel == Vec() && bs.angVel == Vec())
				bs.vel.z = -WAKE_VEL;

			arena->ball->SetState(bs);
			continue;
		}

		Car* car = NULL;
		for (Car* c : arena->_cars) {
			if (c->id == edit.carId) {
				car = c;
				break;
			}
		}
		if (!car)
			continue; // stale id from a page that outlived a team-size change

		CarState cs = car->GetState();
		if (edit.hasPos) cs.pos = edit.pos;
		if (edit.hasVel) cs.vel = edit.vel;
		if (edit.hasAngVel) cs.angVel = edit.angVel;
		if (edit.hasRot) cs.rotMat = RotMat::LookAt(edit.forward, edit.up);
		if (edit.hasBoost) cs.boost = edit.boost;
		if (edit.hasDemoed) {
			cs.isDemoed = edit.demoed;
			// Un-demoing without clearing the timer leaves a car that respawns on top
			// of wherever it was just placed.
			cs.demoRespawnTimer = edit.demoed ? RS_MAX(cs.demoRespawnTimer, 1.f) : 0.f;
		}

		// Deliberately Car::SetState, not the engine-native path the snapshot uses: an
		// edit is a hand-authored change, and the suspension/bump state left over from
		// wherever the car used to be is exactly what should NOT survive a teleport.
		car->SetState(cs);
	}

	pendingEdits.clear();

	// The frame on screen must keep describing the arena, or seeking away and back
	// would quietly discard the edit. Editing also forks the timeline: whatever was
	// recorded after this frame belongs to the run that wasn't edited.
	int64_t current = viewFrameId >= 0 ? viewFrameId : history.NewestId();
	if (current >= 0) {
		history.ReplaceFrame(current, arena);
		branchFrom = current;
		branchPending = true;
	}
	return true;
}

void VizControl::RecordFrame(Arena* arena) {
	// The first frame recorded after a seek is the start of a new timeline; the future
	// that was rewound past never happened and must not stay seekable.
	if (branchPending) {
		history.TruncateAfter(branchFrom);
		branchPending = false;
	}

	viewFrameId = -1; // live again
	history.Push(arena);
}

std::string VizControl::StatusJSON() const {
	nlohmann::json j;
	j["paused"] = paused;
	j["speed"] = speed;
	j["deterministic"] = deterministic;
	j["listening"] = sock >= 0;
	j["oldestFrame"] = history.OldestId();
	j["newestFrame"] = history.NewestId();
	// Which frame is on screen: a specific one while parked after a seek, otherwise
	// the newest. The page uses this to place its scrubber.
	j["viewFrame"] = viewFrameId >= 0 ? viewFrameId : history.NewestId();
	j["historyCapacity"] = (int64_t)history.capacity;
	j["obsHash"] = std::to_string(lastObsHash);

	j["blueAgent"] = blueAgent;
	j["orangeAgent"] = orangeAgent;
	j["opponents"] = availableOpponents;
	j["bots"] = availableBots;
	j["opponentError"] = opponentError;
	j["rlbotRunning"] = rlbotRunning;
	j["rlbotConnected"] = rlbotConnected;
	j["rlbotControlling"] = rlbotControlling;
	{
		auto it = records.find(MatchupKey());
		Record rec = (it == records.end()) ? Record{} : it->second;
		j["blueGoals"] = rec.blueGoals;
		j["orangeGoals"] = rec.orangeGoals;
	}
	return j.dump();
}
