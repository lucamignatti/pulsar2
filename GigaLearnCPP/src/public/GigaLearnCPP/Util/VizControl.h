#pragma once
// GigaLearnCPP's Framework (not RLGymCPP's): this is where RG_IMEXPORT lives, and it
// pulls in the RLGymCPP one — which brings the engine types used below.
#include "../Framework.h"

#include <deque>
#include <map>

namespace GGL {

	// One external RLBot agent the viewer may offer as an opponent. Declared out here
	// rather than inside VizControl because it is also the return type of the finder
	// callback in LearnerConfig — whoever can serve an agent is the only thing that knows
	// how to find one, and both ends need to agree on what a found agent looks like.
	struct VizBotEntry {
		std::string path; // absolute path to the agent's .toml
		std::string name; // display name from that config's [settings] name
	};

	// The per-car payload of a snapshot.
	//
	// Under v3 this is the ENGINE'S OWN state struct, captured through the FFI rather
	// than through the v2-compat `Car::GetState`/`SetState` pair. That is not a
	// shortcut, it is the whole point: the facade's `CarState` cannot represent four
	// fields the Rust engine simulates —
	//
	//     timeSinceBoosted, isBoosting, supersonicGraceTimer, bumpCooldownTimer
	//
	// — so a Get/Set round trip through it silently zeroes them (RocketSimCompat.cpp
	// GetState never reads them; SetState hardcodes 0 or re-derives). Boost has a
	// minimum-application time and supersonic has a grace window, so dropping those
	// changes the physics: restoring a mid-air, mid-boost car through the facade and
	// replaying the same inputs visibly diverges within a couple of seconds (measured
	// — see tests/test_viz_snapshot.cpp, which fails if this regresses).
	//
	// Under the legacy v2 engine `CarState` IS the engine's native type, so it is
	// lossless there and used directly.
#ifdef RG_ROCKETSIM_V3
	typedef RsfCarState VizCarState;
#else
	typedef CarState VizCarState;
#endif

	// A complete, restorable picture of one arena at one decision step.
	//
	// Deliberately NOT built on RLGC::GameState: that is a lossy derived summary
	// (PhysState plus a few flags) made for observation-building, and restoring from
	// it would drop every engine internal the policy can feel — jump and flip timers,
	// air time since jump, boost pad cooldowns.
	//
	// KNOWN LIMIT: `tickCount` is captured for the timeline but CANNOT be restored.
	// The v3 engine owns it (`Arena::Step` overwrites the facade's copy from
	// `rsf_arena_tick_count` every step) and the FFI exposes no setter. It feeds
	// episode bookkeeping and event tracking, not physics, so rewind is unaffected;
	// but a rewound arena keeps counting ticks forward from where it was.
	struct ArenaSnapshot {
		// Monotonic id assigned by the ring, so the browser can name a frame to seek to
		// without depending on ring positions that shift as it rotates.
		int64_t frameId = -1;

		uint64_t tickCount = 0;
		BallState ball = {};
		// Arena RNG state. Only demo respawns consume it (to pick a spawn point), but
		// that alone is enough to make a rewind across a demo put the victim somewhere
		// else. Unused under the legacy v2 engine.
		uint64_t rngState = 0;

		struct CarSnapshot {
			uint32_t id = 0;
			Team team = Team::BLUE;
			VizCarState state = {};
			// The facade's *pending* controls, applied at the next Step(): distinct from
			// the state's prevControls (what was applied on the previous one). Restoring
			// only the latter would let the pre-seek action drive the first tick after a
			// seek.
			CarControls controls = {};
			// Synthesized facade-side from the engine's hit events and persisting across
			// steps, so it lives outside the engine state struct and must be carried
			// separately or touch detection misfires right after a seek.
			BallHitInfo ballHitInfo = {};
			// Opaque engine block (raycast suspension state + pending bump impulse) that
			// even the engine's own car state struct cannot express. Sized at runtime by
			// rsf_car_extra_state_size(); this side never interprets the bytes. Without
			// it a restore looks exact and then diverges ~23uu within a second. Empty
			// under the legacy v2 engine, which has no equivalent.
			std::vector<uint8_t> engineExtra = {};
		};
		std::vector<CarSnapshot> cars = {};

		struct PadSnapshot {
			bool isActive = true;
			float cooldown = 0;
			// BoostPadState::curLockedCar is a raw Car* into the live arena, meaningless
			// once stored. Keep the car *id* and remap on restore.
			uint32_t lockedCarId = 0;
		};
		std::vector<PadSnapshot> pads = {};

		void CaptureFrom(Arena* arena);
		// No-op returning false if the snapshot's car/pad counts don't match the arena's
		// — the only way that happens is a snapshot crossing a team-size change, and
		// half-applying it would be worse than refusing.
		bool ApplyTo(Arena* arena) const;
	};

	// Fixed-capacity ring of snapshots, oldest evicted first. Owns frame-id assignment.
	//
	// Memory is not a concern at these sizes (~1KB per snapshot at 3v3, so a 60s /
	// 900-frame history is under a megabyte) and this only ever exists in the render
	// viewer, never in the trainer.
	struct SnapshotRing {
		size_t capacity;
		std::deque<ArenaSnapshot> frames = {};
		int64_t nextFrameId = 0;

		explicit SnapshotRing(size_t capacity = 900) : capacity(capacity) {}

		// Capture the arena as the newest frame and return its id.
		int64_t Push(Arena* arena);

		void Clear() { frames.clear(); }

		bool Empty() const { return frames.empty(); }
		int64_t OldestId() const { return frames.empty() ? -1 : frames.front().frameId; }
		int64_t NewestId() const { return frames.empty() ? -1 : frames.back().frameId; }

		// Nearest stored frame at or before `frameId` (frames are pushed in id order, so
		// this is a binary search). NULL if the ring is empty or has rotated past it.
		const ArenaSnapshot* Find(int64_t frameId) const;

		// Re-capture the arena into a frame already in the ring, keeping its id. Used
		// when state is edited while parked: the frame on screen has to keep matching
		// what is actually in the arena, or seeking away and back would silently undo
		// the edit. False if the id isn't held.
		bool ReplaceFrame(int64_t frameId, Arena* arena);

		// Drop every frame after `frameId`. Called when play resumes from a seek: the
		// future that was rewound past didn't happen any more, and leaving it in the ring
		// would let a later seek jump into a timeline the arena never took.
		void TruncateAfter(int64_t frameId);
	};

	// The render viewer's control channel: transport state plus the UDP socket the
	// browser's control panel talks to.
	//
	// Transport is authoritative HERE, not in the browser: the C++ side owns the arena,
	// so it owns whether the arena steps. The page is a remote control, and a page
	// reload or a dead socket leaves the sim exactly as it was.
	//
	// Commands arrive as one JSON object per datagram on `cmdPort`, relayed from the
	// page's websocket by the viz web server. Everything about parsing here is
	// deliberately forgiving — an unknown command, a malformed packet, or a stray
	// datagram from anything else on the box is dropped silently. A viewer must never
	// die because of what arrived on a socket.
	//
	// EXISTS ONLY IN RENDER MODE. The trainer never constructs this and never opens a
	// port.
	struct RG_IMEXPORT VizControl {
		// Command port. The viz stack's other ports are 9273 (state in), 9274 (page
		// websocket) and 9275 (http); this is the fourth, browser -> sim.
		static constexpr int DEFAULT_CMD_PORT = 9276;

		VizControl(int cmdPort, size_t historyFrames);
		~VizControl();
		RG_NO_COPY(VizControl);

		bool paused = false;
		// Wall-clock pacing multiplier handed to RenderSender (1 = real time).
		float speed = 1.f;
		// Take the policy's argmax instead of sampling from it. Off matches how the bot
		// actually plays; ON is what makes a rewind replay the SAME play, because the
		// physics is deterministic but the policy is not — with sampling, rewinding and
		// resuming re-rolls every action and the play diverges immediately.
		bool deterministic = false;

		// -- Opponent selection --------------------------------------------------
		// Who plays the non-viewer team. Empty = mirror self-play (both sides are the
		// live policy). Otherwise a checkpoint directory: a golden archive entry
		// (best_r<rating>_<ts>), a numbered checkpoint, or a reference version.
		// WHO PLAYS EACH TEAM. Empty = the live policy; otherwise a checkpoint directory
		// name, or "rlbot:<path/to/bot.toml>" for an external RLBot agent.
		//
		// Two independent slots rather than one "opponent + which side": the interesting
		// matchups include checkpoint-vs-checkpoint and checkpoint-vs-bot, and a single
		// opponent field can only ever describe "us versus one other thing".
		std::string blueAgent;
		std::string orangeAgent;

		static constexpr const char* RLBOT_PREFIX = "rlbot:";
		static bool IsRLBot(const std::string& spec) {
			return spec.rfind(RLBOT_PREFIX, 0) == 0;
		}
		static std::string RLBotConfig(const std::string& spec) {
			return IsRLBot(spec) ? spec.substr(strlen(RLBOT_PREFIX)) : std::string();
		}
		const std::string& Agent(Team team) const {
			return team == Team::BLUE ? blueAgent : orangeAgent;
		}
		// The team an external agent plays, if any. Only one side may be an RLBot at a
		// time — two bot fleets in one arena is not a thing anyone has asked for, and it
		// would double the process management for no measured use.
		bool RLBotTeam(Team& out) const {
			if (IsRLBot(orangeAgent)) { out = Team::ORANGE; return true; }
			if (IsRLBot(blueAgent)) { out = Team::BLUE; return true; }
			return false;
		}

		// RLBot agent configs the panel offers. ABSOLUTE paths: bots are discovered across
		// several harness roots (this checkout's and the sibling one Element lives in), so
		// there is no single root left to be relative to — and an absolute path is also the
		// safer thing to hand back, since nothing downstream has to join browser-supplied
		// text onto a directory. `name` is what the config calls itself ("Nexto (Toxic!)"),
		// which is the only label that separates two configs sharing a directory.
		std::vector<VizBotEntry> availableBots;
		// Live status for the panel: is the agent up and has it connected.
		bool rlbotRunning = false;
		bool rlbotConnected = false;
		// Connected is only a handshake; this is the bit that means it is actually
		// driving. The two differ for an agent that connects and then wedges.
		bool rlbotControlling = false;

		bool opponentDirty = false;
		// Checkpoint directory names the panel offers, newest-interesting first. Filled
		// by the Learner (which owns the checkpoint folder) and refreshed as it polls,
		// so a golden entry saved mid-session shows up without a restart.
		std::vector<std::string> availableOpponents;
		// Set when the last requested opponent failed to load, so the panel can say so
		// instead of silently continuing to play the previous one.
		std::string opponentError;

		// Head-to-head record, keyed by the MATCHUP ("blue|orange"). Kept per
		// SIDE rather than as a single win count: the question a viewer is usually asked
		// is "does it still win from the orange side", and a merged tally cannot answer
		// that. Persisted nowhere — a viewer session is the unit of measurement.
		struct Record {
			int blueGoals = 0;
			int orangeGoals = 0;
		};
		std::map<std::string, Record> records;

		// Identifies the current matchup for the record book.
		std::string MatchupKey() const;

		// Call once per step with the arena; counts goals as they happen.
		void TrackScore(const RLGC::GameState& state);

		// Wipe the tally (panel button). Scoped to the current opponent only.
		void ResetRecord();

		SnapshotRing history;

		// True if the socket is open. When it isn't, the viewer still runs — it just
		// can't be controlled, which is strictly better than refusing to start.
		bool Listening() const { return sock >= 0; }

		// Drain every pending command datagram. Call once per render step.
		void Poll();

		// Whether the arena must NOT be stepped this iteration. Consumes one queued
		// single-step if there is one, so it is called exactly once per step decision.
		bool ConsumeHalt();

		// Restore a pending seek into the arena. Returns true if the arena changed, in
		// which case the caller must re-derive the env's view of it
		// (EnvSet::RefreshArenaState) before anything reads obs.
		bool ApplySeek(Arena* arena);

		// Write any queued state edits (dragged ball/cars, inspector fields) into the
		// arena. Same contract as ApplySeek: true means the caller must refresh.
		//
		// Edits are PATCHES — each names only the fields it changes, and everything
		// else is read from the live state and written straight back. That is what
		// keeps dragging a car from wiping the jump and flip timers that make it the
		// car it was.
		bool ApplyEdits(Arena* arena);

		// Record the arena as the newest history frame, discarding any rewound-past
		// future first.
		void RecordFrame(Arena* arena);

		// Checksum of the observation the policy was last given. Diagnostic only: it is
		// the one way to tell "the policy saw something different" apart from "the
		// policy behaved differently given the same input" when a replay diverges, and
		// the streamed gamestate is too lossy to reconstruct the obs from.
		// It caught a real one: the padded obs builder shuffles player slots from a
		// clock-seeded engine, so identical states hashed differently and the policy
		// picked different actions. To localise a future divergence to an element
		// rather than just detect it, stream the obs row itself alongside this.
		uint64_t lastObsHash = 0;

		// Panel state, embedded in the frame the viewer streams to the page.
		std::string StatusJSON() const;

	private:
		int sock = -1;
		int64_t viewFrameId = -1;   // frame currently shown; -1 = live
		int stepsQueued = 0;
		bool seekPending = false;
		int64_t seekTarget = 0;
		// Set when a seek lands; the discarded future is dropped at the next recorded
		// frame rather than at seek time, so scrubbing back and forth stays lossless
		// until play actually resumes.
		//
		// branchFrom is tracked SEPARATELY from viewFrameId rather than reusing it:
		// viewFrameId goes to -1 to mean "live", and truncating the ring at -1 deletes
		// every frame in it. That is exactly what happened when `play` reset viewFrameId
		// before the branch had been applied — the whole rewind history vanished and
		// frame ids restarted from zero.
		bool branchPending = false;
		int64_t branchFrom = 0;
		// Goals are counted on the RISING edge of the arena's scored flag, which stays
		// set for the rest of the step it fires on.
		bool goalLatched = false;

		// One queued state edit. Every field is optional: the panel sends only what the
		// user actually changed, so a drag moves a car without disturbing its velocity
		// and an inspector edit to boost doesn't teleport it.
		struct PendingEdit {
			bool isBall = false;
			uint32_t carId = 0;

			bool hasPos = false, hasVel = false, hasAngVel = false, hasRot = false;
			bool hasBoost = false, hasDemoed = false;

			Vec pos = {}, vel = {}, angVel = {};
			Vec forward = {}, up = {};
			float boost = 0;
			bool demoed = false;
		};
		std::vector<PendingEdit> pendingEdits;

		void HandleCommand(const std::string& text);
	};
}
