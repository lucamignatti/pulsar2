#include "RLBotClient.h"

#include <rlbot/BotManager.h>

#include <RLGymCPP/CommonValues.h>

#include <cmath>
#include <cstdlib>
#include <sstream>
#include <unistd.h>

using namespace RLGC;
using namespace GGL;

// Global params passed to bots spawned by the BotManager.
RLBotParams g_RLBotParams = {};

RLBotBot::RLBotBot(std::unordered_set<unsigned> indices, unsigned team, std::string name) noexcept
	: rlbot::Bot(std::move(indices), team, std::move(name)), params(g_RLBotParams) {

	for (unsigned index : this->indices)
		RG_LOG("Created RLBot bot: index " << index << ", team " << team << ", name: " << this->name << "...");
}

RLBotBot::~RLBotBot() noexcept = default;

// ---- flat -> RLGymCPP conversions (v5 schema) -------------------------------

static Vec ToVec(const rlbot::flat::Vector3& v) {
	return Vec(v.x(), v.y(), v.z());
}

static PhysState ToPhysObj(const rlbot::flat::Physics* phys) {
	PhysState obj = {};
	if (!phys) // early/partial packets during map load can omit physics
		return obj;
	obj.pos = ToVec(phys->location());

	// v5 Rotator: rotation() returns a struct ref with pitch()/yaw()/roll().
	Angle ang = Angle(phys->rotation().yaw(), phys->rotation().pitch(), phys->rotation().roll());
	obj.rotMat = ang.ToRotMat();

	obj.vel = ToVec(phys->velocity());
	obj.angVel = ToVec(phys->angular_velocity());
	return obj;
}

static Player ToPlayer(const rlbot::flat::PlayerInfo* p) {
	Player pd = {};

	static_cast<PhysState&>(pd) = ToPhysObj(p->physics());

	pd.carId = (uint32_t)p->player_id(); // v5: player_id replaces v4 spawnId
	pd.team = (Team)p->team();

	pd.boost = p->boost();
	// v5 exposes an AirState enum instead of a hasWheelContact bool.
	//
	// KNOWN RESIDUAL: the real game reports Jumping from the moment jump is pressed,
	// while the wheels stay grounded for ~6 more ticks (RocketSim keeps isOnGround true
	// there). Sub-decision-window transient (we act every 8 ticks, and our OWN jumps
	// start on decision ticks, so self-obs never lands inside it) - only opponents'
	// blocks can catch it. Tracking it needs per-car jump-time state; not worth it.
	const auto airState = p->air_state();
	pd.isOnGround = (airState == rlbot::flat::AirState::OnGround);
	pd.hasJumped = p->has_jumped();
	pd.hasDoubleJumped = p->has_double_jumped();

	// FLIP STATE (fixed 2026-07-31; the third instance of the has_flip bug family, after
	// the two Nexto-side fixes the same day). These two fields were never set, so
	// HasFlipOrJump() - which the obs exposes per player AND the action parser gates its
	// jump rows on - reduced to `!hasDoubleJumped`: the bot played every real match
	// believing it always had a dodge while airborne, after flipping and forever after
	// the window lapsed. Off-distribution obs plus a mask offering jumps the car cannot
	// perform; the visible symptom is exactly "flails in the air, never converts".
	pd.hasFlipped = p->has_dodged();
	// dodge_timeout = seconds of dodge window remaining; -1 "while on ground or when
	// airborne for too long after jumping" (schema). Invert it into RocketSim's
	// airTimeSinceJump. The -1 case is ambiguous, so disambiguate with the jump flags:
	// airborne after a jump with no live window means the window is spent - pin the
	// field PAST the threshold so HasFlipOrJump() reads false, matching the sim. (May
	// come out slightly negative while a held jump extends the window past 1.25s;
	// negative still compares < threshold, i.e. flip available, which is correct.)
	const float dodgeTimeout = p->dodge_timeout();
	if (dodgeTimeout >= 0.f)
		pd.airTimeSinceJump = RLConst::DOUBLEJUMP_MAX_DELAY - dodgeTimeout;
	else if (airState == rlbot::flat::AirState::Jumping)
		// First-jump forces still active (button held, up to 0.2s): the dodge window
		// has not OPENED yet, so dodge_timeout is -1 - but the flip is NOT spent.
		// RocketSim pins airTimeSinceJump to 0 while jumping, so match it. The first
		// version of this reconstruction missed the case and pinned it PAST the
		// threshold instead: at the decision right after every jump press the obs
		// said "flip gone" and the mask stripped every jump row, then the flip
		// "reappeared" a step later - the policy dodged late from a nose-up posture,
		// which is a backflip. Caught by watching real matches (2026-07-31).
		pd.airTimeSinceJump = 0.f;
	else if (pd.hasJumped && !pd.isOnGround)
		pd.airTimeSinceJump = RLConst::DOUBLEJUMP_MAX_DELAY;
	else
		pd.airTimeSinceJump = 0.f;

	// TURTLE CONTACT (approximation). The action parser un-masks jump for a turtled car
	// via worldContact (chassis-on-floor, normal.z > 0.9), which no packet field carries.
	// Without it a bot that lands on its roof has every jump row masked and cannot right
	// itself - in training that state always offered jump. Reconstruct the resting-turtle
	// case only: inverted, chassis-height off the floor, not on wheels, and not moving
	// vertically (excludes mid-flip inverted passes, where training had no contact
	// either). False negatives cost one masked escape; false positives offer a jump row
	// the sim also offered on any chassis contact, so err tight.
	if (!pd.isOnGround && pd.rotMat.up.z < -0.9f
		&& pd.pos.z < 60.f && std::abs(pd.vel.z) < 50.f) {
		pd.worldContact.hasContact = true;
		pd.worldContact.contactNormal = Vec(0, 0, 1);
	}

	// v5: demolished_timeout is seconds until respawn; > 0 means currently demolished.
	pd.isDemoed = (p->demolished_timeout() > 0.f);

	return pd;
}

// ---- boost pad mapping ------------------------------------------------------
// The v5 packet orders boost_pads "by y-coordinate and then x-coordinate" (schema),
// which is NOT CommonValues::BOOST_LOCATIONS order - e.g. y-then-x puts both
// (+-2048, -1036) pads BEFORE (0, -1024), where CommonValues interleaves them. The old
// identity assumption misattributed state/timers across the whole midfield cluster.
// Match by position from FieldInfo once, at initialize().

void RLBotBot::initialize(
	rlbot::flat::ControllableTeamInfo const* controllableTeamInfo,
	rlbot::flat::FieldInfo const* fieldInfo,
	rlbot::flat::MatchConfiguration const* matchConfiguration) noexcept {
	rlbot::Bot::initialize(controllableTeamInfo, fieldInfo, matchConfiguration);

	padMap.clear();

	auto pads = fieldInfo ? fieldInfo->boost_pads() : nullptr;
	if (!pads || pads->size() != CommonValues::BOOST_LOCATIONS_AMOUNT) {
		RG_LOG("RLBotBot::initialize(): FieldInfo has " << (pads ? pads->size() : 0)
			<< " boost pads, expected " << CommonValues::BOOST_LOCATIONS_AMOUNT
			<< " - pad state will read all-available");
		return;
	}

	// Nearest-neighbor by 2D distance: the real game's pad coordinates differ from
	// CommonValues by a few uu (e.g. 3308.45 vs 3310), so exact matching is wrong and a
	// fixed small tolerance is fragile. Uniqueness check below catches any bad pairing.
	std::vector<int> map(CommonValues::BOOST_LOCATIONS_AMOUNT, -1);
	std::vector<bool> used(pads->size(), false);
	for (int i = 0; i < CommonValues::BOOST_LOCATIONS_AMOUNT; i++) {
		const Vec& want = CommonValues::BOOST_LOCATIONS[i];
		float bestDistSq = -1;
		int best = -1;
		for (uint32_t j = 0; j < pads->size(); j++) {
			auto loc = pads->Get(j)->location();
			float dx = loc->x() - want.x, dy = loc->y() - want.y;
			float distSq = dx * dx + dy * dy;
			if (best < 0 || distSq < bestDistSq) {
				bestDistSq = distSq;
				best = (int)j;
			}
		}
		if (used[best]) {
			RG_LOG("RLBotBot::initialize(): duplicate boost pad match at index " << best
				<< " - pad layout unrecognized, pad state will read all-available");
			return;
		}
		used[best] = true;
		map[i] = best;
	}

	padMap = std::move(map);
	RG_LOG("RLBotBot::initialize(): boost pad index map built ("
		<< CommonValues::BOOST_LOCATIONS_AMOUNT << " pads matched by position)");
}

static GameState ToGameState(const rlbot::flat::GamePacket* packet,
	const std::vector<int>& padMap) {
	GameState gs = {};

	// Every flatbuffer vector accessor returns null when the field is absent, which
	// happens in the early/partial packets RLBotServer emits while the map is still
	// loading (the moment the bot joins). Guard every one - a bare ->size() on null
	// segfaults, which is exactly what killed the bot on its first live packet.
	auto players = packet->players();
	if (players) {
		for (uint32_t i = 0; i < players->size(); i++) {
			Player pl = ToPlayer(players->Get(i));
			pl.index = (int)i;
			gs.players.push_back(pl);
		}
	}

	auto balls = packet->balls();
	if (balls && balls->size() > 0)
		static_cast<PhysState&>(gs.ball) = ToPhysObj(balls->Get(0)->physics());

	auto boostPads = packet->boost_pads();
	if (!boostPads || boostPads->size() != CommonValues::BOOST_LOCATIONS_AMOUNT
		|| padMap.size() != (size_t)CommonValues::BOOST_LOCATIONS_AMOUNT) {
		if (rand() % 20 == 0) { // Don't spam-log as that will lag the bot
			RG_LOG(
				"RLBotClient ToGameState(): Bad boost pad amount or no pad map, expected "
				<< CommonValues::BOOST_LOCATIONS_AMOUNT << " but got " << (boostPads ? boostPads->size() : 0)
			);
		}
		// GameState's constructor already defaults every pad to available, so leave as-is.
	} else {
		for (int i = 0; i < CommonValues::BOOST_LOCATIONS_AMOUNT; i++) {
			// padMap: packet pads are y-then-x ordered, not CommonValues order (see
			// initialize()). Read this slot's pad through the position-matched index.
			auto pad = boostPads->Get(padMap[i]);
			gs.boostPads[i] = pad->is_active();
			gs.boostPadsInv[CommonValues::BOOST_LOCATIONS_AMOUNT - i - 1] = gs.boostPads[i];

			// TIMER DIRECTION - MEASURED, do not "fix" from the schema again. The schema
			// doc claims timer is "seconds SINCE the boost has been picked up" (count-up),
			// and on 2026-07-31 this code briefly converted accordingly (cooldown - timer).
			// The debug JSONL from a real match (debug.3496495.jsonl) proved the server
			// actually sends a COUNTDOWN: at the first inactive frame after pickup the
			// packet read ~3.94 on a 4s pad, and the obs pad feature fell 0.94 -> 0.56
			// across the window (37 falling runs vs 2 rising) - i.e. the conversion had
			// re-inverted an already-correct value. Raw pass-through IS training's
			// semantics (RocketSim cooldown, seconds remaining). If a different backend
			// ever honors the schema's count-up wording, the pad-direction check in the
			// debug-log analysis is the detector: inactive-pad runs must RISE toward 1.
			float remaining = pad->is_active() ? 0.f : RS_MAX(pad->timer(), 0.f);
			gs.boostPadTimers[i] = remaining;
			gs.boostPadTimersInv[CommonValues::BOOST_LOCATIONS_AMOUNT - i - 1] = remaining;
		}
	}

	return gs;
}

// ---- scripted kickoff -------------------------------------------------------
// A 120Hz control tape, same deployment-harness architecture as Nexto's
// hardcoded_kickoffs (rlbot-run/nexto/bot.py KICKOFF_CONTROLS): the script overrides
// the policy per tick while the match phase is Kickoff and the ball is untouched at
// center, then hands back. REAL-GAME ONLY by construction - the viz's RLBot bridge
// always reports MatchPhase::Active, and neither training path runs this client.
//
// The tape is Nexto's speedflip MIRRORED (steer/yaw/roll sign-flipped): RL physics is
// x-mirror symmetric, so the known-good timings carry over exactly - we flip left
// where Nexto flips right and roll out the opposite side. Deliberately NOT re-timed
// and NOT a wavedash: a wavedash also spends the flip but converts it to less speed
// later, i.e. it arrives second to every scripted speedflip. Any future re-timing
// should be tuned in RocketSim first, not shipped blind.
//
// Segments (ticks x [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]):
static const std::vector<RLGC::Action>& KickoffTape() {
	static std::vector<RLGC::Action> tape = [] {
		std::vector<RLGC::Action> t;
		auto seg = [&](int ticks, float throttle, float steer, float pitch, float yaw,
			float roll, float jump, float boost) {
			RLGC::Action a = {};
			a.throttle = throttle; a.steer = steer; a.pitch = pitch; a.yaw = yaw;
			a.roll = roll; a.jump = jump; a.boost = boost; a.handbrake = 0;
			for (int i = 0; i < ticks; i++)
				t.push_back(a);
		};
		seg(44, 1, 0, 0, 0, 0, 0, 1);            // straight, build speed
		seg(16, 1, +1, 0, 0, 0, 0, 1);           // angle off-line (Nexto: steer -1)
		seg(8, 1, 0, 0, 0, 0, 1, 1);             // first jump
		seg(4, 1, 0, 0, 0, 0, 0, 1);             // release
		seg(4, 1, 0, -0.7f, -0.8f, 0, 1, 1);     // diagonal dodge (Nexto: yaw +0.8)
		seg(52, 1, 0, +1, 0, 0, 0, 1);           // flip cancel, nose down through it
		seg(40, 1, 0, +0.5f, 0, -1, 0, 0);       // land rolling out left (Nexto: +1)
		return t;
	}();
	return tape;
}

// Nexto's taker selection, ported: the closest car goes; a same-team car at
// effectively equal distance yields to the LEFT one (the community convention - both
// bots agreeing on it is what prevents a teammate double-commit in 2v2/3v3).
static bool IsKickoffTaker(const rlbot::flat::GamePacket* packet, unsigned selfIndex, unsigned team) {
	auto players = packet->players();
	auto balls = packet->balls();
	if (!players || selfIndex >= players->size() || !balls || balls->size() == 0
		|| !balls->Get(0)->physics() || !players->Get(selfIndex)->physics())
		return false; // early/partial packet (same guard rationale as ToPhysObj)

	const auto ballLoc = balls->Get(0)->physics()->location();
	auto distTo = [&](unsigned i) {
		auto phys = players->Get(i)->physics();
		if (!phys)
			return 1e9f; // never the closest
		const auto loc = phys->location();
		float dx = loc.x() - ballLoc.x(), dy = loc.y() - ballLoc.y();
		return sqrtf(dx * dx + dy * dy);
	};

	const float selfDist = distTo(selfIndex);
	for (uint32_t i = 0; i < players->size(); i++) {
		if (i == selfIndex)
			continue;
		float d = distTo(i);
		if (d < selfDist - 10.f)
			return false; // someone is strictly closer
		if (fabsf(d - selfDist) <= 10.f && players->Get(i)->team() == team) {
			// Equal-distance teammate: left goes. "Left" faces the orange goal for
			// blue and the blue goal for orange, so the sign of x flips per team.
			const float selfX = players->Get(selfIndex)->physics()->location().x();
			const float otherX = players->Get(i)->physics()->location().x();
			bool otherIsLeft = (team == 0) ? (otherX < selfX) : (otherX > selfX);
			if (otherIsLeft)
				return false;
		}
	}
	return true;
}

// ---- debug logging ----------------------------------------------------------
// GGL_DEBUG_JSONL=1 -> one JSONL file per bot process in the cwd (pulsar-bot/, next
// to the bot.<pid>.log the run wrapper already writes). Two record types:
//   "decision":   every inference - the EXACT obs/mask/action the policy consumed
//                 (via InferUnit::InferDebug, not a rebuild), the raw packet fields
//                 for self, and their reconstruction. ~15Hz, a few MB per match.
//   "transition": self air_state changes, per packet. Sparse; this is what settles
//                 packet-semantics questions the schema leaves open (e.g. whether
//                 has_jumped resets on landing, what dodge_timeout reads while
//                 Jumping) - the exact assumptions the reconstruction above makes.

void RLBotBot::DebugLogLine(const std::string& line) {
	if (!debugLogTried) {
		debugLogTried = true;
		const char* v = std::getenv("GGL_DEBUG_JSONL");
		if (v && *v && std::string(v) != "0") {
			std::string path = "debug." + std::to_string(getpid()) + ".jsonl";
			debugLog.open(path, std::ios::out | std::ios::app);
			if (debugLog.is_open())
				RG_LOG("RLBotBot: debug JSONL logging to " << path);
		}
	}
	if (debugLog.is_open())
		debugLog << line << "\n" << std::flush;
}

static void AppendFloatArray(std::ostringstream& s, const float* v, size_t n) {
	s << "[";
	char buf[24];
	for (size_t i = 0; i < n; i++) {
		snprintf(buf, sizeof(buf), "%.5g", v[i]);
		s << (i ? "," : "") << buf;
	}
	s << "]";
}

// 90-bit action mask as hex (LSB-first nibbles), plus how many jump rows survived it.
static std::string MaskToHex(const std::vector<uint8_t>& mask) {
	std::string out;
	for (size_t base = 0; base < mask.size(); base += 4) {
		int nib = 0;
		for (size_t b = 0; b < 4 && base + b < mask.size(); b++)
			if (mask[base + b])
				nib |= 1 << b;
		out += "0123456789abcdef"[nib];
	}
	return out;
}

// ---- per-tick control -------------------------------------------------------

void RLBotBot::update(
	rlbot::flat::GamePacket const* packet,
	rlbot::flat::BallPrediction const* ballPrediction) noexcept {
	(void)ballPrediction;

	if (!params.inferUnit)
		return;

	auto matchInfo = packet->match_info();
	float curTime = matchInfo ? matchInfo->seconds_elapsed() : prevTime;
	float deltaTime = curTime - prevTime;
	prevTime = curTime;

	int ticksElapsed = (int)roundf(deltaTime * 120);

	GameState gs = ToGameState(packet, padMap);

	for (unsigned index : this->indices) {
		// Default-constructs the ctx (updateAction = true, ticks = -1) on first sight.
		CarCtx& ctx = ctxByIndex[index];

		if (index >= gs.players.size()) {
			// We're not in the packet yet; coast.
			setOutput(index, {});
			continue;
		}

		ctx.ticks += ticksElapsed;

		auto& localPlayer = gs.players[index];
		localPlayer.prevAction = ctx.controls;

		static const bool dbgOn = [] {
			const char* v = std::getenv("GGL_DEBUG_JSONL");
			return v && *v && std::string(v) != "0";
		}();
		auto rawSelf = (packet->players() && index < packet->players()->size())
			? packet->players()->Get(index) : nullptr;

		// Air-state transition line: per packet, sparse. This is the record that
		// settles what the packet ACTUALLY does at jump boundaries (dodge_timeout
		// while Jumping, has_jumped across landings) versus what ToPlayer assumes.
		if (dbgOn && rawSelf) {
			uint8_t as = (uint8_t)rawSelf->air_state();
			if (ctx.prevAirState != as) {
				std::ostringstream s;
				s << "{\"type\":\"transition\",\"t\":" << curTime << ",\"i\":" << index
					<< ",\"as\":" << (int)as << ",\"prev_as\":" << (int)ctx.prevAirState
					<< ",\"dt\":" << rawSelf->dodge_timeout()
					<< ",\"hj\":" << (int)rawSelf->has_jumped()
					<< ",\"hd\":" << (int)rawSelf->has_dodged()
					<< ",\"hdj\":" << (int)rawSelf->has_double_jumped()
					<< ",\"z\":" << localPlayer.pos.z << ",\"vz\":" << localPlayer.vel.z << "}";
				DebugLogLine(s.str());
				ctx.prevAirState = as;
			}
		}

		if (ctx.updateAction) {
			ctx.updateAction = false;
			// GGL_SAMPLE_ACTIONS=1 -> sample from the policy instead of argmax. The viz
			// viewer samples by default (matching how the bot trains and self-plays), so
			// sim-parity experiments should set this; deployment default stays argmax.
			static const bool sampleActions = [] {
				const char* v = std::getenv("GGL_SAMPLE_ACTIONS");
				return v && *v && std::string(v) != "0";
			}();
			GGL::InferUnit::InferDebug dbg;
			ctx.action = params.inferUnit->InferAction(localPlayer, gs, !sampleActions, 1,
				dbgOn ? &dbg : nullptr);

			// Decision line: the exact obs/mask/action inference consumed, the raw
			// packet fields it was reconstructed from, and where the car/ball were.
			if (dbgOn && rawSelf) {
				std::ostringstream s;
				s << "{\"type\":\"decision\",\"t\":" << curTime << ",\"i\":" << index
					<< ",\"ko\":" << ctx.kickoffIndex
					<< ",\"as\":" << (int)rawSelf->air_state()
					<< ",\"dt\":" << rawSelf->dodge_timeout()
					<< ",\"hj\":" << (int)rawSelf->has_jumped()
					<< ",\"hd\":" << (int)rawSelf->has_dodged()
					<< ",\"hdj\":" << (int)rawSelf->has_double_jumped()
					<< ",\"boost\":" << localPlayer.boost
					<< ",\"g\":" << (int)localPlayer.isOnGround
					<< ",\"hf\":" << (int)localPlayer.hasFlipped
					<< ",\"atsj\":" << localPlayer.airTimeSinceJump
					<< ",\"flip\":" << (int)localPlayer.HasFlipOrJump()
					<< ",\"turtle\":" << (int)localPlayer.worldContact.hasContact
					<< ",\"act\":" << dbg.actionIndex;
				float tuple[8];
				for (int d = 0; d < 8; d++)
					tuple[d] = ctx.action[d];
				s << ",\"act_tuple\":"; AppendFloatArray(s, tuple, 8);
				float sp[3] = { localPlayer.pos.x, localPlayer.pos.y, localPlayer.pos.z };
				float sv[3] = { localPlayer.vel.x, localPlayer.vel.y, localPlayer.vel.z };
				float sf[3] = { localPlayer.rotMat.forward.x, localPlayer.rotMat.forward.y, localPlayer.rotMat.forward.z };
				float bp[3] = { gs.ball.pos.x, gs.ball.pos.y, gs.ball.pos.z };
				float bv[3] = { gs.ball.vel.x, gs.ball.vel.y, gs.ball.vel.z };
				s << ",\"p\":"; AppendFloatArray(s, sp, 3);
				s << ",\"v\":"; AppendFloatArray(s, sv, 3);
				s << ",\"f\":"; AppendFloatArray(s, sf, 3);
				s << ",\"b\":"; AppendFloatArray(s, bp, 3);
				s << ",\"bv\":"; AppendFloatArray(s, bv, 3);
				s << ",\"mask\":\"" << MaskToHex(dbg.actionMask) << "\"";
				s << ",\"obs\":"; AppendFloatArray(s, dbg.obs.data(), dbg.obs.size());
				s << "}";
				DebugLogLine(s.str());
			}
		}

		if (ctx.ticks >= (params.actionDelay - 1) || ctx.ticks == -1) {
			// Apply new action
			ctx.controls = ctx.action;
		}

		if (ctx.ticks >= params.tickSkip || ctx.ticks == -1) {
			// Trigger action update next tick
			ctx.ticks = 0;
			ctx.updateAction = true;
		}

		// Scripted kickoff override (see KickoffTape above). Runs AFTER the policy
		// cadence so ctx.controls carries the tape row - which also makes next
		// packet's prevAction (and therefore the obs) truthful about what was
		// actually applied; the policy keeps inferring underneath and takes over
		// the instant the tape ends or the phase leaves Kickoff. The ball-at-center
		// check mirrors Nexto's: the tape only makes sense while the kickoff ball
		// is untouched at (0, 0) - exact float compare is safe there.
		static const bool noKickoffScript = [] {
			const char* v = std::getenv("GGL_NO_KICKOFF_SCRIPT");
			return v && *v && std::string(v) != "0";
		}();
		const bool kickoffPhase = matchInfo
			&& matchInfo->match_phase() == rlbot::flat::MatchPhase::Kickoff;
		if (kickoffPhase && !noKickoffScript) {
			if (ctx.kickoffIndex == -1)
				ctx.kickoffIndex = IsKickoffTaker(packet, index, this->team) ? 0 : -2;
			else if (ctx.kickoffIndex >= 0)
				ctx.kickoffIndex += ticksElapsed;

			auto balls = packet->balls();
			const bool ballCentered = balls && balls->size() > 0
				&& balls->Get(0)->physics()->location().y() == 0.f;
			const auto& tape = KickoffTape();
			if (ctx.kickoffIndex >= 0 && ctx.kickoffIndex < (int)tape.size() && ballCentered)
				ctx.controls = tape[ctx.kickoffIndex];
		} else {
			ctx.kickoffIndex = -1;
		}

		const Action& c = ctx.controls;
		setOutput(index, {
			c.throttle, c.steer,
			c.pitch, c.yaw, c.roll,
			c.jump != 0.f, c.boost != 0.f, c.handbrake != 0.f,
			false /* use_item */
		});
	}
}

// ---- entry ------------------------------------------------------------------

void RLBotClient::Run(const RLBotParams& params) {
	g_RLBotParams = params;

	const char* host = std::getenv("RLBOT_SERVER_IP");
	if (!host || !*host)
		host = "127.0.0.1";

	const char* port = std::getenv("RLBOT_SERVER_PORT");
	if (!port || !*port)
		port = "23234";

	const char* agentId = std::getenv("RLBOT_AGENT_ID");
	if (!agentId || !*agentId)
		RG_ERR_CLOSE("RLBotClient: RLBOT_AGENT_ID environment variable is not set");

	RG_LOG("RLBotClient: connecting to RLBotServer at " << host << ":" << port << " as \"" << agentId << "\"...");

	rlbot::BotManager<RLBotBot> manager{false /* batchHivemind */};
	if (!manager.connect(host, port, agentId, true /* request ball prediction */))
		RG_ERR_CLOSE("RLBotClient: failed to connect to RLBotServer at " << host << ":" << port);

	// connect() returns once connected; the manager blocks on its service threads until
	// the server ends the match when it goes out of scope here.
}
