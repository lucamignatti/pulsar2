#pragma once

// RLBot v5 client. The bot is a *client* that connects to the RLBotServer (or RLBotSim)
// rather than a bot *server* the framework connects to. The inference core (InferUnit +
// AdvancedObsPadded + DefaultAction, wired up in RLBotMain.cpp to match ExampleMain.cpp's
// current model config) is transport-agnostic; this file only handles the v5 protocol.

#include <rlbot/Bot.h>

#include <RLGymCPP/ObsBuilders/ObsBuilder.h>
#include <RLGymCPP/ActionParsers/ActionParser.h>
#include <GigaLearnCPP/Util/InferUnit.h>

#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>

struct RLBotParams {
	int tickSkip;    // Your tick skip
	int actionDelay; // Your action delay

	GGL::InferUnit* inferUnit = nullptr;
};

// The BotManager only ever calls a fixed (indices, team, name) constructor, so the
// InferUnit/config is handed to spawned bots through this global.
extern RLBotParams g_RLBotParams;

// ---- scripted maneuver mode (GGL_SCRIPT=<file>) -----------------------------------
// Replays research/maneuvers/maneuvers.txt in the REAL game: state-sets the car at the
// start of each segment, applies the scripted controls by tick, and logs per-packet state
// in the same TSV the sim runner emits. The policy is bypassed entirely, so sim and game
// see identical inputs from identical states and any divergence is pure physics.
struct ScriptAct { int at; float t, s, p, y, r; bool jump, boost, hb; };
struct ScriptSeg { std::string name; int dur; float st[13]; std::vector<ScriptAct> acts; };

class RLBotBot final : public rlbot::Bot {
public:
	RLBotBot(std::unordered_set<unsigned> indices, unsigned team, std::string name) noexcept;
	~RLBotBot() noexcept override;

	// --- scripted mode state ---
	std::vector<ScriptSeg> script;      // empty => normal policy mode
	int scriptSeg = -1;                 // current segment (-1 = not started)
	int scriptTick = 0;                 // ticks since this segment's state set landed
	int scriptSettle = 0;               // packets spent waiting for the state set to apply
	bool scriptStateSent = false;
	// The game CARRIES jump/flip state across a state set, so a segment started with
	// whatever the previous one left behind: `stall` inherited a spent flip from
	// `speed_flip` and could not jump at all, while `corner_flip_into` inherited a fresh
	// one and took a +280.8 uu/s jump impulse in mid-air having never touched the ground.
	// That makes flip segments depend on script ORDER, is invisible in the output, and
	// nearly produced a bogus physics "fix" (see SIM2REAL_AUDIT.md S23). Park the car on
	// the ground before each segment: ground contact clears the jump/flip state in both
	// engines, so every segment starts from the same place.
	int scriptGrounded = -1;            // packets confirmed grounded during the reset phase
	std::ofstream scriptLog;
	void LoadScriptIfRequested();
	void RunScripted(rlbot::flat::GamePacket const* packet, unsigned index, int ticksElapsed);
	void SendSegmentState(const ScriptSeg& seg, unsigned index);
	void SendGroundReset(unsigned index);

	// Called once with FieldInfo (valid for the bot's lifetime): builds the boost pad
	// index map below.
	void initialize(
		rlbot::flat::ControllableTeamInfo const* controllableTeamInfo,
		rlbot::flat::FieldInfo const* fieldInfo,
		rlbot::flat::MatchConfiguration const* matchConfiguration) noexcept override;

	// Called by the BotManager on every received GamePacket.
	void update(
		rlbot::flat::GamePacket const* packet,
		rlbot::flat::BallPrediction const* ballPrediction) noexcept override;

private:
	RLBotParams params;

	// padMap[i] = index into the packet's boost_pads for CommonValues::BOOST_LOCATIONS[i],
	// matched by position from FieldInfo (the v5 packet orders pads y-then-x, which is NOT
	// CommonValues order - assuming identity misattributed the whole midfield cluster).
	// Empty until initialize(); ToGameState falls back to all-pads-available without it.
	std::vector<int> padMap;

	// Per-controlled-car state (keyed by index into packet->players()). With
	// hivemind = false this holds a single entry, but keying by index keeps it correct
	// if the same manager ever controls several cars.
	struct CarCtx {
		RLGC::Action action = {};   // Most recently inferred action
		RLGC::Action controls = {}; // Action currently being applied
		bool updateAction = true;
		int ticks = -1;
		// Scripted kickoff state (Nexto's state machine, ported): -1 = kickoff not yet
		// evaluated, -2 = evaluated and we are not the taker, >= 0 = tape position in
		// ticks. Reset to -1 whenever the match phase leaves Kickoff.
		int kickoffIndex = -1;
		// Previous packet's air_state for this car, for transition logging (255 = unseen).
		uint8_t prevAirState = 255;
		// Actuation-latency probe: when the controls we SEND change, remember what and
		// when; the packet's last_input echoes what the game APPLIED, so the first echo
		// matching the new controls dates the actuation. Lag in ticks = the venue's
		// real actionDelay (training uses 0).
		RLGC::Action echoSent = {};
		uint32_t echoSentFrame = 0;
		bool echoWaiting = false;
	};
	std::unordered_map<unsigned, CarCtx> ctxByIndex;

	float prevTime = 0; // secondsElapsed of the previous packet (shared across this bot's cars)

	// GGL_DEBUG_JSONL: per-decision debug log (exact obs/mask/action the policy saw,
	// plus the raw packet fields and their reconstruction) + air-state transition
	// lines. One file per bot process in the cwd; opened lazily on first use.
	std::ofstream debugLog;
	bool debugLogTried = false;
	void DebugLogLine(const std::string& line);
};

namespace RLBotClient {
	// Reads RLBOT_SERVER_IP / RLBOT_SERVER_PORT / RLBOT_AGENT_ID, connects to the
	// RLBotServer, and blocks until the match connection ends.
	void Run(const RLBotParams& params);
}
