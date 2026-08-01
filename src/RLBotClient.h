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
#include <string>

struct RLBotParams {
	int tickSkip;    // Your tick skip
	int actionDelay; // Your action delay

	GGL::InferUnit* inferUnit = nullptr;
};

// The BotManager only ever calls a fixed (indices, team, name) constructor, so the
// InferUnit/config is handed to spawned bots through this global.
extern RLBotParams g_RLBotParams;

class RLBotBot final : public rlbot::Bot {
public:
	RLBotBot(std::unordered_set<unsigned> indices, unsigned team, std::string name) noexcept;
	~RLBotBot() noexcept override;

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
