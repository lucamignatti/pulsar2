#pragma once

// RLBot v5 client. The bot is a *client* that connects to the RLBotServer (or RLBotSim)
// rather than a bot *server* the framework connects to. The inference core (InferUnit +
// AdvancedObsPadded + DefaultAction, wired up in RLBotMain.cpp to match ExampleMain.cpp's
// current model config) is transport-agnostic; this file only handles the v5 protocol.

#include <rlbot/Bot.h>

#include <RLGymCPP/ObsBuilders/ObsBuilder.h>
#include <RLGymCPP/ActionParsers/ActionParser.h>
#include <GigaLearnCPP/Util/InferUnit.h>

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

	// Called by the BotManager on every received GamePacket.
	void update(
		rlbot::flat::GamePacket const* packet,
		rlbot::flat::BallPrediction const* ballPrediction) noexcept override;

private:
	RLBotParams params;

	// Per-controlled-car state (keyed by index into packet->players()). With
	// hivemind = false this holds a single entry, but keying by index keeps it correct
	// if the same manager ever controls several cars.
	struct CarCtx {
		RLGC::Action action = {};   // Most recently inferred action
		RLGC::Action controls = {}; // Action currently being applied
		bool updateAction = true;
		int ticks = -1;
	};
	std::unordered_map<unsigned, CarCtx> ctxByIndex;

	float prevTime = 0; // secondsElapsed of the previous packet (shared across this bot's cars)
};

namespace RLBotClient {
	// Reads RLBOT_SERVER_IP / RLBOT_SERVER_PORT / RLBOT_AGENT_ID, connects to the
	// RLBotServer, and blocks until the match connection ends.
	void Run(const RLBotParams& params);
}
