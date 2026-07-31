#pragma once
#include <RLGymCPP/Gamestates/GameState.h>
#include <RLGymCPP/BasicTypes/Action.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Lets any RLBot v5 bot play the opponent's car(s) inside the VIEWER'S OWN arena.
//
// The alternative was to hand the match to RLBotSim, which runs its own simulation —
// and that would cost the whole point of the control panel: pause, rewind, and state
// editing only work while our arena stays authoritative. So the viewer instead acts as
// a minimal RLBotServer: it listens, launches the bot, feeds it GamePackets built from
// our arena, and reads its ControllerState back.
//
// Only the slice of the protocol a playing bot needs is implemented. No ball
// prediction, no rendering, no match comms, no script support, one bot at a time.
// Anything unrecognised on the wire is skipped rather than treated as an error, since
// bots and managers send optional traffic we have no opinion about.
//
// Controls are applied RAW: a bot emits continuous throttle/steer/pitch/yaw/roll and
// squashing that onto our 90-row action table would measure a handicapped copy of the
// bot rather than the bot.
//
// Render-mode only, and compiled only when GGL_VIZ_RLBOT is on — the trainer never
// links it (see the CMake option).
namespace GGL {

	class VizRLBotServer {
	public:
		VizRLBotServer();
		~VizRLBotServer();
		VizRLBotServer(const VizRLBotServer&) = delete;
		VizRLBotServer& operator=(const VizRLBotServer&) = delete;

		// RLBot's conventional port. The bot inherits it via RLBOT_SERVER_PORT, so this
		// only needs to not collide with the viz stack's 9273-9276.
		static constexpr int DEFAULT_PORT = 23234;

		// Begin listening on loopback. Returns false (and sets LastError) if the port is
		// unavailable. Idempotent.
		bool Listen(int port);
		int Port() const { return port; }

		// Launch a bot from its RLBot config, pointed at our port. `botConfigPath` is a
		// bot.toml; its run_command and agent id are read from it. Any previously
		// launched bot is stopped first. Returns false and sets LastError on failure.
		bool LaunchBot(const std::string& botConfigPath, Team team, int numCars);

		// Kill the bot process and drop the connection, back to no external opponent.
		void StopBot();

		// Accept/read/write. Call once per decision step BEFORE reading controls.
		// `state` is the arena as it now stands; `opponentIndices` are the player rows
		// the bot owns, in the same order as its controllable list.
		void Poll(const RLGC::GameState& state, const std::vector<int>& opponentIndices);

		// Latest controls for the bot's cars, in `opponentIndices` order. False when the
		// bot isn't connected or has never sent input, in which case the caller must
		// fall back rather than reuse stale controls.
		// Latest controls for the bot's cars, in `opponentIndices` order. `outValid` marks
		// which entries an agent actually supplied — a partially-connected team must leave
		// its remaining cars to the policy rather than freeze them on a zeroed control.
		// Returns true if any entry is valid.
		bool GetControls(std::vector<RLGC::Action>& out, std::vector<uint8_t>& outValid) const;

		bool Connected() const;
		bool BotRunning() const;
		const std::string& LastError() const { return lastError; }
		const std::string& BotName() const { return botName; }

		// bot.toml files discoverable under `searchRoot`, as paths relative to it, so the
		// panel can offer a dropdown instead of asking for a filesystem path.
		static std::vector<std::string> FindBotConfigs(const std::string& searchRoot);

	private:
		struct Impl;
		std::unique_ptr<Impl> impl;

		// One-time-per-connection setup (ControllableTeamInfo + MatchConfiguration +
		// FieldInfo) and the per-step packet.
		void SendSessionInfo(struct BotClient& client, const RLGC::GameState& state,
			const std::vector<int>& opponentIndices);
		void SendGamePacket(struct BotClient& client, const RLGC::GameState& state);

		int port = 0;
		std::string lastError;
		std::string botName;
	};
}
