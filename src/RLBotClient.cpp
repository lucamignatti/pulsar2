#include "RLBotClient.h"

#include <rlbot/BotManager.h>

#include <RLGymCPP/CommonValues.h>

#include <cmath>
#include <cstdlib>

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
	pd.isOnGround = (p->air_state() == rlbot::flat::AirState::OnGround);
	pd.hasJumped = p->has_jumped();
	pd.hasDoubleJumped = p->has_double_jumped();
	// v5: demolished_timeout is seconds until respawn; > 0 means currently demolished.
	pd.isDemoed = (p->demolished_timeout() > 0.f);

	return pd;
}

static GameState ToGameState(const rlbot::flat::GamePacket* packet) {
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
	if (!boostPads || boostPads->size() != CommonValues::BOOST_LOCATIONS_AMOUNT) {
		if (rand() % 20 == 0) { // Don't spam-log as that will lag the bot
			RG_LOG(
				"RLBotClient ToGameState(): Bad boost pad amount, expected "
				<< CommonValues::BOOST_LOCATIONS_AMOUNT << " but got " << (boostPads ? boostPads->size() : 0)
			);
		}
		// GameState's constructor already defaults every pad to available, so leave as-is.
	} else {
		for (int i = 0; i < CommonValues::BOOST_LOCATIONS_AMOUNT; i++) {
			gs.boostPads[i] = boostPads->Get(i)->is_active();
			gs.boostPadsInv[CommonValues::BOOST_LOCATIONS_AMOUNT - i - 1] = gs.boostPads[i];

			gs.boostPadTimers[i] = boostPads->Get(i)->timer();
			gs.boostPadTimersInv[CommonValues::BOOST_LOCATIONS_AMOUNT - i - 1] = gs.boostPadTimers[i];
		}
	}

	return gs;
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

	GameState gs = ToGameState(packet);

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

		if (ctx.updateAction) {
			ctx.updateAction = false;
			ctx.action = params.inferUnit->InferAction(localPlayer, gs, true);
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
