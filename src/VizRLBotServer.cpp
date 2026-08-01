#include "VizRLBotServer.h"

#include <RLGymCPP/CommonValues.h>

#include <corepacket_generated.h>
#include <interfacepacket_generated.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

using namespace GGL;
namespace flat = rlbot::flat;

// Wire framing, matching cpp-interface/library/Message.cpp: a 2-byte BIG-ENDIAN payload
// length, then a flatbuffer. Server->bot payloads are CorePacket, bot->server are
// InterfacePacket; both are unions, so the message type rides inside the buffer rather
// than in the header.
static constexpr size_t HEADER_SIZE = 2;
static constexpr size_t MAX_PAYLOAD = 65535;

#define VIZ_LOG(s) { std::cout << "[viz/rlbot] " << s << std::endl; }

// One connected agent process, owning exactly ONE car.
//
// RLBot runs a process per car unless the bot declares hivemind, and Nexto (like most
// bots) does not. Handing a single process two controllables produced exactly what you
// would expect: it drove one car and the other sat frozen on a zeroed control.
namespace GGL {
struct BotClient {
	int fd = -1;
	// Inbound byte stream; reassembled here since TCP gives no message boundaries.
	std::vector<uint8_t> inBuf;
	bool sentSessionInfo = false;
	bool haveInput = false;
	int ownedRow = -1;            // arena player row this process drives
	RLGC::Action control = {};    // its latest controller state
};
}

struct VizRLBotServer::Impl {
	int listenFd = -1;
	std::vector<BotClient> clients;
	std::vector<pid_t> botPids;
	// Arena rows the agents collectively own, in the order the caller gave them; used
	// to map a client's row back to the caller's control slot.
	std::vector<int> ownedRows;

	Team botTeam = Team::ORANGE;
	int numCars = 1;
	// The agent_id the bot was launched with. MatchConfiguration must attribute the
	// bot's own cars to it, or the bot cannot tell which of them it is playing.
	std::string agentId;

	// Latest controller state per controllable slot, indexed by the bot's own
	// player_index (which we assign as the arena row it owns).
	std::vector<RLGC::Action> controls;
	// Arena rows the bot owns, as told to it in ControllableTeamInfo.

	~Impl() {
		for (BotClient& c : clients)
			if (c.fd >= 0) close(c.fd);
		if (listenFd >= 0) close(listenFd);
	}

	void DropAllClients() {
		for (BotClient& c : clients)
			if (c.fd >= 0) close(c.fd);
		clients.clear();
	}

	// Send one CorePacket. Best-effort: a bot that has gone away or stopped reading
	// costs us the connection, never the viewer.
	bool Send(BotClient& client, flatbuffers::FlatBufferBuilder& fbb) {
		if (client.fd < 0)
			return false;
		const size_t size = fbb.GetSize();
		if (size > MAX_PAYLOAD) {
			VIZ_LOG("payload too large (" << size << " bytes), dropping");
			return false;
		}

		std::vector<uint8_t> out(HEADER_SIZE + size);
		out[0] = (uint8_t)(size >> 8);
		out[1] = (uint8_t)(size & 0xFF);
		memcpy(out.data() + HEADER_SIZE, fbb.GetBufferPointer(), size);

		size_t sent = 0;
		while (sent < out.size()) {
			ssize_t n = send(client.fd, out.data() + sent, out.size() - sent, MSG_NOSIGNAL);
			if (n > 0) {
				sent += (size_t)n;
				continue;
			}
			if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
				// The bot is not draining. Blocking here would stall the whole viewer,
				// so give up on this message; the next step sends a fresher one anyway.
				return false;
			}
			close(client.fd);
			client.fd = -1;
			return false;
		}
		return true;
	}
};

VizRLBotServer::VizRLBotServer() : impl(std::make_unique<Impl>()) {}

VizRLBotServer::~VizRLBotServer() {
	StopBot();
}

bool VizRLBotServer::Listen(int wantPort) {
	if (impl->listenFd >= 0)
		return true;

	int fd = socket(AF_INET, SOCK_STREAM, 0);
	if (fd < 0) {
		lastError = "socket() failed";
		return false;
	}

	int one = 1;
	setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
	fcntl(fd, F_SETFL, fcntl(fd, F_GETFL, 0) | O_NONBLOCK);

	sockaddr_in addr = {};
	addr.sin_family = AF_INET;
	addr.sin_port = htons((uint16_t)wantPort);
	// Loopback only: this port lets a process drive cars in the arena.
	addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

	if (bind(fd, (sockaddr*)&addr, sizeof(addr)) < 0) {
		lastError = "port " + std::to_string(wantPort) + " unavailable";
		close(fd);
		return false;
	}
	if (listen(fd, 1) < 0) {
		lastError = "listen() failed";
		close(fd);
		return false;
	}

	impl->listenFd = fd;
	port = wantPort;
	VIZ_LOG("listening on 127.0.0.1:" << wantPort);
	return true;
}

// Minimal reader for the two bot.toml keys we need. A real TOML parser would be a
// dependency for two lookups, and the RLBot config format keeps both at top level of
// their tables as plain `key = "value"`.
static std::string ReadTomlString(const std::string& path, const std::string& key) {
	std::ifstream f(path);
	if (!f)
		return "";
	std::string line;
	while (std::getline(f, line)) {
		// Strip comments outside quotes (RLBot configs don't use '#' inside these values)
		size_t hash = line.find('#');
		if (hash != std::string::npos)
			line = line.substr(0, hash);

		size_t eq = line.find('=');
		if (eq == std::string::npos)
			continue;

		std::string k = line.substr(0, eq);
		k.erase(std::remove_if(k.begin(), k.end(), ::isspace), k.end());
		if (k != key)
			continue;

		std::string v = line.substr(eq + 1);
		size_t a = v.find('"'), b = v.rfind('"');
		if (a != std::string::npos && b != std::string::npos && b > a)
			return v.substr(a + 1, b - a - 1);
	}
	return "";
}

std::vector<VizBotEntry> VizRLBotServer::FindBotConfigs(const std::vector<std::string>& searchRoots) {
	std::vector<VizBotEntry> found;
	// Which root each entry came from, kept only long enough to disambiguate names below.
	std::vector<std::string> entryRoot;

	for (const std::string& rawRoot : searchRoots) {
		std::error_code ec;
		// Normalized up front, because the label below names the root's PARENT directory to
		// say which checkout an entry came from — and callers pass roots like
		// `absolute("../rlbot-run")`, whose parent is the literal "..".  A trailing
		// separator would hide the real leaf the same way, so drop that too.
		std::filesystem::path searchRoot =
			std::filesystem::absolute(rawRoot, ec).lexically_normal();
		if (searchRoot.filename().empty())
			searchRoot = searchRoot.parent_path();
		if (!std::filesystem::exists(searchRoot, ec))
			continue; // a checkout that isn't on this machine is not an error

		// Shallow: bots live a directory or two under the harness root, and recursing the
		// whole tree would sweep in vendored copies nobody wants to play against. Depth 2
		// is load-bearing, not slack — Element's config is at `Elementv5/src/bot.toml`.
		for (auto it = std::filesystem::recursive_directory_iterator(
				searchRoot, std::filesystem::directory_options::skip_permission_denied, ec);
			it != std::filesystem::recursive_directory_iterator(); it.increment(ec)) {
			if (ec)
				break;
			if (it.depth() > 2) {
				it.disable_recursion_pending();
				continue;
			}
			if (!it->is_regular_file(ec))
				continue;
			if (it->path().extension() != ".toml")
				continue;

			// The content sniff. An agent config is the file that says how to launch an
			// agent, so a run command is both necessary and sufficient — and it is what
			// separates a real config from the other .toml files lying around a harness
			// root (loadout.toml, match*.toml, RLBotSim's Cargo.toml), none of which
			// carry one.
			std::string path = std::filesystem::absolute(it->path(), ec).lexically_normal().string();
			if (ReadTomlString(path, "run_command_linux").empty()
				&& ReadTomlString(path, "run_command").empty())
				continue;

			// Roots may overlap or be listed twice; the same file must not appear twice in
			// a dropdown where selecting it launches a process.
			if (std::any_of(found.begin(), found.end(),
					[&](const VizBotEntry& e) { return e.path == path; }))
				continue;

			std::string name = ReadTomlString(path, "name");
			if (name.empty())
				name = it->path().parent_path().filename().string();
			found.push_back({ path, name });
			entryRoot.push_back(searchRoot.string());
		}
	}

	// Two checkouts can hold configs calling themselves the same thing (both harness roots
	// have a "Pulsar2"), and a dropdown with two identical rows is a dropdown you cannot
	// use. The distinguishing fact is which checkout it came from, so say that.
	// Flag first, rename second: renaming in one pass would make the first of a pair stop
	// matching the second, and only one of the two would get disambiguated.
	std::vector<bool> ambiguous(found.size(), false);
	for (size_t i = 0; i < found.size(); i++)
		for (size_t k = i + 1; k < found.size(); k++)
			if (found[k].name == found[i].name)
				ambiguous[i] = ambiguous[k] = true;
	for (size_t i = 0; i < found.size(); i++) {
		if (!ambiguous[i])
			continue;
		std::string where = std::filesystem::path(entryRoot[i]).parent_path().filename().string();
		if (!where.empty())
			found[i].name += " (" + where + ")";
	}

	std::sort(found.begin(), found.end(), [](const VizBotEntry& a, const VizBotEntry& b) {
		return a.name != b.name ? a.name < b.name : a.path < b.path;
	});
	return found;
}

bool VizRLBotServer::LaunchBot(const std::string& botConfigPath, Team team, int numCars) {
	StopBot();

	if (!std::filesystem::exists(botConfigPath)) {
		lastError = "no such bot config: " + botConfigPath;
		return false;
	}
	if (impl->listenFd < 0) {
		lastError = "server is not listening";
		return false;
	}

	// run_command is the WINDOWS command; RLBot configs carry the Linux one separately
	// and it is what must run here. Nexto's, for instance, is a backslashed
	// `..\..\venv\Scripts\python`, which a shell on this box happily fails to find.
	std::string runCommand = ReadTomlString(botConfigPath, "run_command_linux");
	if (runCommand.empty())
		runCommand = ReadTomlString(botConfigPath, "run_command");
	if (runCommand.empty()) {
		lastError = "bot.toml has no run_command_linux or run_command";
		return false;
	}
	std::string agentId = ReadTomlString(botConfigPath, "agent_id");
	if (agentId.empty())
		agentId = "pulsar/viz-opponent";
	botName = ReadTomlString(botConfigPath, "name");
	if (botName.empty())
		botName = std::filesystem::path(botConfigPath).parent_path().filename().string();

	impl->botTeam = team;
	impl->numCars = numCars;
	impl->agentId = agentId;

	// The bot's run_command is relative to its own config directory, same as RLBot core
	// treats it.
	const std::string workDir = std::filesystem::path(botConfigPath).parent_path().string();

	// One process per car. RLBot only gives a single process multiple cars when the
	// bot declares hivemind; Nexto and friends do not, so N cars means N agents.
	for (int car = 0; car < numCars; car++) {
		pid_t pid = fork();
		if (pid < 0) {
			lastError = "fork() failed";
			return false;
		}

		if (pid == 0) {
			// -- child --
			// New process group so StopBot can take down the bot AND anything it spawned
			// (RLBot bots are often a shell script wrapping a Python process).
			setpgid(0, 0);
			if (!workDir.empty())
				(void)!chdir(workDir.c_str());

			setenv("RLBOT_SERVER_PORT", std::to_string(port).c_str(), 1);
			setenv("RLBOT_SERVER_IP", "127.0.0.1", 1);
			setenv("RLBOT_AGENT_ID", agentId.c_str(), 1);

			execl("/bin/sh", "sh", "-c", runCommand.c_str(), (char*)NULL);
			_exit(127); // exec failed; the parent notices via the dead connection
		}

		setpgid(pid, pid); // race-free with the child's own setpgid
		impl->botPids.push_back(pid);
	}

	VIZ_LOG("launched " << numCars << "x \"" << botName << "\" on team "
		<< (team == Team::BLUE ? "BLUE" : "ORANGE"));
	return true;
}

void VizRLBotServer::StopBot() {
	impl->DropAllClients();

	// The whole group per pid: a bot started through a shell wrapper leaves the real
	// process behind if only the shell is signalled.
	for (pid_t pid : impl->botPids)
		if (pid > 0) kill(-pid, SIGTERM);

	for (int i = 0; i < 50 && !impl->botPids.empty(); i++) {
		bool anyAlive = false;
		for (pid_t pid : impl->botPids) {
			if (pid <= 0) continue;
			int status = 0;
			pid_t r = waitpid(pid, &status, WNOHANG);
			if (r == 0) anyAlive = true;
		}
		if (!anyAlive) break;
		usleep(20000);
	}
	for (pid_t pid : impl->botPids) {
		if (pid <= 0) continue;
		kill(-pid, SIGKILL);
		waitpid(pid, NULL, WNOHANG);
	}
	if (!impl->botPids.empty())
		VIZ_LOG("stopped " << impl->botPids.size() << " bot process(es)");
	impl->botPids.clear();
	botName.clear();
}

// Connected only once EVERY car has an agent: a half-connected 2v2 is not a match
// anyone wants to read a scoreline off.
bool VizRLBotServer::Connected() const {
	if (impl->clients.empty())
		return false;
	for (const BotClient& c : impl->clients)
		if (c.fd < 0 || c.ownedRow < 0)
			return false;
	return (int)impl->clients.size() >= impl->numCars;
}

bool VizRLBotServer::BotRunning() const {
	bool any = false;
	for (pid_t& pid : const_cast<Impl*>(impl.get())->botPids) {
		if (pid <= 0)
			continue;
		// Reap so a crashed bot stops reporting as running.
		int status = 0;
		if (waitpid(pid, &status, WNOHANG) == pid) {
			VIZ_LOG("bot pid " << pid << " exited");
			pid = -1;
			continue;
		}
		any = true;
	}
	return any;
}

// ---- packet construction -------------------------------------------------------

static flat::Vector3 ToFlat(const Vec& v) { return flat::Vector3(v.x, v.y, v.z); }

static flat::Rotator ToFlatRot(const RotMat& m) {
	// RLBot wants pitch/yaw/roll; derive them from the basis the same way RocketSim's
	// own Angle conversion does, so a bot sees exactly the orientation the sim has.
	Angle a = Angle::FromRotMat(m);
	return flat::Rotator(a.pitch, a.yaw, a.roll);
}

// Physics is a flatbuffers STRUCT (fixed layout, by value), not a table — hence the
// by-value construction rather than a ...T native table like everything else here.
static std::unique_ptr<flat::Physics> MakePhysics(const PhysState& p) {
	return std::make_unique<flat::Physics>(
		ToFlat(p.pos), ToFlatRot(p.rotMat), ToFlat(p.vel), ToFlat(p.angVel));
}

// Everything an agent needs before it can play, sent once per connection: which car it
// drives, the shape of the match, and the static field.
void VizRLBotServer::SendSessionInfo(BotClient& client, const RLGC::GameState& state,
	const std::vector<int>& opponentIndices) {

	{	// ControllableTeamInfo: exactly ONE car, because this is one agent process.
		flatbuffers::FlatBufferBuilder fbb;
		auto info = std::make_unique<flat::ControllableTeamInfoT>();
		info->team = (uint32_t)impl->botTeam;
		auto c = std::make_unique<flat::ControllableInfoT>();
		// The index the agent stamps on its PlayerInput; making it the arena row means
		// its inputs address a car directly.
		c->index = (uint32_t)client.ownedRow;
		c->identifier = (int32_t)(client.ownedRow + 1);
		info->controllables.push_back(std::move(c));

		flat::CorePacketT wrapper;
		wrapper.message.Set(std::move(*info));
		fbb.Finish(flat::CorePacket::Pack(fbb, &wrapper));
		impl->Send(client, fbb);
	}

	{	// MatchConfiguration. Not optional in practice: the bot manager only runs the
		// bot's initialize() once it has this, and without it a bot happily receives
		// packets and then dies inside get_output on state it never set up (measured
		// with Nexto: "'Nexto' object has no attribute 'game_state'").
		//
		// It describes the match the agent THINKS it is in. Only the shape matters —
		// teams and player ids — since the viewer's arena is the real authority.
		flatbuffers::FlatBufferBuilder fbb;
		auto cfg = std::make_unique<flat::MatchConfigurationT>();
		cfg->game_mode = flat::GameMode::Soccar;
		cfg->launcher = flat::Launcher::NoLaunch;
		// We already started the agents ourselves, and there is nothing to wait for.
		cfg->auto_start_agents = false;
		cfg->wait_for_agents = false;
		cfg->instant_start = true;
		cfg->existing_match_behavior = flat::ExistingMatchBehavior::ContinueAndSpawn;
		cfg->enable_state_setting = false;
		cfg->freeplay = false;

		for (size_t row = 0; row < state.players.size(); row++) {
			const RLGC::Player& player = state.players[row];
			auto pc = std::make_unique<flat::PlayerConfigurationT>();
			pc->team = (uint32_t)player.team;
			pc->player_id = (int32_t)player.carId;

			// Every car is presented as a custom bot; the viewer's own policy has no
			// RLBot identity, and the agent needs a full roster to reason about the match.
			auto bot = std::make_unique<flat::CustomBotT>();
			bot->name = "car" + std::to_string(player.carId);
			bot->root_dir = "";
			bot->run_command = "";
			bot->agent_id = (std::find(opponentIndices.begin(), opponentIndices.end(),
				(int)row) != opponentIndices.end()) ? impl->agentId : "pulsar/self";
			pc->variety.Set(std::move(*bot));
			cfg->player_configurations.push_back(std::move(pc));
		}

		flat::CorePacketT wrapper;
		wrapper.message.Set(std::move(*cfg));
		fbb.Finish(flat::CorePacket::Pack(fbb, &wrapper));
		impl->Send(client, fbb);
	}

	{	// FieldInfo: static geometry. Boost pads must be in the same order as the
		// GamePacket's boost_pads array, or agents read the wrong pad's state.
		flatbuffers::FlatBufferBuilder fbb;
		auto field = std::make_unique<flat::FieldInfoT>();
		for (int i = 0; i < RLGC::CommonValues::BOOST_LOCATIONS_AMOUNT; i++) {
			const Vec& loc = RLGC::CommonValues::BOOST_LOCATIONS[i];
			auto pad = std::make_unique<flat::BoostPadT>();
			pad->location = std::make_unique<flat::Vector3>(ToFlat(loc));
			// Big pads are the full-boost ones; RocketSim marks them by height.
			pad->is_full_boost = loc.z > 72.f;
			field->boost_pads.push_back(std::move(pad));
		}
		for (int team = 0; team < 2; team++) {
			auto goal = std::make_unique<flat::GoalInfoT>();
			goal->team_num = team;
			const float y = (team == 0 ? -1.f : 1.f) * RLGC::CommonValues::BACK_WALL_Y;
			goal->location = std::make_unique<flat::Vector3>(flat::Vector3(0, y, 321.3875f));
			goal->direction = std::make_unique<flat::Vector3>(
				flat::Vector3(0, team == 0 ? 1.f : -1.f, 0));
			goal->width = 892.755f;
			goal->height = 642.775f;
			field->goals.push_back(std::move(goal));
		}
		flat::CorePacketT wrapper;
		wrapper.message.Set(std::move(*field));
		fbb.Finish(flat::CorePacket::Pack(fbb, &wrapper));
		impl->Send(client, fbb);
	}
}

void VizRLBotServer::SendGamePacket(BotClient& client, const RLGC::GameState& state) {
	flatbuffers::FlatBufferBuilder fbb;
	auto packet = std::make_unique<flat::GamePacketT>();

	for (const RLGC::Player& player : state.players) {
		auto info = std::make_unique<flat::PlayerInfoT>();
		info->physics = MakePhysics(player);
		info->team = (uint32_t)player.team;
		info->boost = player.boost;
		info->is_bot = true;
		info->name = "car" + std::to_string(player.carId);
		info->player_id = (int32_t)player.carId;
		info->is_supersonic = player.isSupersonic;

		// score_info, hitbox and dodge_dir are REQUIRED by the schema. Leaving them null
		// produces a packet that serialises fine and then fails verification inside the
		// AGENT ("Missing required field"), killing the bot rather than us. The viewer
		// keeps no per-player scoreline, so it reports zeroes rather than inventing them.
		info->score_info = std::make_unique<flat::ScoreInfo>(0, 0, 0, 0, 0, 0, 0);

		auto hitbox = std::make_unique<flat::BoxShapeT>();
		hitbox->length = 118.0074f; // Octane, the only body this run trains on
		hitbox->width = 84.2f;
		hitbox->height = 36.16f;
		info->hitbox = std::move(hitbox);
		info->hitbox_offset = std::make_unique<flat::Vector3>(
			flat::Vector3(13.87566f, 0.f, 20.755f));

		const RLGC::Action& prev = player.prevAction;
		info->last_input = std::make_unique<flat::ControllerState>(
			prev[0], prev[1], prev[2], prev[3], prev[4],
			prev[5] != 0.f, prev[6] != 0.f, prev[7] != 0.f, false);
		// A car that is not dodging has no dodge direction; zero is what RLBot reports.
		info->dodge_dir = std::make_unique<flat::Vector2>(flat::Vector2(0.f, 0.f));

		info->air_state = player.isOnGround ? flat::AirState::OnGround : flat::AirState::InAir;
		// Negative means "not applicable" in RLBot's timeout convention.
		info->demolished_timeout = player.isDemoed ? player.demoRespawnTimer : -1.f;
		info->dodge_timeout = player.HasFlipOrJump() ? 1.f : -1.f;
		info->has_dodged = player.hasFlipped;
		info->has_jumped = player.hasJumped;
		info->has_double_jumped = player.hasDoubleJumped;
		packet->players.push_back(std::move(info));
	}

	{
		auto ball = std::make_unique<flat::BallInfoT>();
		ball->physics = MakePhysics(state.ball);
		auto shape = std::make_unique<flat::SphereShapeT>();
		shape->diameter = RLGC::CommonValues::BALL_RADIUS * 2;
		ball->shape.Set(std::move(*shape));
		packet->balls.push_back(std::move(ball));
	}

	for (size_t i = 0; i < state.boostPads.size(); i++) {
		// The timer counts DOWN to availability, which is what boostPadTimers holds.
		packet->boost_pads.push_back(flat::BoostPadState(
			state.boostPads[i], state.boostPadTimers[i]));
	}

	for (int team = 0; team < 2; team++)
		packet->teams.push_back(flat::TeamInfo((uint32_t)team, 0));

	auto match = std::make_unique<flat::MatchInfoT>();
	// Always Active from the agent's point of view: the viewer has no countdown or
	// replay phase, and a paused arena simply stops sending packets rather than
	// announcing a pause the bot would have to reason about.
	match->match_phase = flat::MatchPhase::Active;
	match->world_gravity_z = -650.f;
	match->game_speed = 1.f;
	match->is_unlimited_time = true;
	match->seconds_elapsed = (float)state.lastTickCount / 120.f;
	match->game_time_remaining = 300.f;
	match->frame_num = (uint32_t)state.lastTickCount;
	packet->match_info = std::move(match);

	flat::CorePacketT wrapper;
	wrapper.message.Set(std::move(*packet));
	fbb.Finish(flat::CorePacket::Pack(fbb, &wrapper));
	impl->Send(client, fbb);
}

void VizRLBotServer::Poll(const RLGC::GameState& state, const std::vector<int>& opponentIndices) {
	if (impl->listenFd < 0)
		return;
	impl->numCars = (int)opponentIndices.size();
	impl->ownedRows = opponentIndices;

	// Accept up to one agent per car; each is handed exactly ONE controllable, in the
	// order they connect. Anything past that is closed rather than left in the backlog
	// pretending to be a player.
	while ((int)impl->clients.size() < impl->numCars) {
		int fd = accept(impl->listenFd, NULL, NULL);
		if (fd < 0)
			break;
		fcntl(fd, F_SETFL, fcntl(fd, F_GETFL, 0) | O_NONBLOCK);
		int one = 1;
		// Controls are tiny and latency-critical; Nagle would batch them into the next tick.
		setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

		BotClient client;
		client.fd = fd;
		client.ownedRow = opponentIndices[impl->clients.size()];
		impl->clients.push_back(std::move(client));
		VIZ_LOG("agent connected for car row " << impl->clients.back().ownedRow
			<< " (" << impl->clients.size() << "/" << impl->numCars << ")");
	}

	for (BotClient& client : impl->clients) {
		if (client.fd < 0)
			continue;

		// -- read whatever arrived --
		uint8_t chunk[8192];
		bool dead = false;
		for (;;) {
			ssize_t n = recv(client.fd, chunk, sizeof(chunk), 0);
			if (n > 0) {
				client.inBuf.insert(client.inBuf.end(), chunk, chunk + n);
				continue;
			}
			if (n == 0) { // orderly shutdown
				VIZ_LOG("agent for row " << client.ownedRow << " disconnected");
				dead = true;
				break;
			}
			if (errno == EAGAIN || errno == EWOULDBLOCK)
				break;
			dead = true;
			break;
		}
		if (dead) {
			close(client.fd);
			client.fd = -1;
			client.haveInput = false;
			continue;
		}

		// -- reassemble and handle --
		size_t offset = 0;
		while (client.inBuf.size() - offset >= HEADER_SIZE) {
			const size_t size = ((size_t)client.inBuf[offset] << 8) | client.inBuf[offset + 1];
			if (client.inBuf.size() - offset - HEADER_SIZE < size)
				break; // partial message; wait for the rest
			const uint8_t* payload = client.inBuf.data() + offset + HEADER_SIZE;
			offset += HEADER_SIZE + size;

			flatbuffers::Verifier verifier(payload, size);
			if (!flat::VerifyInterfacePacketBuffer(verifier))
				continue; // not ours / corrupt; skip this frame rather than drop the link

			const flat::InterfacePacket* packet = flat::GetInterfacePacket(payload);
			if (!packet)
				continue;

			switch (packet->message_type()) {
			case flat::InterfaceMessage::ConnectionSettings:
				// Said hello: (re)send everything it needs before it can play.
				client.sentSessionInfo = false;
				break;

			case flat::InterfaceMessage::PlayerInput: {
				const flat::PlayerInput* input = packet->message_as_PlayerInput();
				if (!input || !input->controller_state())
					break;
				// Trust the connection, not the index it claims: each process owns one
				// car here, and a bot that misreports would otherwise drive someone else.
				if ((int)input->player_index() != client.ownedRow)
					break;

				const flat::ControllerState* c = input->controller_state();
				RLGC::Action& a = client.control;
				a[0] = c->throttle();
				a[1] = c->steer();
				a[2] = c->pitch();
				a[3] = c->yaw();
				a[4] = c->roll();
				a[5] = c->jump() ? 1.f : 0.f;
				a[6] = c->boost() ? 1.f : 0.f;
				a[7] = c->handbrake() ? 1.f : 0.f;
				client.haveInput = true;
				break;
			}
			default:
				// StartCommand, MatchComm, DesiredGameState, rendering, ping... optional
				// traffic a playing bot may emit that we have no opinion about.
				break;
			}
		}
		if (offset > 0)
			client.inBuf.erase(client.inBuf.begin(), client.inBuf.begin() + offset);

		// -- session info, once per connection --
		if (!client.sentSessionInfo) {
			SendSessionInfo(client, state, opponentIndices);
			client.sentSessionInfo = true;
		}

		// -- per-step GamePacket --
		SendGamePacket(client, state);
	}
}

bool VizRLBotServer::GetControls(std::vector<RLGC::Action>& out, std::vector<uint8_t>& outValid) const {
	// Per-car validity, not all-or-nothing. A 2v2 where only one agent has connected
	// must leave the other car to the policy — the first version overrode both and the
	// un-driven one sat frozen on a zeroed control.
	outValid.assign(out.size(), 0);
	bool any = false;

	for (const BotClient& client : impl->clients) {
		if (client.fd < 0 || !client.haveInput || client.ownedRow < 0)
			continue;
		for (size_t i = 0; i < impl->ownedRows.size(); i++) {
			if (impl->ownedRows[i] != client.ownedRow || i >= out.size())
				continue;
			out[i] = client.control;
			outValid[i] = 1;
			any = true;
		}
	}
	return any;
}
