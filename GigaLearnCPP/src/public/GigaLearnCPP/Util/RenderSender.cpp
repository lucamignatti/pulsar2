#include "RenderSender.h"

#include <nlohmann/json.hpp>
#include <cmath>

using namespace nlohmann;
using namespace RLGC;

GGL::RenderSender::RenderSender(float timeScale) : timeScale(timeScale) {
	RG_LOG("Initializing RenderSender...");

	if (timeScale <= 0 || !std::isfinite(timeScale))
		RG_ERR_CLOSE("RenderSender timeScale must be positive and finite");

	try {
		RG_LOG("Current dir: " << std::filesystem::current_path());
		pyMod = pybind11::module::import("python_scripts.render_receiver");
	} catch (std::exception& e) {
		RG_ERR_CLOSE("RenderSender: Failed to import render receiver, exception: " << e.what());
	}

	RG_LOG(" > RenderSender initalized.");
}

FList VecToList(const Vec& vec) {
	return FList({ vec.x, vec.y, vec.z });
}

json PhysToJSON(const PhysState& obj) {
	json j = {};

	j["pos"] = VecToList(obj.pos);

	j["forward"] = VecToList(obj.rotMat.forward);
	j["right"] = VecToList(obj.rotMat.right);
	j["up"] = VecToList(obj.rotMat.up);

	j["vel"] = VecToList(obj.vel);
	j["ang_vel"] = VecToList(obj.angVel);

	return j;
}

json PlayerToJSON(const Player& player) {
	json j = {};

	j["car_id"] = player.carId;
	j["team_num"] = (int)player.team;
	j["phys"] = PhysToJSON(player);
	j["is_demoed"] = player.isDemoed;
	j["on_ground"] = player.isOnGround;
	j["ball_touched"] = player.ballTouchedStep;
	j["has_flip"] = player.HasFlipOrJump();
	j["boost_amount"] = player.boost / 100;

	return j;
}

json GameStateToJSON(const GameState& state) {
	json j = {};
	
	j["ball"] = PhysToJSON(state.ball);

	std::vector<json> players;
	for (auto& player : state.players)
		players.push_back(PlayerToJSON(player));

	j["players"] = players;
	j["boost_pads"] = state.boostPads;

	return j;
}

std::vector<json> ActionSetToJSON(const std::vector<Action>& actions) {
	std::vector<json> js = {};
	for (auto& action : actions) {
		FList vals  = FList(action.begin(), action.end());
		js.push_back(json(vals));
	}

	return js;
}

void GGL::RenderSender::Send(const GameState& state) {
	json j = {};
	j["gamemode"] = state.lastArena ? GAMEMODE_STRS[(int)state.lastArena->gameMode] : "soccar";
	j["state"] = GameStateToJSON(state);

	std::vector<Action> actions = {};
	for (auto& player : state.players)
		actions.push_back(player.prevAction);

	j["actions"] = ActionSetToJSON(actions);
	
	std::string jStr = j.dump();

	try {
		pyMod.attr("render_state")(jStr);
	} catch (std::exception& e) {
		RG_ERR_CLOSE("RenderSender: Failed to send gamestate, exception: " << e.what());
	}

	// Delay
	{
		namespace chr = std::chrono;

		// Determine the desired delay and the actual delay (in seconds)
		double targetDelay = state.deltaTime / timeScale;
		double realDelay = renderTimer.Elapsed();
		renderTimer.Reset();

		constexpr double CORRECTION_SCALE = 0.3f; // Portion of the error we wil compensate for each step
		double error = targetDelay - realDelay;

		if (adaptiveRenderDelay == -1) {
			// Just initialize render delay as target delay
			adaptiveRenderDelay = targetDelay;
		} else {
			adaptiveRenderDelay += error * CORRECTION_SCALE;
		}
		adaptiveRenderDelay = RS_CLAMP(adaptiveRenderDelay, 0, targetDelay);

		// Sleep for the new adaptive delay
		int64_t sleepMics = (int64_t)(adaptiveRenderDelay * 1'000'000);
		std::this_thread::sleep_for(chr::microseconds(sleepMics));
	}
}

GGL::RenderSender::~RenderSender() {}
