#pragma once
#include "Report.h"
#include <pybind11/pybind11.h>
#include <RLGymCPP/Gamestates/GameState.h>
#include <RLGymCPP/BasicTypes/Action.h>
#include <GigaLearnCPP/Util/Timer.h>

namespace GGL {
	struct RG_IMEXPORT RenderSender {
		pybind11::module pyMod;

		float timeScale;
		double adaptiveRenderDelay = -1;
		Timer renderTimer = {};

		RenderSender(float timeScale);

		RG_NO_COPY(RenderSender);

		// `controlJson`, when non-empty, is a JSON object forwarded verbatim to the page
		// as the frame's "pulsar" field — the control panel's view of transport state.
		// `pace` false skips the frame-pacing sleep: a paused viewer is re-sending the
		// same state to keep the page live, not simulating, so it must not also sleep a
		// simulated frame's worth of wall clock.
		void Send(const RLGC::GameState& state, const std::string& controlJson = "", bool pace = true);

		~RenderSender();
	};
}