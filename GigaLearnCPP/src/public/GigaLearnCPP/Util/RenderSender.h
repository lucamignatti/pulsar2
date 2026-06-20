#pragma once
#include "RenderSink.h"
#include "PythonRuntime.h"
#include "Report.h"
#include <pybind11/pybind11.h>
#include <RLGymCPP/Gamestates/GameState.h>
#include <RLGymCPP/BasicTypes/Action.h>
#include <GigaLearnCPP/Util/Timer.h>

namespace GGL {
	// Python-renderer adapter for RenderSink.
	struct RG_IMEXPORT RenderSender : RenderSink {
		// Owns the interpreter; declared first so it is constructed before (and destroyed after) pyMod.
		PythonRuntime _pyRuntime;

		pybind11::module pyMod;

		float timeScale;
		double adaptiveRenderDelay = -1;
		Timer renderTimer = {};

		RenderSender(float timeScale);

		RG_NO_COPY(RenderSender);

		void Send(const RLGC::GameState& state) override;

		~RenderSender();
	};
}