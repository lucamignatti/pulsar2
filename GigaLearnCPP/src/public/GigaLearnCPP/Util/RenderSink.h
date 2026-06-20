#pragma once
#include <GigaLearnCPP/Framework.h>
#include <RLGymCPP/Gamestates/GameState.h>

namespace GGL {
	// Port for the render side-channel. The production adapter (RenderSender) streams game states to
	// the Python renderer; a headless run supplies its own adapter or none at all.
	struct RG_IMEXPORT RenderSink {
		virtual void Send(const RLGC::GameState& state) = 0;

		virtual ~RenderSink() {}
	};
}
