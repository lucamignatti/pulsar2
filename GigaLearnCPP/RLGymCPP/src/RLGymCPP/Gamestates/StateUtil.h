#pragma once
#include "../Framework.h"
#include "../BasicTypes/Action.h"
#include "Player.h"

namespace RLGC {
	PhysState InvertPhys(const PhysState& physState, bool shouldInvert = true);
	PhysState MirrorPhysX(const PhysState& physState, bool shouldMirror = true);
	Action MirrorActionX(const Action& action, bool shouldMirror = true);
	bool ShouldMirrorXForPlayer(const Player& player);
}
