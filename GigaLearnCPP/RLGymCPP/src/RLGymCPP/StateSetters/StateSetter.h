#pragma once
#include "../Gamestates/GameState.h"

namespace RLGC {
	class StateSetter {
	public:
		virtual ~StateSetter() = default;

		virtual void ResetArena(Arena* arena) = 0;
	};
}
