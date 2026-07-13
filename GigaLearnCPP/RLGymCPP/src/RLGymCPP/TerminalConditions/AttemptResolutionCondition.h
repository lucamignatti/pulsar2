#pragma once
#include "TerminalCondition.h"

namespace RLGC {
	// Steered-practice episode boundary: end the episode the moment an aerial-ball attempt
	// RESOLVES, so whatever follows (the counterattack that punishes a whiff) never enters the
	// return. Deliberately a NORMAL terminal, NOT a truncation: GAE must not bootstrap V at the
	// cut - the critic knows post-whiff states precede concessions, so a truncation-bootstrap
	// would re-inject the punishment this condition exists to delete.
	//
	// v1 resolution rule (attach only to practice arenas):
	//   ARM     when the ball rises above armZ (an airborne play exists)
	//   RESOLVE when, while armed: any player touches the ball  (contest happened),
	//           the ball comes back down below groundZ           (nobody went - whiff/decline),
	//           or the play stays unresolved for maxArmedSec     (degenerate ceiling rolls etc.)
	// Ground play never terminates here; GoalScore/NoTouch conditions still apply.
	class AttemptResolutionCondition : public TerminalCondition {
	public:
		float armZ, groundZ, maxArmedSec;

		bool armed = false;
		uint64_t armTick = 0;

		AttemptResolutionCondition(float armZ = 300, float groundZ = 150, float maxArmedSec = 6)
			: armZ(armZ), groundZ(groundZ), maxArmedSec(maxArmedSec) {
		}

		virtual void Reset(const GameState& initialState) override {
			// A reset can drop the ball anywhere (RandomState spawns it airborne); arm from the
			// first post-reset step so drills starting mid-air resolve like anything else
			armed = false;
		}

		virtual bool IsTerminal(const GameState& currentState) override {
			float ballZ = currentState.ball.pos.z;

			if (!armed) {
				if (ballZ > armZ) {
					armed = true;
					armTick = currentState.lastTickCount;
				}
				return false;
			}

			for (auto& player : currentState.players)
				if (player.ballTouchedStep)
					return true; // contest resolved the attempt (pay/whiff already in the return)

			if (ballZ < groundZ)
				return true; // ball came down untouched - the attempt was declined or missed

			if ((currentState.lastTickCount - armTick) > (uint64_t)(maxArmedSec * 120))
				return true; // stuck airborne play (ceiling crawl etc.) - resolve by fiat

			return false;
		}

		virtual bool IsTruncation() override {
			return false; // TRUE terminal: no value bootstrap past the resolution (see header)
		}
	};
}
