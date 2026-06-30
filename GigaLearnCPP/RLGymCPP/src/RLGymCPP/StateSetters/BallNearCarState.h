#pragma once
#include "StateSetter.h"

namespace RLGC {
	// Cold-touch bootstrap setter. Places the ball low + STATIONARY, then places each car ON THE GROUND
	// within a horizontal radius band [radiusMin, radiusMax] of the ball, at rest, facing a random heading.
	//
	// Unlike RandomState -- which draws the car and ball as two INDEPENDENT uniform points over the field, so
	// P(car within a car-length of the ball) ~ 0.1% (the same order as a frozen bot's touch rate) -- the car
	// here is placed RELATIVE to the ball. So a random policy gets a near-ball, slow-ball strike opportunity
	// EVERY reset, and the impulse-on-touch reward (StrongTouch = |dBallVel|, whiff = 0; action-attributable,
	// not value-absorbed) fires orders of magnitude more than from kickoff-only resets. This is the RLGym
	// community's "place the bot near the ball" cold-start, the bootstrap our kickoff-only setter never gave.
	//
	// radiusMin/radiusMax are PUBLIC so a later near->far competence curriculum can widen them as the
	// touch-ratio EMA rises (start tight for guaranteed cold touches, widen toward drive-across-field).
	class BallNearCarState : public StateSetter {
	public:
		float radiusMin, radiusMax;

		BallNearCarState(float radiusMin = 300.f, float radiusMax = 900.f) :
			radiusMin(radiusMin), radiusMax(radiusMax) {
		}

		virtual void ResetArena(Arena* arena);
	};
}
