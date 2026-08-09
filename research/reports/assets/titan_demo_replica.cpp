bool CarWithinForwardEllipticalCone(
	const Vec& carForward, const Vec& carRight, const Vec& carUp,
	const Vec& impactDir,
	float maxYawDeg, float maxPitchDeg,
	bool reverseForward)
{
	const float fudge = 1.01f;  // exactly as in the original code

	// Optionally flip the forward direction (for backwards demos)
	Vec forward = reverseForward ? -carForward : carForward;

	// ----- Pitch check (forward‑up plane) -----
	// Remove the right component with a slightly scaled projection
	float dotRight = impactDir.Dot(carRight);
	float scaledRight = (dotRight >= 0.0f) ? (dotRight / fudge) : (dotRight * fudge);
	Vec projFP = impactDir - carRight * scaledRight;  // roughly in forward‑up plane

	// Normalise the projected vector (safe against zero length)
	float lenSqFP = projFP.LengthSq();
	Vec dirFP;
	if (std::abs(lenSqFP - 1.0f) < 1e-6f) {
		dirFP = projFP;                       // already unit
	}
	else if (lenSqFP >= 1e-9f) {
		dirFP = projFP.Normalized();
	}
	else {
		dirFP = Vec(0, 0, 0);                  // degenerate case
	}

	float dotFwdPitch = forward.Dot(dirFP);
	float cosPitch = std::clamp(dotFwdPitch, -1.0f, 1.0f);
	float pitchRad = std::acos(cosPitch);
	float pitchDeg = pitchRad * 57.29578f;    // radians → degrees

	if (pitchDeg > maxPitchDeg)
		return false;

	// ----- Yaw check (forward‑right plane) -----
	// Remove the up component with the same fudge factor
	float dotUp = impactDir.Dot(carUp);
	float scaledUp = (dotUp >= 0.0f) ? (dotUp / fudge) : (dotUp * fudge);
	Vec projFY = impactDir - carUp * scaledUp;  // roughly in forward‑right plane

	float lenSqFY = projFY.LengthSq();
	Vec dirFY;
	if (std::abs(lenSqFY - 1.0f) < 1e-6f) {
		dirFY = projFY;
	}
	else if (lenSqFY >= 1e-9f) {
		dirFY = projFY.Normalized();
	}
	else {
		dirFY = Vec(0, 0, 0);
	}

	float dotFwdYaw = forward.Dot(dirFY);
	float cosYaw = std::clamp(dotFwdYaw, -1.0f, 1.0f);
	float yawRad = std::acos(cosYaw);
	float yawDeg = yawRad * 57.29578f;

	return (yawDeg <= maxYawDeg);
}

void Arena::_BtCallback_OnCarCarCollision(Car* car1, Car* car2, btManifoldPoint& manifoldPoint) {
	using namespace RLConst;

	// Manually override manifold friction/restitution
	manifoldPoint.m_combinedFriction = RLConst::CARCAR_COLLISION_FRICTION;
	manifoldPoint.m_combinedRestitution = RLConst::CARCAR_COLLISION_RESTITUTION;

	car1->_internalState.touchingCar.otherCarID = car2->id;
	car2->_internalState.touchingCar.otherCarID = car1->id;
	car1->_internalState.touchingCar.touchTime = 0;
	car2->_internalState.touchingCar.touchTime = 0;

	// Deferred action storage
	struct PendingAction {
		Car* attacker;
		Car* victim;
		bool isDemo;
		btVector3 bumpImpulseBT;   // only used when isDemo == false
	};
	PendingAction pendingActions[2];
	int numPending = 0;

	Vec targetCollisionPoint = BT_TO_UU * manifoldPoint.getPositionWorldOnB();
	Vec attackerCollisionPoint = BT_TO_UU * manifoldPoint.getPositionWorldOnA();

	// Test collision both ways, but only mark actions instead of executing immediately
	for (int i = 0; i < 2; i++) {
		bool isSwapped = (i == 1);

		if (isSwapped) {
			std::swap(car1, car2);
			std::swap(targetCollisionPoint, attackerCollisionPoint);
		}

		CarState state = car1->GetState(),
			otherState = car2->GetState();

		// Skip if either car is already demolished
		if (state.isDemoed || otherState.isDemoed)
			continue;

		// Cooldown check
		if ((state.carContact.otherCarID == car2->id) && (state.carContact.cooldownTimer > 0))
			continue;

		
		if (state.vel.Length() > 0) {

			Vec velDir = state.vel.Normalized();
			Vec dirToCollisionPoint = (targetCollisionPoint - state.pos).Normalized();
			Vec dirToCenter = (otherState.pos - state.pos).Normalized();

			float speedTowardsOtherCar = state.vel.Dot(dirToCollisionPoint);
			float otherCarAwaySpeed = otherState.vel.Dot(velDir);

			if (speedTowardsOtherCar > 0 && speedTowardsOtherCar > otherCarAwaySpeed) { // Faster than they are moving away
				// Determine if this would be a demo (mode + supersonic + team check)
				bool isDemo;
				switch (_mutatorConfig.demoMode) {
				case DemoMode::ON_CONTACT: isDemo = true; break;
				case DemoMode::DISABLED:   isDemo = false; break;
				default:                    isDemo = state.isSupersonic;
				}
				if (isDemo && !_mutatorConfig.enableTeamDemos)
					isDemo = (car1->team != car2->team);

				// COM elliptical cone check with demotion
				Vec carForward = car1->GetForwardDir();
				Vec carRight = car1->GetRightDir();
				Vec carUp = car1->GetUpDir();

				bool bReverseCarForward = false;
				//if (_mutatorConfig.bAllowBackwardsDemolitions)
				bReverseCarForward = (state.vel.Dot(carForward) < 0.0f);

				// Check forward-aligned speed requirement (absolute value for backwards demos)
				float forwardAlignedSpeed = fabsf(state.vel.Dot(carForward));
				const float MIN_DEMO_SPEED = 2100.0f;
				bool hasDemoSpeed = (forwardAlignedSpeed >= MIN_DEMO_SPEED);

				// Demote demo attempt to bump if speed requirement not met
				if (isDemo && !hasDemoSpeed) {
					isDemo = false;
				}

				// Angle limits
				const float demoYaw = 45.572994f;
				const float demoPitch = 36.869896f;
				const float bumpYaw = 70.0f;
				const float bumpPitch = 36.869896f;

				bool passed = false;

				if (isDemo) {
					// First try the strict demo cone
					passed = CarWithinForwardEllipticalCone(
						carForward, carRight, carUp, dirToCenter,
						demoYaw, demoPitch, bReverseCarForward);

					if (!passed) {
						// Demotion: check the wider bump cone
						passed = CarWithinForwardEllipticalCone(
							carForward, carRight, carUp, dirToCenter,
							bumpYaw, bumpPitch, bReverseCarForward);
						if (passed) {
							isDemo = false;   // convert to bump
						}
					}
				}
				else {
					// Not a demo attempt – only bump cone matters
					passed = CarWithinForwardEllipticalCone(
						carForward, carRight, carUp, dirToCenter,
						bumpYaw, bumpPitch, bReverseCarForward);
				}

				if (passed) {
					// ---- Record the intended action instead of executing ----
					PendingAction action;
					action.attacker = car1;
					action.victim = car2;
					action.isDemo = isDemo;

					if (!isDemo) {
						bool groundHit = car2->_internalState.isOnGround;

						float baseScale =
							(groundHit ? BUMP_VEL_AMOUNT_GROUND_CURVE : BUMP_VEL_AMOUNT_AIR_CURVE)
							.GetOutput(speedTowardsOtherCar);

						Vec hitUpDir = otherState.isOnGround ? (Vec)car2->GetUpDir() : Vec(0, 0, 1);

						Vec bumpImpulse =
							velDir * baseScale +
							hitUpDir * BUMP_UPWARD_VEL_AMOUNT_CURVE.GetOutput(speedTowardsOtherCar)
							* _mutatorConfig.bumpForceScale;

						action.bumpImpulseBT = bumpImpulse * UU_TO_BT;
					}

					pendingActions[numPending++] = action;
				}

			}
		}
	}

	// ---- Apply all recorded actions ----
	for (int i = 0; i < numPending; i++) {
		PendingAction& action = pendingActions[i];

		if (action.isDemo) {
			// Demolish the victim
			action.victim->Demolish(_mutatorConfig.respawnDelay);
		}
		else {
			// Apply bump impulse only if the victim isn't already demolished
			// (it could have been demolished by a mutual demo from the other direction)
			if (!action.victim->GetState().isDemoed) {
				action.victim->_velocityImpulseCache += action.bumpImpulseBT;
			}
		}

		// Set cooldown on the attacker (both demo and bump set cooldown)
		action.attacker->_internalState.carContact.otherCarID = action.victim->id;
		action.attacker->_internalState.carContact.cooldownTimer = _mutatorConfig.bumpCooldownTime;

		// Fire callback for this individual action
		if (_carBumpCallback.func)
			_carBumpCallback.func(this, action.attacker, action.victim, action.isDemo, _carBumpCallback.userInfo);
	}
}
