#include "GameEventTracker.h"

RS_NS_START

static bool GetShooterPasser(Arena* arena, Team team, Car*& shooterOut, bool findPasser, Car*& passerOut, uint64_t maxShooterTicks, uint64_t maxPasserTicks) {
	shooterOut = passerOut = NULL;

	for (Car* car : arena->_cars) {
		if (car->team != team)
			continue;

		if (!car->_ballHitInfo.isValid)
			continue;

		if (car->_ballHitInfo.tickCountWhenHit + maxShooterTicks >= arena->tickCount) {
			if (!shooterOut || car->_ballHitInfo.tickCountWhenHit > shooterOut->_ballHitInfo.tickCountWhenHit) {
				shooterOut = car;
			}
		}
	}

	if (shooterOut && findPasser) {
		uint64_t shootTick = shooterOut->_ballHitInfo.tickCountWhenHit;

		for (Car* car : arena->_cars) {
			if (car->team != team)
				continue;

			if (!car->_ballHitInfo.isValid)
				continue;

			if (car == shooterOut)
				continue;

			if (car->_ballHitInfo.tickCountWhenHit + maxPasserTicks >= shootTick) {
				if (!passerOut || car->_ballHitInfo.tickCountWhenHit > passerOut->_ballHitInfo.tickCountWhenHit) {
					passerOut = car;
				}
			}
		}
	}

	return shooterOut != NULL;
}

void GameEventTracker::Update(Arena* arena) {
	bool scored = arena->IsBallScored();

	float tickrate = arena->GetTickRate();
	uint64_t ballUpdateCount = arena->ball->_updateCounter;

	if (ballUpdateCount > _lastBallUpdateCount || !autoStateSetDetection) {
		// Game is continuing

		uint64_t deltaTicks = ballUpdateCount - _lastBallUpdateCount;
		float deltaTime = deltaTicks * arena->tickTime;

		BallState ballState = arena->ball->GetState();

		if (scored && !_ballScoredLast) {
			Car* shooter;
			Car* passer;
			if (GetShooterPasser(
				arena,
				RS_TEAM_FROM_Y(-ballState.pos.y),
				shooter, true, passer,
				config.goalMaxTouchTime * tickrate,
				config.passMaxTouchTime * tickrate
			)) {

				if (_goalCallback.func)
					_goalCallback.func(arena, shooter, passer, _goalCallback.userInfo);
			}
		} else {
			if (!_ballShot) {

				if (_shotCooldown > 0) {
					_shotCooldown = RS_MAX(_shotCooldown - deltaTime, 0);
				} else {

					float speedSq = ballState.vel.LengthSq();
					if (speedSq >= config.shotMinSpeed * config.shotMinSpeed) {
						Team goalTeam;
						if (arena->IsBallProbablyGoingIn(config.shotMinScoreTime, config.predScoreExtraMargin, &goalTeam)) {
							Team shooterTeam = RS_OPPOSITE_TEAM(goalTeam);

							uint64_t shotMinTouchDelayTicks = config.shotTouchMinDelay * tickrate;

							Car* shooter;
							Car* passer;
							if (GetShooterPasser(
								arena,
								shooterTeam,
								shooter, true, passer,
								deltaTicks + shotMinTouchDelayTicks,
								config.passMaxTouchTime * tickrate
							)) {

								uint64_t ticksSinceHit = arena->tickCount - shooter->_ballHitInfo.tickCountWhenHit;
								if (ticksSinceHit >= shotMinTouchDelayTicks) {

									_ballShot = true;
									_ballShotGoalTeam = goalTeam;
									_shotCooldown = config.shotEventCooldown;
									if (_shotCallback.func)
										_shotCallback.func(arena, shooter, passer, _shotCallback.userInfo);
								}
							}
						}
					}
				}
			} else {
				// Ball is currently shot

				bool willScore = arena->IsBallProbablyGoingIn(config.shotMinScoreTime, config.predScoreExtraMargin);
				if (!willScore) {
					Car* saver;
					Car* _unused;
					if (GetShooterPasser(
						arena,
						_ballShotGoalTeam,
						saver, false, _unused,
						deltaTicks,
						0
					)) {

						if (_saveCallback.func)
							_saveCallback.func(arena, saver, _saveCallback.userInfo);
					}

					_ballShot = false;
				}
			}
		}
	} else if (ballUpdateCount == _lastBallUpdateCount) {
		return;
	} else {
		// Ball update count decreased (state was set): reset persistent info
		ResetPersistentInfo();
	}

	_ballScoredLast = scored;
	_lastBallUpdateCount = ballUpdateCount;
}

void GameEventTracker::ResetPersistentInfo() {
	_ballScoredLast = false;
	_ballShot = false;
	_shotCooldown = 0;
}

RS_NS_END
