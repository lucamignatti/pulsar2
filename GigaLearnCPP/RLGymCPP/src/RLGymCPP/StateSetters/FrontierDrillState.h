#pragma once
#include "StateSetter.h"
#include "../Math.h"

#include <memory>
#include <mutex>

namespace RLGC {
	// Shared reset pool for frontier practice (STEERING_ROADMAP phase 3). The learner
	// mines FEASIBLE-BUT-DECLINED readings (airborne ball someone could have won, nobody
	// went) from each iteration's MATCH rows, reconstructs them from obs-visible
	// quantities (pos/vel/angVel/forward/up/boost - reconstruction error is part of the
	// intended perturbation), and banks them here; FrontierDrillState resets practice
	// arenas into perturbed copies, so practice reps START at the frontier instead of
	// paying the approach. This is the post-mortem-sound successor to stage-2
	// termination: the critic SEES episode starts at resets and learns the drill value
	// profile - no GAE surgery.
	//
	// Offline validation (research/tools/frontier_validate.py): reconstruction is 100%
	// playable, and 250uu / 250uu/s Gaussian perturbation yields balanced coin-flip
	// first-touch races. Entries are staleness-bounded: a pool older than ~2 iterations
	// re-imports the stale-direction problem in state space (roadmap phase-3 guard).
	//
	// Entries live in the CANONICAL frame (self's team attacking +y == BLUE's world
	// frame), cars ordered [self+teammates (BLUE) | opponents (ORANGE)].
	struct FrontierPool {
		struct CarSpawn {
			Vec pos, vel, angVel, forward, up;
			float boost;
		};
		struct Entry {
			BallState ball;
			std::vector<CarSpawn> blueCars, orangeCars;
		};

		std::mutex mutex;
		std::vector<Entry> entries[3]; // per team size (playersPerTeam - 1)
		int64_t filledIteration[3] = { -1000, -1000, -1000 };
		int64_t currentIteration = 0;
		int maxAgeIters = 2;

		void Advance(int64_t iteration) {
			std::lock_guard<std::mutex> g(mutex);
			currentIteration = iteration;
		}

		void Fill(int modeIdx, std::vector<Entry>&& newEntries) {
			std::lock_guard<std::mutex> g(mutex);
			entries[modeIdx] = std::move(newEntries);
			filledIteration[modeIdx] = currentIteration;
		}

		// False when the mode's pool is empty or stale (caller falls back to its normal mix)
		bool Sample(int modeIdx, Entry& out) {
			std::lock_guard<std::mutex> g(mutex);
			auto& es = entries[modeIdx];
			if (es.empty() || filledIteration[modeIdx] + maxAgeIters < currentIteration)
				return false;
			out = es[RocketSim::Math::RandInt(0, (int)es.size())];
			return true;
		}
	};

	// Resets a practice arena into a perturbed copy of a banked frontier state; falls
	// back to the wrapped setter when the pool has nothing fresh for this arena's mode
	// (or by the 1-useFrac coin - practice arenas keep a share of the normal reset mix
	// so their episode distribution never becomes pure-drill).
	class FrontierDrillState : public StateSetter {
	public:
		std::shared_ptr<FrontierPool> pool;
		StateSetter* fallback; // owned
		float useFrac;
		float posNoise, velNoise; // Gaussian sigmas (validated: 250 / 250)

		FrontierDrillState(std::shared_ptr<FrontierPool> pool, StateSetter* fallback,
			float useFrac = 0.35f, float posNoise = 250, float velNoise = 250)
			: pool(std::move(pool)), fallback(fallback),
			  useFrac(useFrac), posNoise(posNoise), velNoise(velNoise) {}

		~FrontierDrillState() { delete fallback; }

		virtual void ResetArena(Arena* arena) {
			using RocketSim::Math::RandFloat;

			int ppt = (int)arena->_cars.size() / 2;
			FrontierPool::Entry e;
			if (RandFloat() > useFrac || !pool || !pool->Sample(ppt - 1, e)
				|| (int)e.blueCars.size() != ppt || (int)e.orangeCars.size() != ppt) {
				fallback->ResetArena(arena);
				return;
			}

			arena->ResetToRandomKickoff(); // pads etc.

			auto fnNoise = [&](float sigma) {
				// Box-Muller; RocketSim::Math has no gaussian
				float u1 = RS_MAX(RandFloat(0, 1), 1e-7f), u2 = RandFloat(0, 1);
				return sigma * sqrtf(-2.f * logf(u1)) * cosf(2.f * (float)M_PI * u2);
			};

			BallState bs = e.ball;
			bs.pos += Vec(fnNoise(posNoise), fnNoise(posNoise), fnNoise(posNoise * 0.3f));
			bs.pos.z = RS_MAX(bs.pos.z, 94.f);
			bs.pos.x = RS_CLAMP(bs.pos.x, -3990.f, 3990.f);
			bs.pos.y = RS_CLAMP(bs.pos.y, -5010.f, 5010.f);
			bs.vel += Vec(fnNoise(velNoise), fnNoise(velNoise), fnNoise(velNoise));
			arena->ball->SetState(bs);

			int blueIdx = 0, orangeIdx = 0;
			for (Car* car : arena->_cars) {
				const FrontierPool::CarSpawn& c = (car->team == Team::BLUE)
					? e.blueCars[blueIdx++] : e.orangeCars[orangeIdx++];

				Vec pos = c.pos + Vec(fnNoise(posNoise), fnNoise(posNoise), fnNoise(posNoise * 0.2f));
				pos.z = RS_MAX(pos.z, 17.f);
				pos.x = RS_CLAMP(pos.x, -3990.f, 3990.f);
				pos.y = RS_CLAMP(pos.y, -5010.f, 5010.f);

				// Rotation from forward/up (obs-visible); re-orthogonalize the up hint
				Vec f = c.forward.Normalized();
				Vec u = (c.up - f * c.up.Dot(f)).Normalized();
				Vec r = u.Cross(f).Normalized();

				CarState cs = {};
				cs.pos = pos;
				cs.vel = c.vel + Vec(fnNoise(velNoise), fnNoise(velNoise), fnNoise(velNoise));
				cs.angVel = c.angVel;
				cs.rotMat = RotMat(f, r, u);
				cs.boost = RS_CLAMP(c.boost, 0.f, 100.f);
				car->SetState(cs);
			}
		}
	};
}
