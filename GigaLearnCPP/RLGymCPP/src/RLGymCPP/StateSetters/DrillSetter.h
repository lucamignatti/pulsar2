#pragma once
#include "StateSetter.h"
#include "DrillBank.h"
#include "../Math.h"

namespace RLGC {

	// Deliberate-practice drill (Stage 3): restores a snapshot sampled from a shared DrillBank
	// (ball + cars + boost pads, with small position/velocity jitter so the exact same state
	// isn't replayed verbatim) and arms this arena's practice window so the collection loop can
	// tag the following steps with the goal that was committed when the mistake was banked.
	// Falls back to a normal setter when the bank is empty - including the (rare) case where
	// CombinedState selects this setter at weight 0 due to a float-boundary rounding edge.
	class DrillSetter : public StateSetter {
	public:
		DrillBank* bank;
		int arenaIdx;
		StateSetter* fallback;
		float jitterPos, jitterVel;

		DrillSetter(DrillBank* bank, int arenaIdx, StateSetter* fallback, float jitterPos = 100, float jitterVel = 150)
			: bank(bank), arenaIdx(arenaIdx), fallback(fallback), jitterPos(jitterPos), jitterVel(jitterVel) {
		}

		virtual void ResetArena(Arena* arena) override {
			using RocketSim::Math::RandFloat;

			ArenaSnapshot snap;
			DrillBank::PracticeWindow win;
			if (!bank || !bank->TryBeginDrill(arenaIdx, snap, win)) {
				fallback->ResetArena(arena);
				return;
			}

			// Team-count compatibility guard: a mismatched snapshot (e.g. banked under a
			// different team-size config) would otherwise silently drop or duplicate cars
			int snapCount[2] = { 0, 0 };
			for (auto& c : snap.cars)
				snapCount[(int)c.team]++;

			int arenaCount[2] = { 0, 0 };
			for (Car* car : arena->GetCars())
				arenaCount[(int)car->team]++;

			if (snapCount[0] != arenaCount[0] || snapCount[1] != arenaCount[1]) {
				fallback->ResetArena(arena);
				return;
			}

			// Clean slate first (boost pads, ball, cars all reset to a sane default), matching
			// BallNearCarState's own idiom, before we overwrite with the snapshot
			arena->ResetToRandomKickoff();

			auto fnJitteredPos = [&](Vec pos) {
				return pos + Vec(RandFloat(-jitterPos, jitterPos), RandFloat(-jitterPos, jitterPos), 0);
			};
			auto fnJitteredVel = [&](Vec vel) {
				return vel + Vec(
					RandFloat(-jitterVel, jitterVel), RandFloat(-jitterVel, jitterVel), RandFloat(-jitterVel, jitterVel) * 0.25f);
			};

			BallState bs = snap.ball;
			bs.pos = fnJitteredPos(bs.pos);
			bs.vel = fnJitteredVel(bs.vel);
			arena->ball->SetState(bs);

			// Assign snapshot cars to this arena's cars by (team, per-team encounter order).
			// RocketSim's car set has no ordering guarantee stable across arenas, so this is the
			// best available match; cars on the same team are interchangeable for training.
			std::vector<int> teamCarIdx[2];
			for (int i = 0; i < (int)snap.cars.size(); i++)
				teamCarIdx[(int)snap.cars[i].team].push_back(i);

			int cursor[2] = { 0, 0 };
			for (Car* car : arena->GetCars()) {
				int t = (int)car->team;
				if (cursor[t] >= (int)teamCarIdx[t].size())
					continue; // unreachable given the count guard above

				CarState cs = snap.cars[teamCarIdx[t][cursor[t]]].state;
				cursor[t]++;

				cs.pos = fnJitteredPos(cs.pos);
				cs.vel = fnJitteredVel(cs.vel);
				car->SetState(cs);
			}

			// Boost pads are index-aligned: GetBoostPads() order is fixed per-arena, and per the
			// shared boostPadIndexMap (GameState.cpp), index i is the SAME physical pad location
			// across every arena. curLockedCar is a raw pointer into the SOURCE arena and is not
			// portable - null it out (accepted v1 fidelity loss).
			auto& pads = arena->GetBoostPads();
			if (pads.size() == snap.pads.size()) {
				for (int i = 0; i < (int)pads.size(); i++) {
					BoostPadState ps = snap.pads[i];
					ps.curLockedCar = NULL;
					pads[i]->SetState(ps);
				}
			}
		}
	};
}
