#pragma once
#include <RLGymCPP/Gamestates/GameState.h>
#include <RLGymCPP/ActionParsers/DefaultAction.h>

namespace GGL {
	// Scripted kickoff driver (2026-08-13, user-directed). Mirror self-play settled into a
	// delay-kickoff equilibrium: neither copy ever goes for the ball (vs an opponent that
	// commits, the policy contests fine - user-verified vs Nexto). Two costs: contested
	// kickoffs vanish from the training data, and the boot sanity probe (which plays the
	// mirror) reads the deadlock as corruption - two golden-archive rollbacks (2026-08-12,
	// 2026-08-13) quarantined billions of steps of healthy checkpoints that way.
	//
	// This is the minimal committed opponent: full throttle + boost straight at the ball,
	// steering by the car-local ball bearing, ground actions only. Deliberately dumb - it
	// exists to make the 50/50 exist, not to win it. It emits ACTION-TABLE INDICES (like
	// the Nexto adapter), so the env applies it through the normal parser and prevAction /
	// obs consistency is preserved for free. No flips: a boosting straight drive reaches
	// the ball in ~2.5s, which is committed enough to force a contest.
	//
	// Scripted rows are training-poison (the recorded logprob would belong to an action the
	// car did not take), so the Learner suppresses recording for the scripted car until the
	// window ends; the boot probe just counts only non-scripted touches. If the parser is
	// not a DefaultAction, `valid` stays false and everything degrades to no-op.
	struct KickoffScript {
		// [steer bucket: 0=-1, 1=0, 2=+1][boost 0/1] -> action index
		int idx[3][2] = { {-1,-1},{-1,-1},{-1,-1} };
		bool valid = false;

		KickoffScript() {}

		explicit KickoffScript(RLGC::ActionParser* parser) {
			auto* def = dynamic_cast<RLGC::DefaultAction*>(parser);
			if (!def)
				return;
			// Ground rows are uniquely pitch=roll=jump=0 (the table's dedup filter removes
			// the aerial all-zero combos), so this match cannot alias an air action.
			for (int i = 0; i < (int)def->actions.size(); i++) {
				auto& a = def->actions[i];
				if (a.throttle == 1 && a.jump == 0 && a.pitch == 0 && a.roll == 0 && a.handbrake == 0)
					idx[(int)a.steer + 1][(int)a.boost] = i;
			}
			valid = true;
			for (auto& row : idx)
				for (int v : row)
					valid &= (v >= 0);
		}

		// Same detection as KickoffRaceReward: only a reset can leave the ball at rest at
		// field center, so mid-play states can never arm a window.
		static bool IsKickoffSpawn(const RLGC::GameState& gs) {
			return fabsf(gs.ball.pos.x) < 200 && fabsf(gs.ball.pos.y) < 200
				&& gs.ball.vel.Length() < 50;
		}

		// Action index for this step, or -1 for "no override" (invalid table, airborne car,
		// or degenerate geometry). Airborne handoff to the policy is deliberate: the air
		// half of the table is a different action set and a bumped script car should
		// recover like a player, not hold a ground action.
		int ChooseAction(const RLGC::Player& p, const RLGC::GameState& gs) const {
			if (!valid || !p.isOnGround)
				return -1;
			auto to = gs.ball.pos - p.pos;
			to.z = 0;
			float dist = to.Length();
			if (dist < 1e-3f)
				return -1;
			to = to * (1.f / dist);
			float fwd = p.rotMat.forward.Dot(to);
			float right = p.rotMat.right.Dot(to);
			int steerBucket = (right > 0.12f) ? 2 : (right < -0.12f) ? 0 : 1;
			// Boost only while roughly facing the ball, so a spun-around car turns in
			// place instead of boosting away
			int boost = (fwd > 0.6f) ? 1 : 0;
			return idx[steerBucket][boost];
		}
	};
}
