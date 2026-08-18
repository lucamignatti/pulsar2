#pragma once
#include "RewardWrapper.h"
#include <atomic>

namespace RLGC {
	// Multiplies a child reward by an externally-published, time-varying scale, so a
	// SCAFFOLD weight can be annealed on a schedule instead of by hand-editing the
	// weight and restarting (CLAUDE.md: "either implement it or delete the promise" -
	// six scaffold weights named a `mechanic_census` trigger that was never built).
	//
	// THE LATCH IS THE WHOLE POINT. The scale is read ONCE per episode, in Reset(),
	// and held constant for that episode's whole trajectory. A PBRS term wrapped in a
	// *continuously* varying scale is no longer a potential: r = gamma*k'*Phi(s') -
	// k*Phi(s) only telescopes when k' == k, so a scale that moved mid-episode would
	// leak a non-telescoping residual into the return - exactly the "never gate a
	// potential" failure this file's siblings warn about, in a slower disguise. Latched
	// at Reset, k is an episode constant, so k*Phi is itself an exact potential and every
	// PBRS guarantee survives at any point on the schedule. The published scale is
	// therefore free to change at any time on the learner side.
	//
	// PLACEMENT: put this INSIDE the ZeroSumReward, not outside -
	//   ZeroSumReward(new ScheduledScaleReward(child, &scale), TEAM_SPIRIT)   // correct
	//   ScheduledScaleReward(new ZeroSumReward(child, TEAM_SPIRIT), &scale)   // WRONG
	// EnvSet caches `dynamic_cast<ZeroSumReward*>(weighted.reward)` once at startup to
	// drive per-term reward logging; wrapping the ZeroSum hides it from that cast and
	// silently drops the term's wandb panel. Scaling inside is algebraically identical
	// (ZeroSum is linear in the child) and keeps the panel. It does mean the logged
	// `_lastRewards` are post-scale, which is what you want: the panel then shows the
	// term's LIVE contribution rather than its unannealed shape.
	//
	// Thread model: the atomic is written by the learner (main thread, barrier zone) and
	// read by env threads in Reset(). Relaxed ordering is sufficient - a torn or
	// one-episode-stale read of a slowly-ramping scale is harmless, and there is no other
	// state whose visibility this needs to order.
	class ScheduledScaleReward : public RewardWrapper {
	public:
		const std::atomic<float>* scaleSrc;
		float latched;

		// `scaleSrc` must outlive this reward (in practice a file-scope static in the
		// config TU). Not owned.
		ScheduledScaleReward(Reward* child, const std::atomic<float>* scaleSrc)
			: RewardWrapper(child), scaleSrc(scaleSrc),
			latched(scaleSrc->load(std::memory_order_relaxed)) {}

		virtual void Reset(const GameState& initialState) override {
			latched = scaleSrc->load(std::memory_order_relaxed);
			child->Reset(initialState);
		}

		virtual void PreStep(const GameState& state) override {
			child->PreStep(state);
		}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			return latched * child->GetReward(player, state, isFinal);
		}

		// MUST forward through the child's GetAllRewards, not fall back to the base-class
		// per-player loop: several terms here are team-aware (BallProximityPotentialReward
		// is team-closest) and implement their real behaviour in GetAllRewards. The default
		// RewardWrapper does NOT forward it, so a child like that would be silently
		// downgraded to its per-player path by any wrapper that doesn't do this.
		virtual std::vector<float> GetAllRewards(const GameState& state, bool isFinal) override {
			std::vector<float> out = child->GetAllRewards(state, isFinal);
			for (float& v : out)
				v *= latched;
			return out;
		}

		virtual std::string GetName() override {
			return child->GetName();
		}
	};
}
