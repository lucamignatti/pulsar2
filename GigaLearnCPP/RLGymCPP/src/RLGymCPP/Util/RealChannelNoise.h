#pragma once
#include "../ObsBuilders/ObsBuilder.h"
#include "../ActionParsers/ActionParser.h"

// Real-channel observation noise (2026-08-26).
//
// The deployed RLBot client cannot observe the game's engine-internal ground flag:
// the packet masks it around jumps, and the sim-mirror that reconstructs it desyncs
// during exactly the jump-mash windows that matter. Measured on real captures
// (debug.3914029, 49,866 decisions, unambiguous-state subset): the isOnGround the
// policy actually consumed was wrong 4.9% of all decisions and 9.0% of decisions
// within 12 ticks of a jump press. Opponents are worse: their flags come from the
// raw masked packet with no mirror at all. Training on clean flags therefore
// produces a policy whose knife-edge press habits misfire only in the real game
// ("accidental flips") - sim physics matched real to a few uu everywhere while the
// OBS CHANNEL did not (research/reports/WAVEDASH_GATE.md).
//
// These wrappers corrupt isOnGround (the one measured-wrong field; hasJumped /
// hasFlipped / demolished arrive unmasked in the packet) on the player copies fed
// to the obs builder AND the action mask, with matched statistics:
//   - transition-adjacent (recent jump / flip / takeoff / landing): P_NEAR
//   - otherwise: P_FAR
// The draw is a deterministic hash of (tick, carId), so the obs and the mask see
// the SAME corrupted state for a given decision, with zero shared mutable state.
// Flipping isOnGround also flips HasFlipOrJump()'s ground term and the parser's
// ground/air table selection - exactly the real failure surface.
namespace RLGC {

	struct RealChannelNoiseCfg {
		float pNear = 0.09f; // measured: wrong-flag rate within 12 ticks of a press
		float pFar = 0.02f;  // measured: steady-state wrong-flag rate
		uint64_t seed = 0x9E3779B97F4A7C15ull;
	};

	namespace detail {
		inline uint64_t SplitMix(uint64_t x) {
			x += 0x9E3779B97F4A7C15ull;
			x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
			x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
			return x ^ (x >> 31);
		}

		inline bool TransitionAdjacent(const Player& p) {
			return p.isJumping || p.isFlipping
				|| (p.hasJumped && p.airTimeSinceJump < 0.12f)
				|| p.airTime < 0.12f;
		}

		inline void CorruptPlayer(Player& p, uint64_t tick, const RealChannelNoiseCfg& cfg) {
			const float pr = TransitionAdjacent(p) ? cfg.pNear : cfg.pFar;
			const uint64_t h = SplitMix(cfg.seed ^ (tick * 0x100000001B3ull) ^ p.carId);
			const float u = (float)(h >> 40) / (float)(1ull << 24);
			if (u < pr)
				p.isOnGround = !p.isOnGround;
		}

		inline GameState CorruptState(const GameState& state, const RealChannelNoiseCfg& cfg) {
			GameState gs = state;
			for (auto& pl : gs.players)
				CorruptPlayer(pl, state.lastTickCount, cfg);
			return gs;
		}
	}

	class NoisyChannelObs : public ObsBuilder {
	public:
		ObsBuilder* inner;
		RealChannelNoiseCfg cfg;
		NoisyChannelObs(ObsBuilder* inner, RealChannelNoiseCfg cfg = {})
			: inner(inner), cfg(cfg) {}

		virtual void Reset(const GameState& initialState) override { inner->Reset(initialState); }

		virtual FList BuildObs(const Player& player, const GameState& state) override {
			GameState gs = detail::CorruptState(state, cfg);
			for (auto& pl : gs.players)
				if (pl.carId == player.carId)
					return inner->BuildObs(pl, gs);
			return inner->BuildObs(player, gs); // unreachable in practice
		}
	};

	class NoisyChannelParser : public ActionParser {
	public:
		ActionParser* inner;
		RealChannelNoiseCfg cfg;
		NoisyChannelParser(ActionParser* inner, RealChannelNoiseCfg cfg = {})
			: inner(inner), cfg(cfg) {}

		// Actions execute on TRUE state (the game does not care what we believed);
		// only the DECISION inputs (obs + mask) see the corrupted channel.
		virtual Action ParseAction(int actionIdx, const Player& player, const GameState& state) override {
			return inner->ParseAction(actionIdx, player, state);
		}
		virtual int GetActionAmount() override { return inner->GetActionAmount(); }

		virtual std::vector<uint8_t> GetActionMask(const Player& player, const GameState& state) override {
			Player pl = player;
			detail::CorruptPlayer(pl, state.lastTickCount, cfg);
			return inner->GetActionMask(pl, state);
		}
	};
}
