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

	// Refit 2026-08-26 (v2, after side-by-side viz review read the first model as
	// "way overdone"): the real channel's errors are ONE-DIRECTIONAL and BURSTY.
	// Measured (debug.3914029, unambiguous-state subset):
	//   - grounded-shown-as-air: 0 of 7,498 ticks. NEVER inject it (the v1 symmetric
	//     flip did, handing a grounded bot the air table - a failure mode the real
	//     game does not have).
	//   - air-shown-as-grounded: 6.6% of airborne ticks, arriving as BURSTS:
	//     starts 0.54%/tick, median length 2, heavy tail (p75 6, max 116) => mean ~12.
	// Model: per car, hashed 64-tick epochs schedule at most one burst (p such that
	// starts match), length mixture 80% short (mean ~2) / 20% long (mean ~50, cap 60).
	// During a burst an AIRBORNE car reads isOnGround=true; grounded cars are never
	// touched. i.i.d. injection at the same average wrongness produces many times
	// more distinct decision derailments than bursts - that was the "overdone".
	struct RealChannelNoiseCfg {
		float burstStartsPerTick = 0.0054f; // measured burst-start rate
		float pLong = 0.20f;                // share of long desync episodes
		int shortMean = 2, longMean = 50, longCap = 60;
		uint64_t seed = 0x9E3779B97F4A7C15ull;
	};

	namespace detail {
		inline uint64_t SplitMix(uint64_t x) {
			x += 0x9E3779B97F4A7C15ull;
			x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
			x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
			return x ^ (x >> 31);
		}

		inline void CorruptPlayer(Player& p, uint64_t tick, const RealChannelNoiseCfg& cfg) {
			if (p.isOnGround)
				return; // measured: the channel never shows a grounded car as airborne
			constexpr uint64_t EPOCH = 64;
			// A burst may spill from the previous epoch; check this epoch and the last.
			for (int back = 0; back < 2; back++) {
				const uint64_t epoch = tick / EPOCH - back;
				const uint64_t h = SplitMix(cfg.seed ^ (epoch * 0x100000001B3ull) ^ p.carId);
				const float u0 = (float)(h & 0xFFFFFF) / (float)(1 << 24);
				if (u0 >= cfg.burstStartsPerTick * (float)EPOCH)
					continue; // no burst scheduled in that epoch
				const uint64_t start = epoch * EPOCH + ((h >> 24) & (EPOCH - 1));
				const bool isLong = ((float)((h >> 32) & 0xFF) / 256.f) < cfg.pLong;
				// geometric-ish length from hash bits
				uint32_t bits = (uint32_t)(h >> 40);
				int len = 1;
				const int mean = isLong ? cfg.longMean : cfg.shortMean;
				while ((bits & 1) == 0 && len < (isLong ? cfg.longCap : 8)) {
					len += mean / 2 + 1;
					bits >>= 1;
				}
				if (tick >= start && tick < start + (uint64_t)len) {
					p.isOnGround = true;
					return;
				}
			}
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
