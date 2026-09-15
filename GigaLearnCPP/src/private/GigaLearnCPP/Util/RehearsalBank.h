#pragma once
#include <torch/torch.h>
#include <filesystem>
#include <vector>
#include <string>
#include <unordered_map>

namespace GGL {
	struct Report;

	// ===================== REHEARSAL BANK (witnessed-success memory) =====================
	// Ported 2026-09-11 from branch main2.1-private (RehearsalBank of the gco-2.1 run) into
	// the intent-class tree, WITHOUT the league scout role: here the intent class supplies
	// the attempts, the bank supplies the persistence, and the main consolidates by
	// imitation (research/reports/NATIVE_INTENT_PROTOCOL.md, "v2: bank").
	//
	// Why a memory: the production self-imitation term weights rows of the CURRENT buffer
	// only, so a success that occurs once in thousands of episodes gets one iteration of
	// gradient and is forgotten (measured: 24/24 production-faithful toy forks at 0%). A
	// bank of <= capacity whole episode tails, kept across iterations and rehearsed by a
	// weighted imitation loss every update, crossed the toy's acquisition wall on four
	// layouts (research/testbeds/causal_features/REHEARSAL_RESULT_20260906.md).
	//
	// Admission (per row-contiguous episode segment of the main buffer):
	//   * SUCCESS by outcome: the segment's best extrinsic target beats the targetQ
	//     quantile of the buffer's targets (and the absolute minTarget floor) - under
	//     goal/concede reward, "it ended in a goal";
	//   * RARITY: some row whose action the acting policy assigned probability below the
	//     pAdmit quantile of the buffer's behaviour probabilities (routine successes are
	//     already practised by SIL);
	//   * optionally (requireConv) a SIL-converting row. OFF by default here, per the
	//     user's direction (2026-09-11): consolidation keys on success, not on headroom -
	//     the gco-2.1 bank admitted least-bad concedes through the headroom-relative gate.
	//   * not a duplicate.
	// Ranking / eviction at capacity: COMPETITIVE by the rarest single decision (see rankBySum);
	// expiry after `life` iterations. Imitation weight clamp(target - V, 0, cap) recomputed
	// each iteration with the live critic, so an entry retires once the main prices it.
	//
	// Intent class: rows carry the (z, clock) features the policy saw, so the imitation
	// forward is exactly conditioned as the acting forward was.
	// Threading: touched only at learn-prep (main thread, barrier zone).
	struct RehearsalBankConfig {
		bool enabled = false;
		int capacity = 8;          // episodes
		int maxRowsPerEntry = 512; // tail kept, ending at the segment's last row
		float pAdmit = 0.05f;      // rarity quantile of the buffer's behaviour log-probs
		int64_t life = 2000;       // iterations before an entry expires
		float targetQ = 0.98f;     // success bar: best target beats this quantile of the buffer's targets
		float minTarget = 0.0f;    // absolute floor on that best target
		bool requireConv = false;  // additionally require a SIL-converting row (headroom gate)
		float coeff = 0.f;         // imitation coefficient; 0 = the SIL coefficient
		float wFloor = 0.f;        // v2.7 (GGL_BANK_W_FLOOR, in units of the SIL cap): minimum imitation weight per banked
		                           // row. On a transferred critic (Pulsar fork) target - V is ~.05, so the bank was
		                           // nearly inert even when it held the right chains; a floor rehearses them anyway.
		bool roadOnly = false;     // rehearse every row but each entry's final decision
		bool requireResetEdge = false; // v2.4 (GGL_BANK_REQUIRE_RESET=1): admit only chains containing a flip-reset
		                               // edge (hasFlip rising while airborne). Three generic rarity rankings (max
		                               // surprisal = flukes, summed = flail, state leverage = ceiling outliers)
		                               // never admitted a reset chain in 3.5B steps; this isolates the
		                               // CONSOLIDATOR (does rehearsing reset->goal chains raise the rate?) from
		                               // the SELECTOR (how to find such chains without naming the mechanic).
		bool carryTails = false;   // v2.3 (GGL_BANK_CARRY=1): carry each player's un-terminated tail across the iteration boundary
		                           // (measured: reset->goal latency p50 5.1 s, only 9.5% within one 3.2 s cluster
		                           // slice, 78% within 128 rows - without this the bank is blind to ~90% of chains)
		int rankMode = 2;          // ranking score of a segment (competition at capacity):
		                           //  0 = max single-step action surprisal  -> selected one-off FLUKES
		                           //      (gco-intent-bank 611.6B decode: 8 goal tails whose only
		                           //      distinction was one p~1e-5 action; 20k qualified successes refused)
		                           //  1 = summed action surprisal            -> selected FLAIL (gco-2.1: 8x48
		                           //      rows at ~4 nats/row, air share .85; that policy scored 29 goals/300)
		                           //  2 = STATE NOVELTY: max over the segment of the Mahalanobis leverage of a
		                           //      26-dim physical descriptor (ball pos/vel, car pos/orientation/vel,
		                           //      ball-relative pos/vel, onGround, hasFlip) against a running mean/cov
		                           //      of PAST iterations. "A success reached through a region of state
		                           //      space the policy rarely visits" - the user's class definition; action
		                           //      noise in familiar states cannot game it. Default.
		                           //  3 = HEADROOM (v2.5, generic): max over the chain of H = relu(min(V-dagger1,
		                           //      V-dagger2) - V_real), the composition critic's "what compositions of things I
		                           //      sometimes do could achieve beyond what I usually do". Success-gated, so the
		                           //      gco-2.1 failure (H peaks in conceding states) cannot recur. Names no mechanic.
	};

	class RehearsalBank {
	public:
		struct Entry {
			torch::Tensor states, actions, masks, targets; // CPU; [n,obs] f32, [n] i64, [n,A] u8, [n] f32
			torch::Tensor intentIds, intentFeat;           // CPU; [n] i64, [n,F] f32 (undefined without intents)
			float score = 0.f;      // ranking score (see rankMode)
			float maxSurprisal = 0.f, maxNovelty = 0.f, maxHeadroom = 0.f; // telemetry
			float headroom = 0.f;   // mean H along the path at admission (telemetry only)
			float bestTarget = 0.f; // the success that admitted it (telemetry only)
			int64_t carried = 0;    // rows that came from the carried tail
			int64_t resetEdges = 0; // AdvancedObsPadded hasFlip (77) rising while airborne (onGround 76 == 0) - a flip reset in the tail
			int64_t born = 0;
		};
		RehearsalBankConfig cfg;
		std::vector<Entry> entries;
		int64_t admitted = 0, refusedCommon = 0, refusedDuplicate = 0, refusedNegative = 0, refusedWeak = 0,
			refusedNoConv = 0, refusedNoEvent = 0, evicted = 0, expired = 0, candidates = 0;

		explicit RehearsalBank(RehearsalBankConfig c) : cfg(c) {}

		// All tensors CPU, row-aligned over one buffer. `terminals` nonzero marks the last row
		// of an episode segment. `conv`: SIL-converting rows (bool; may be all false).
		// `mainProb`: pi(a_t|s_t) of the stored action under the acting policy. `headroom`:
		// H per row (zeros if none). `intentIds`/`intentFeat` may be undefined.
		void Admit(const torch::Tensor& states, const torch::Tensor& actions, const torch::Tensor& masks,
			const torch::Tensor& targets, const torch::Tensor& terminals, const torch::Tensor& conv,
			const torch::Tensor& mainProb, const torch::Tensor& headroom,
			const torch::Tensor& intentIds, const torch::Tensor& intentFeat,
			const torch::Tensor& stateNovelty, // [n] f32 per-row leverage (undefined = zeros; needed for rankMode 2)
			const torch::Tensor& playerIds,    // [n] int per-row player index (undefined = no tail carrying)
			int64_t iteration, float pThresh, float targetThresh);

		// Per-player carried tail (CPU): the last <= maxRowsPerEntry rows of the player's final,
		// TRUNCATED segment of the previous iteration, prepended to its first segment now.
		struct Prefix { torch::Tensor states, actions, masks, targets, prob, novelty, headroom, intentIds, intentFeat; };
		std::unordered_map<int32_t, Prefix> prefixes;
		int64_t carriedRows = 0, carriedSegments = 0; // this iteration's telemetry

		void Expire(int64_t iteration);

		// Concatenated rows for imitation (CPU). dropLast: omit each entry's final decision.
		struct Rows { torch::Tensor states, actions, masks, targets, intentIds, intentFeat; int64_t n = 0; };
		Rows GetRows(bool dropLast) const;

		int64_t RowCount() const;
		size_t Bytes() const;
		void Save(const std::filesystem::path& folder) const;
		void Load(const std::filesystem::path& folder);
		void ReportTo(Report& report, const std::string& prefix) const;
	};
}
