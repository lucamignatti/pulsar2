#pragma once
#include "../Util/Models.h"
#include <RLGymCPP/Gamestates/GameState.h>

#include <random>

namespace GGL {

	// ===================== LEAGUE OF LoRA VARIANTS (GGL_LEAGUE) =====================
	// A fixed roster of policy VARIANTS, each a rank-r LoRA adapter riding the LIVE main
	// weights (shared_head + policy). Two roles:
	//   - DIVERSE  (ids [0, numDiverse)): trained on the env reward PLUS a discriminator-
	//     derived diversity term r_div (variational I(z ; s_{t+lag} | s_t): "given where
	//     you started, what you did with it must be recognizably yours").
	//   - EXPLOITER (ids [numDiverse, Total())): trained on the plain zero-sum env reward
	//     against the live main - AlphaStar's main-exploiter dynamic, automatic here
	//     because the adapter base IS the live main (no snapshot, no sync).
	// Every variant also owns a rank-r LoRA on the CRITIC HEAD (its return includes r_div /
	// a different opponent distribution, and one critic must never price two reward
	// structures - the critic-aliasing law). The critic adapters read the value trunk
	// DETACHED: variant value error must not reshape shared perception.
	//
	// The discriminator is a standalone pair of small MLPs on curated canonical
	// DESCRIPTORS (never the 230-dim obs, never prevAction - a discriminator handed the
	// action column learns "variant 3 presses button 47" and calls it a strategy):
	//   discPair(d_t, d_{t+lag}, lagOnehot) -> numDiverse logits
	//   discMarg(d_t)                       -> numDiverse logits
	//   r_div contribution = log q_pair(z | d_t, d_{t+lag}) - log q_marg(z | d_t)
	// summed over the configured lags, credited AT THE LATER ROW (the actions that made
	// the transition sit inside its discounted return), centered/clamped per variant by
	// running EMAs, scaled by divBeta. The marginal subtraction kills the DIAYN camping
	// failure (being in an identifiable state without doing anything identifiable).
	//
	// Threading contract (mirrors the Learner's snapshot discipline): the collect worker
	// only ever touches *Snap copies + the rings/reservoir/EMAs; Learn() only touches the
	// live nets + tensors prepared in the barrier zone; SyncCollectSnapshot() and
	// PrepareLearnData() are barrier-zone-only.
	struct LeagueConfig {
		int numDiverse = 8;
		int numExploiters = 2;
		int rank = 4;             // LoRA rank r
		float arenaFrac = 0.25f;  // share of arenas hosting main-vs-variant play
		float divBeta = 0.1f;     // r_div scale (raw centered log-prob units)
		float adapterLR = 3e-4f; // adapters walk FAST at rank 4 (deltas reach O(hidden) within ~1k steps at 1e-3); keep near the main's policy LR scale
		float discLR = 1e-3f;
		int lagShort = 300;       // decision steps (~2.5s at ts1 120Hz)
		int lagLong = 1080;       // decision steps (~9s at ts1 120Hz)
		int discBatch = 4096;     // tuples sampled per iteration for disc training
		int discWarmupUpdates = 50; // r_div forced 0 until the disc has trained this often
		int epochs = 2;           // league PPO epochs per iteration
		float clipRange = 0.2f;   // PPO clip (mirrors main)
		float entropyScale = 0.f; // set from main config by the Learner
		// Use the MAIN critic (read-only) as the variants' GAE baseline instead of the
		// per-variant critic adapters. A fresh critic's error IS the advantage in the
		// mostly-goal-free 24-step fragments this run collects, and PPO's clip
		// asymmetry grinds that noise into policy diffusion (measured: variants at
		// normalized entropy ~1.0 within ~350 iterations, both roles). Read-only, so
		// no critic ever trains on two reward structures (the aliasing law holds).
		bool useMainCritic = true;
		// Mirror the main's advantage filtering (MAGNITUDE mode): the policy trains
		// only on the top-|A| fraction of rows. This is the third leg of the main's
		// tuned economy (raw advantages + filtering + SIL); without it the league
		// PPO grinds on the TD-residual noise sea of goal-free fragments and the
		// clip asymmetry diffuses variants toward uniform (measured twice).
		float advFilterFrac = 0.5f;
		// SIL, the fourth leg (set from the main's silCoeff): positive-only BC on rows
		// whose advantage residual beats +1 sigma (conversion, not routine luck),
		// weight = clamp(A, 0, 2 sigma). This is the GCO run's load-bearing sparse
		// mechanism - without it a variant below break-even gets ~pure suppression
		// gradient ("avoid what you did", never "do this instead") and death-spirals:
		// measured as the residual gsDiv decay after the first three economy legs.
		float silCoeff = 0.05f;
		// Iterations of league rows to ACCUMULATE before one adapter update. The fleet
		// gives the main 4176 rows/rank/iter for ONE policy; the league gets 594 split
		// across Total() variants = ~59/variant/rank, a 70x smaller batch taking the
		// SAME number of update steps. That is far past the advantage-noise floor this
		// env has (episode-cluster variance ~5x binomial). Accumulating is free in data
		// terms and costs only off-policyness: the adapters are CONSTANT across a
		// window, so the only staleness is base drift, which the clip absorbs.
		int accumEvery = 4;
		// Iterations the windowed goal share spans before it resets. Goals are RARE
		// under GCO - a 1-iteration window almost never contains one, so the first
		// version of this panel read 0/-1 forever and was unreadable. The window must
		// be long enough to hold goals but short enough to still show a slope.
		int winResetEvery = 200;
		int64_t miniBatch = 32768;

		int Total() const { return numDiverse + numExploiters; }
	};

	class PPOLearner; // fwd
	struct Report;    // fwd

	class LeagueModule {
	public:
		static constexpr int DESC_DIM = 28;
		static constexpr int NUM_LAGS = 2;

		LeagueConfig cfg;
		torch::Device device;

		// Rank-r adapters per Linear layer of a model chain. A[l]: [V, r, out],
		// B[l]: [V, r, in]. B zero-init => every variant starts EXACTLY as the main.
		struct AdapterStack {
			std::vector<torch::Tensor> A, B;
		};
		AdapterStack pol, cri;         // live (trained)
		AdapterStack polSnap, criSnap; // collect-side frozen copies (device)

		torch::nn::Sequential discPair{ nullptr }, discMarg{ nullptr };         // live (device)
		torch::nn::Sequential discPairSnap{ nullptr }, discMargSnap{ nullptr }; // collect copies (CPU)

		torch::optim::Adam* adapterOptim = nullptr;
		torch::optim::Adam* discOptim = nullptr;
		int64_t discUpdates = 0;

		// ---- collect-thread state ----
		// Per-player descriptor ring (capacity lagLong+1); episodes spanning iteration
		// boundaries keep their ring, terminals clear it - so lag pairs are exactly
		// "same episode, lag steps apart" even though the trajectory buffer is chopped
		// per iteration (at ts1 a collection window is far shorter than lagLong).
		struct DescRing {
			std::vector<float> buf; // (lagLong+1) * DESC_DIM
			int head = 0, count = 0;
		};
		std::vector<DescRing> rings; // indexed by global player idx (only league slots used)

		// Reservoir of disc training tuples (d0, d1, lagIdx, z), diverse variants only.
		struct Reservoir {
			std::vector<float> d0, d1;   // n * DESC_DIM
			std::vector<int8_t> lag, z;
			int64_t seen = 0;
			int64_t cap = 1 << 16;
			int64_t Size() const { return (int64_t)lag.size(); }
		} reservoir;
		std::mt19937_64 rng{ 0xC0FFEE };

		// Per-variant r_div running stats (EMA mean/var) - collect thread only.
		std::vector<float> rdivMean, rdivVar;
		// Per-variant goal counters (collect thread writes, barrier reads). CUMULATIVE
		// since process start, plus a WINDOW pair zeroed at every barrier harvest: a
		// cumulative share has so much inertia that a recovered variant still reads
		// low for hours (the Nexto-counter lesson - the slope is the signal, so
		// publish the slope directly).
		std::vector<int64_t> goalsFor, goalsAgainst;
		std::vector<int64_t> winGoalsFor, winGoalsAgainst;
		// Iteration-window r_div telemetry (collect thread).
		double rdivSum = 0, rdivSqSum = 0; int64_t rdivCnt = 0;

		// ---- barrier-prepared learn data (Learn() consumes these) ----
		torch::Tensor discD0, discD1, discLag, discZ; // sampled disc batch, on device
		// Collect-side telemetry harvested at the barrier (Learn runs concurrently with
		// the next collection under pipelining, so it must never read the live counters).
		float statRdivMean = 0, statRdivStd = 0;
		int64_t statRdivCnt = 0;
		int64_t statReservoirFill = 0;
		std::vector<int64_t> statGoalsFor, statGoalsAgainst;
		std::vector<int64_t> statWinFor, statWinAgainst;

		// Accumulation buffers (see LeagueConfig::accumEvery). Device tensors held
		// across iterations; ~4MB at the fleet's row counts.
		std::vector<torch::Tensor> pendStates, pendMasks, pendActions, pendLogProbs,
			pendAdv, pendTargets, pendVariant;
		int accumCount = 0;
		int winHarvests = 0;

		LeagueModule(ModelSet& baseModels, LeagueConfig config, torch::Device device, int numPlayers);
		~LeagueModule();

		// Canonical curated descriptor. ORANGE rows are mirrored (x,y negated) so both
		// teams describe themselves attacking +y - the same team-canonical frame as the
		// obs builder (a discriminator on world-frame descriptors would learn "variant k
		// defends the -y net", which is side, not style).
		static void BuildDescriptor(const RLGC::GameState& gs, const RLGC::Player& self,
			const RLGC::Player* opp, float* out /*DESC_DIM*/);

		// Barrier zone only: freeze live adapters + disc for the collect worker.
		void SyncCollectSnapshot();

		// Collect thread: batched action sampling for this step's league rows through the
		// SNAPSHOT adapters on top of `base` (the same frozen base the main rows use).
		// rowVariant: [rows] int64 on obs.device(). Sampling semantics mirror
		// InferActionsFromModels (masked softmax, prob clamp, non-finite sanitize).
		void InferActions(ModelSet& base, torch::Tensor obs, torch::Tensor actionMasks,
			torch::Tensor rowVariant, torch::Tensor* outActions, torch::Tensor* outLogProbs);

		// Collect thread, once per env step AFTER descriptors for the step are built:
		// pushes each row's descriptor into its ring, harvests disc tuples into the
		// reservoir, and returns the per-row r_div (CPU float [n]; zeros for exploiters
		// and during disc warmup). players/variants are parallel arrays; descs is
		// row-major [n * DESC_DIM].
		std::vector<float> StepRdivAndPush(const std::vector<int>& players,
			const std::vector<int>& variants, const std::vector<float>& descs);

		void ResetRing(int playerIdx) {
			rings[playerIdx].head = 0;
			rings[playerIdx].count = 0;
		}

		// Barrier zone only: sample the disc batch for the coming Learn().
		void PrepareLearnData();

		// Per-variant critic read for league GAE (no-grad; live adapters - runs in the
		// barrier/learn phase where the worker never touches live nets).
		torch::Tensor InferValues(PPOLearner* ppo, torch::Tensor obs, torch::Tensor rowVariant);

		// The batched league learn pass: ONE forward for all variants (per-row adapter
		// gather), PPO clip + entropy + per-variant critic MSE + disc CE. Steps ONLY the
		// league optimizers, then zeroes every base-model grad it polluted. Call AFTER
		// ppo->Learn() (so the base optimizer step is clean of league gradients).
		void Learn(PPOLearner* ppo, torch::Tensor states, torch::Tensor actionMasks,
			torch::Tensor actions, torch::Tensor logProbs, torch::Tensor advantages,
			torch::Tensor targetValues, torch::Tensor rowVariant,
			Dist::Session* dist, Report& report);

		void Save(std::filesystem::path folder);
		void Load(std::filesystem::path folder); // missing files => fresh init (warm starts)
		void BroadcastParams(Dist::Session* dist);

		// All live trainable tensors (adapters + disc), for save/bcast/allreduce.
		// The two families step on different cadences, hence the split accessors.
		std::vector<torch::Tensor> LiveParams();
		std::vector<torch::Tensor> AdapterParams();
		std::vector<torch::Tensor> DiscParams();

	private:
		// Walk a model chain's fp32 seq with per-row rank-r deltas; honors residual spans.
		// `li` is the running Linear cursor into stack.A/B across chained models.
		torch::Tensor ForwardLora(Model* m, torch::Tensor x, const AdapterStack& stack,
			torch::Tensor rowVariant, int& li);
		torch::Tensor PolicyLogitsLora(ModelSet& base, torch::Tensor obs,
			const AdapterStack& stack, torch::Tensor rowVariant);
		// Disc forward on CPU snapshots: returns [n] log q(z=variant | inputs).
		void MakeAdapters(const std::vector<Model*>& chain, AdapterStack& live, AdapterStack& snap);
	};
}
