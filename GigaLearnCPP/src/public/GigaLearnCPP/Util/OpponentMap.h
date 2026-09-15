#pragma once
// OPPONENT MAP (research/reports/WORLD_MODEL.md): the opponent-conditioned world model.
//
// The transition an on-policy self-play agent faces is physics composed with the opponent:
//   P(s'|s,a) = sum_{a_o} pi_opp(a_o|s) f(s,a,a_o).  f is RocketSim; only pi_opp shifts (ring
// versions, league members). This module models the OPPONENT: q(dy | s, z), where dy is the
// opponent car's next-step displacement (pos, vel; 6 dims, read from the obs's opponent slot)
// and z is a latent inferred by a GRU over the episode's (obs, dy) prefix - the identifier.
// Nothing about physics is learned; nothing about the opponent is allowed anywhere else.
//
//   staleness : TELEMETRY ONLY. NLL of q above its own predicted entropy is a calibration
//               diagnostic, not epistemic uncertainty (E_p[-log q] - H(q) = KL(p||q) + H(p) - H(q);
//               a variance head absorbs mean error, so the average excess -> 0 even when the
//               mean is wrong). Disagreement across INDEPENDENT members is the closer quantity;
//               neither drives exploration until measured in-game.
//   outcome   : R(s, intent, ctx) - the temporally abstract map the planner searches: the
//               return-to-go of "execute this intent for one interval, then continue" from s,
//               trained on INTENT-BOUNDARY rows only (GAE targets). ctx = inferred z ++ an
//               embedding of the served opponent's identity (known in self-play; the fleet
//               faces one opponent per iteration, so a previous-episode z alone would often
//               name the opponent just replaced). Ablation: ctx zeroed.
//   planning  : at each intent boundary a player's intent is argmax_intent R(s, intent, z)
//               with prob planFrac, uniform otherwise (uniform keeps every intent exercised,
//               which the toy showed is what lets conducts coexist and persist).
//
// Toy results (WORLD_MODEL.md T1-T4e): identify PASS, staleness PASS, intent-level outcome
// planning PASS (tick-level imagination FAILS as a planner), ring-regime consolidation PASS.
//
// Threading: Learn() runs on the learn thread; Plan() runs on the collect worker against a
// frozen copy of the outcome head and a frozen per-player z table, both swapped in at the
// barrier (SyncCollectSnapshot). Never call Learn and Plan concurrently on the same copy.

#include "../Framework.h"
#include "../LearnerConfig.h"
#include "../Distributed/Session.h"
#include <torch/torch.h>
#include <unordered_map>
#include <vector>
#include <string>
#include <filesystem>

namespace GGL {

	struct OppMember;   // GRU encoder + Gaussian head (one ensemble member)
	struct OppOutcome;  // R(s, intent, z)

	struct OppMapLearnResult {
		int segments = 0, rows = 0;
		float nll = 0, entropy = 0, excess = 0, excessP90 = 0, disagree = 0, outcomeLoss = 0, zNorm = 0;
		torch::Tensor stale;   // [N] CPU float: prequential across-member disagreement of the opponent-motion
		                       // mean, normalised by the predicted variance (0 on rows not scored). The
		                       // STALENESS potential (WORLD_MODEL.md): where the members disagree about what
		                       // this opponent does next, the map is undrawn or out of date.
	};

	class OpponentMap {
	public:
		OpponentMap(const OppMapConfig& cfg, int obsSize, int intentDim, int numActions, torch::Device device);

		// Learn-thread: build segments from contiguous per-player rows, score prequentially,
		// train one pass, and refresh the pending per-player z table. states [N, obs] (CPU),
		// targetVals [N] (CPU float), intentIds [N] (CPU long, may be undefined if intentDim 0),
		// playerIds [N] int32, terminals [N] int8.
		// actions [N] long (own action; the motion head conditions on it so bumps/collisions are
		// not read as opponent change), oppIds [N] int32 (served-opponent identity per row, known
		// in self-play; the outcome head conditions on its embedding), boundary [N] uint8 (intent
		// interval starts; the outcome target is defined only there: "execute this intent for one
		// interval, then continue").
		OppMapLearnResult Learn(torch::Tensor states, torch::Tensor targetVals, torch::Tensor intentIds,
			torch::Tensor actions, const std::vector<int32_t>& playerIds, const std::vector<int8_t>& terminals,
			const std::vector<int32_t>& oppIds, const std::vector<uint8_t>& boundary);

		// Barrier zone: publish the pending z table and a frozen copy of the outcome head to
		// the collect side.
		void SyncCollectSnapshot();

		// Collect worker: planned intents for the given players from their CURRENT obs rows.
		// obsRows [M, obs] float (CPU, the same values the policy sees); returns [M] intents.
		// Returns an empty vector if planning is not ready (warmup / no intents).
		std::vector<int> Plan(const float* obsRows, const std::vector<int>& players, int M, int oppId);
		bool PlanReady() const { return planReady; }

		void Save(const std::filesystem::path& folder) const;
		void Load(const std::filesystem::path& folder);

		// distributed: all-reduce gradients (called inside Learn if session set)
		void SetDist(Dist::Session* s) { dist = s; }
		void BroadcastParameters();

		int64_t updates = 0;
		OppMapConfig cfg;

	private:
		int obsSize, intentDim, numActions;
		torch::Device device;
		Dist::Session* dist = nullptr;
		std::vector<std::shared_ptr<OppMember>> members;
		std::vector<std::shared_ptr<torch::optim::Adam>> memberOpt;
		std::shared_ptr<OppOutcome> outcome;           // training copy
		std::shared_ptr<OppOutcome> planHead;          // frozen copy for the collect worker
		std::shared_ptr<torch::optim::Adam> outcomeOpt;
		torch::Tensor tgtStd;                          // [6] running std of dy (device)
		bool statsInit = false;
		// per-player z: pending (written by Learn) and live (read by Plan), swapped at barrier
		std::unordered_map<int, std::vector<float>> zPending, zLive;
		bool planReady = false;
		int numPlayersSeen = 0;

		torch::Tensor OppFeat(torch::Tensor states) const;   // [.., 6] opponent pos+vel from the present slot
	};
}
