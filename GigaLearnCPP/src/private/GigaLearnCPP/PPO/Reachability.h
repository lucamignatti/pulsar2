#pragma once
#include "../Util/Models.h"

namespace GGL {

	// Learned goal-reachability (InfoNCE + HER): one shared phi(trunkOut, action) state-action
	// encoder + small per-head psi goal encoders.
	//  CAR  head: "can this car reach the ball"        (goals = car-local ball pos+vel)
	//  BALL head: "can the ball reach the opponent net" (goals = canonical ball pos+vel)
	// Used ONLY as an auxiliary representation loss on the shared head, and as the input of the
	// reward gate. Never enters advantages/returns/critic.
	class ReachabilityModule {
	public:
		ReachabilityConfig config;
		torch::Device device;
		int numActions;

		// Owned by the PPOLearner's ModelSet (saved/loaded/stepped with the other models)
		Model* phi;
		Model* psiCar;
		Model* psiBall;

		struct InfoNCEResult {
			torch::Tensor loss; // Undefined if the batch was degenerate (< 2 rows)
			float categoricalAccuracy = 0;
			float rawLoss = 0;
		};

		ReachabilityModule(int trunkOutSize, int numActions, const ReachabilityConfig& config, torch::Device device, ModelSet& outModels);

		// L2-normalized phi embedding for (trunk output, action index) rows
		torch::Tensor EncodeStateAction(torch::Tensor trunkOut, torch::Tensor actions);

		// Bidirectional InfoNCE of sa vs this head's goals, plus the psi variance and logsumexp
		// penalties. Does NOT include the shared phi variance penalty (add StateActionVarPenalty
		// once per step so it isn't double-counted across heads).
		InfoNCEResult ComputeInfoNCELoss(Model* psiHead, torch::Tensor sa, torch::Tensor goals);
		torch::Tensor StateActionVarPenalty(torch::Tensor sa);

		struct GoalQuery {
			Model* psiHead;
			torch::Tensor goal6;
		};

		// Action-marginalized state reachability: rho(s -> goal) = mean over K uniform VALID
		// actions of Score(s, a, goal) = cosine(phi, psi)/tau. Evaluates ALL queries in one
		// pass over obs (the trunk/phi work is goal-independent, so extra queries cost one
		// matmul each). Returns one [n] float32 CPU tensor per query, in query order.
		// sharedHead may be null (raw obs feed phi directly).
		std::vector<torch::Tensor> EvalRho(Model* sharedHead, const std::vector<GoalQuery>& queries, torch::Tensor obs, torch::Tensor actionMasks);
	};
}
