#include <GigaLearnCPP/Learner.h>

#include <RLGymCPP/Rewards/CommonRewards.h>
#include <RLGymCPP/Rewards/ZeroSumReward.h>
#include <RLGymCPP/TerminalConditions/NoTouchCondition.h>
#include <RLGymCPP/TerminalConditions/GoalScoreCondition.h>
#include <RLGymCPP/ObsBuilders/DefaultObs.h>
#include <RLGymCPP/ObsBuilders/AdvancedObs.h>
#include <RLGymCPP/StateSetters/KickoffState.h>
#include <RLGymCPP/StateSetters/RandomState.h>
#include <RLGymCPP/ActionParsers/DefaultAction.h>

using namespace GGL; // GigaLearn
using namespace RLGC; // RLGymCPP

// Create the RLGymCPP environment for each of our games
EnvCreateResult EnvCreateFunc(int index) {

    std::vector<WeightedReward> rewards = {
        // Objective anchor
        { new GoalReward(-0.85f), 150 },
        { new TimePenalty(-0.1f), 1.0f },

        // Broad pressure / directionality, kept low for GCRL compatibility
        { new ZeroSumReward(new VelocityBallToGoalReward(), 1), 0.1f },

        // General physical readiness, kept tiny to avoid energy farming
        { new EnergyReward(), 0.02f },

        // Light mechanics / resource priors
        { new StrongTouchReward(20, 100), 0.5f },
        { new PickupBoostReward(), 1.0f },
        { new SaveBoostReward(), 0.1f },

        // Physical play, light
        { new BumpReward(), 0.25f },
        { new BumpedPenalty(), 0.25f },
        { new DemoReward(), 1.0f },
        { new DemoedPenalty(), 1.0f },

        // Hard-to-discover mechanics/control, kept very light to avoid farming
        { new AirTouchReward(500), 1.0f },
        { new PossessionReward(), 0.05f },
        { new FlipResetFollowupReward(500), 1.0f }
    };




	std::vector<TerminalCondition*> terminalConditions = {
		new NoTouchCondition(10),
		new GoalScoreCondition()
	};

	// Make the arena
	int playersPerTeam = 1;
	auto arena = Arena::Create(GameMode::SOCCAR);
	for (int i = 0; i < playersPerTeam; i++) {
		arena->AddCar(Team::BLUE);
		arena->AddCar(Team::ORANGE);
	}

	EnvCreateResult result = {};
	result.actionParser = new DefaultAction();
	result.obsBuilder = new AdvancedObs();
	result.stateSetter = new KickoffState();
	result.terminalConditions = terminalConditions;
	result.rewards = rewards;

	result.arena = arena;

	return result;
}

void StepCallback(Learner* learner, const std::vector<GameState>& states, Report& report) {
	// To prevent expensive metrics from eating at performance, we will only run them on 1/4th of steps
	// This doesn't really matter unless you have expensive metrics (which this example doesn't)
	bool doExpensiveMetrics = (rand() % 4) == 0;

	// Add our metrics
	for (auto& state : states) {
		if (doExpensiveMetrics) {
			for (auto& player : state.players) {
				report.AddAvg("Player/In Air Ratio", !player.isOnGround);
				report.AddAvg("Player/Ball Touch Ratio", player.ballTouchedStep);
				report.AddAvg("Player/Demoed Ratio", player.isDemoed);

				report.AddAvg("Player/Speed", player.vel.Length());
				Vec dirToBall = (state.ball.pos - player.pos).Normalized();
				report.AddAvg("Player/Speed Towards Ball", RS_MAX(0, player.vel.Dot(dirToBall)));

				report.AddAvg("Player/Boost", player.boost);

				if (player.ballTouchedStep)
					report.AddAvg("Player/Touch Height", state.ball.pos.z);
			}
		}

		if (state.goalScored)
			report.AddAvg("Game/Goal Speed", state.ball.vel.Length());
	}
}

int main(int argc, char* argv[]) {
	// Initialize RocketSim with collision meshes
	// Change this path to point to your meshes!
	RocketSim::Init("collision_meshes");

	// Make configuration for the learner
	LearnerConfig cfg = {};
	// Let GigaLearnCPP select the best available device automatically.
	cfg.deviceType = LearnerDeviceType::AUTO;


	cfg.tickSkip = 4;
	cfg.actionDelay = cfg.tickSkip - 2;

	// Play around with this to see what the optimal is for your machine, more games will consume more RAM
	cfg.numGames = 512;

	// Leave this empty to use a random seed each run
	// The random seed can have a strong effect on the outcome of a run
	cfg.randomSeed = 67;

	int tsPerItr = 150'000;
	cfg.ppo.tsPerItr = tsPerItr;
	cfg.ppo.batchSize = tsPerItr;
	cfg.ppo.miniBatchSize = 150'000; // Lower this if too much VRAM is being allocated
	cfg.ppo.overbatching = true;

	// Using 2 epochs seems pretty optimal when comparing time training to skill
	// Perhaps 1 or 3 is better for you, test and find out!
	cfg.ppo.epochs = 2;

	// This scales differently than "ent_coef" in other frameworks
	// This is the scale for normalized entropy, which means you won't have to change it if you add more actions
	cfg.ppo.entropyScale = 0.035f;

	// Rate of reward decay
	// Starting low tends to work out
	cfg.ppo.gaeGamma = 0.99;

	// Good learning rate to start
	cfg.ppo.policyLR = 1.5e-4;
	cfg.ppo.criticLR = 1.5e-4;
	cfg.ppo.gcrlLR = 1.5e-4;

	// Three quasimetric critics learn positioning/anticipation contrastively (InfoNCE)
	// over hindsight-relabeled future ball goals, and blend their advantage into the
	// policy gradient on top of the reward-driven GAE advantage. The dense rewards above
	// teach mechanics; GCRL teaches where to be.
	cfg.ppo.useGCRL = true;
	cfg.ppo.gcrlAdvScale = 1.0f;    // GCRL advantage weight (1.0 == equal to the reward advantage)
	cfg.ppo.gcrlAntiScale = 0.85f;   // pessimistic "anti" critic weight in the GCRL advantage
	cfg.ppo.gcrlCarScale = 0.5f;     // car-positioning critic weight in the GCRL advantage
	cfg.ppo.gcrlHorizon = 128;       // max HER goal offset in steps (upper bound; ~4.3s at tickSkip 4)
	cfg.ppo.gcrlMinHorizon = 32;     // min HER goal offset in steps (lower bound; ~1.05s at tickSkip 4)
	cfg.ppo.gcrlUseVariableHER = true; // sample goal offset uniformly in [minH, H]; else fixed H
	cfg.ppo.gcrlTau = 0.02f;         // embedding temperature (lower = sharper contrast)
	cfg.ppo.gcrlReprDim = 2048;       // phi/psi embedding dimension (the metric-space capacity)
	cfg.ppo.gcrlInfoNCECoef = 0.75f;  // weight of the InfoNCE loss in the combined objective
	cfg.ppo.gcrlInfoNCEPenalty = 0.01f; // logsumexp penalty inside InfoNCE
	cfg.ppo.gcrlVarReg = 0.3f;       // embedding variance regularization (anti-collapse)
	cfg.ppo.gcrlInfoSubSample = 768; // contrastive sub-batch size

	cfg.ppo.sharedHead.layerSizes = {};
	cfg.ppo.policy.layerSizes = { 1024, 512, 512, 256 };
	cfg.ppo.critic.layerSizes = { 1024, 768, 768, 512, 512, 256 };
	// GCRL phi/psi hidden layers (output is always gcrlReprDim). These sit on top of the
	// shared head, so they can be much smaller than policy/critic.
	cfg.ppo.gcrlCritic.layerSizes = { 1024, 768, 768, 512, 512, 256 };

	auto optim = ModelOptimType::MUON;
	cfg.ppo.policy.optimType = optim;
	cfg.ppo.critic.optimType = optim;
	cfg.ppo.sharedHead.optimType = optim;
	cfg.ppo.gcrlCritic.optimType = optim;

	auto activation = ModelActivationType::LEAKY_RELU;
	cfg.ppo.policy.activationType = activation;
	cfg.ppo.critic.activationType = activation;
	cfg.ppo.sharedHead.activationType = activation;
	cfg.ppo.gcrlCritic.activationType = activation;

	bool addLayerNorm = true;
	cfg.ppo.policy.addLayerNorm = addLayerNorm;
	cfg.ppo.critic.addLayerNorm = addLayerNorm;
	cfg.ppo.sharedHead.addLayerNorm = addLayerNorm;
	// Optional: the embedding is L2-normalized at scoring time, so trunk LayerNorm is less
	// important here than for policy/critic. Off by default; flip to addLayerNorm to match.
	cfg.ppo.gcrlCritic.addLayerNorm = false;

	cfg.sendMetrics = true; // Send metrics
	cfg.renderMode = false; // Don't render

	// Make the learner with the environment creation function and the config we just made
	Learner* learner = new Learner(EnvCreateFunc, cfg, StepCallback);

	// Start learning!
	learner->Start();

	return EXIT_SUCCESS;
}
