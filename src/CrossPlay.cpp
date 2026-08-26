// Cross-play evaluator: two DIFFERENT GGL-family bots in the certified RocketSimV3,
// each with its own obs builder, architecture, tickSkip and actionDelay.
//
// Why this exists (2026-08-25): the wavedash meter is one narrow mechanic at n~40 from
// real matches — far too noisy to answer "did the ts8->ts1 switch degrade the policy?".
// Goal share against a strong FIXED external opponent at large n is the right meter, and
// BonkDaddy V4.5 shipped full source (AdvancedObsV4, 149-dim, ts4/actionDelay2, trunk
// 1536x2 / policy 1536-1024-1024, LeakyReLU), so it can be run inside our own sim.
//
// Protocol: kickoff-only resets, goal-only terminal (+ time cap), sides swapped halfway,
// argmax both sides. Per-car decision cadence: each bot re-decides every own-tickSkip
// ticks and its action is applied after its own actionDelay, so both play at the rate
// they were trained at. Arena steps 1 tick at a time so the two cadences interleave.
//
// GGL_XP_A / GGL_XP_B      checkpoint dirs
// GGL_XP_A_KIND / _B_KIND  "gco" (AdvancedObsPadded 230, ts from GGL_XP_A_TS) or "bonk4"
// GGL_XP_GOALS             stop after this many decided episodes (default 200)
#include <RLGymCPP/ObsBuilders/AdvancedObsPadded.h>
#include <RLGymCPP/ObsBuilders/AdvancedObsV4.h>
#include <RLGymCPP/ActionParsers/DefaultAction.h>
#include <RLGymCPP/Gamestates/GameState.h>
#include <GigaLearnCPP/Util/InferUnit.h>
#include <RLGymCPP/Framework.h>

#include <cstdlib>
#include <cstdio>
#include <string>
#include <memory>
#include <vector>

using namespace RLGC;

static const char* EnvS(const char* n, const char* d) { const char* v = getenv(n); return (v && *v) ? v : d; }
static int EnvI(const char* n, int d) { const char* v = getenv(n); return (v && *v) ? atoi(v) : d; }

struct Fighter {
	std::string kind;
	int tickSkip, actionDelay;
	ObsBuilder* obs;
	ActionParser* parser;
	GGL::InferUnit* unit;
	// live cadence state
	int ticks = -1;
	Action action = {}, controls = {};

	static Fighter Make(const std::string& kind, const std::string& dir, int tsOverride) {
		Fighter f;
		f.kind = kind;
		f.parser = new DefaultAction();
		GGL::PartialModelConfig head{}, pol{};
		int obsSize;
		if (kind == "bonk4") {
			f.obs = new AdvancedObsV4();
			obsSize = 149;
			f.tickSkip = 4; f.actionDelay = 2;
			head.layerSizes = { 1536, 1536 };
			pol.layerSizes = { 1536, 1024, 1024 };
			head.addLayerNorm = pol.addLayerNorm = true;
			head.activationType = pol.activationType = GGL::ModelActivationType::LEAKY_RELU;
			head.addResiduals = pol.addResiduals = false;
			head.addOutputLayer = false; pol.addOutputLayer = true;
		} else { // gco
			f.obs = new AdvancedObsPadded(3);
			obsSize = 230;
			f.tickSkip = tsOverride; f.actionDelay = 0;
			head.layerSizes = { 1280, 1280, 1280 };
			pol.layerSizes = { 768, 768, 768 };
			head.addLayerNorm = pol.addLayerNorm = true;
			head.activationType = pol.activationType = GGL::ModelActivationType::LEAKY_RELU;
			head.addResiduals = pol.addResiduals = true;
			head.addOutputLayer = false; pol.addOutputLayer = true;
		}
		f.unit = new GGL::InferUnit(f.obs, obsSize, f.parser, head, pol, dir, false);
		return f;
	}
};

int main() {
	RocketSim::Init("collision_meshes", true);
	FILE* jsonl = nullptr;
	if (const char* jp = getenv("GGL_XP_JSONL"); jp && *jp) {
		jsonl = fopen(jp, "w");
		printf("decision JSONL -> %s\n", jp);
	}
	int targetEps = EnvI("GGL_XP_GOALS", 200);

	Fighter A = Fighter::Make(EnvS("GGL_XP_A_KIND", "gco"), EnvS("GGL_XP_A", ""), EnvI("GGL_XP_A_TS", 1));
	Fighter B = Fighter::Make(EnvS("GGL_XP_B_KIND", "bonk4"), EnvS("GGL_XP_B", ""), EnvI("GGL_XP_B_TS", 4));
	printf("A=%s ts%d  vs  B=%s ts%d   target %d episodes\n",
		A.kind.c_str(), A.tickSkip, B.kind.c_str(), B.tickSkip, targetEps);

	int aGoals = 0, bGoals = 0, eps = 0, capped = 0;
	for (int half = 0; half < 2 && eps < targetEps; half++) {
		// half 0: A=blue B=orange; half 1: swapped
		Fighter* blue = half ? &B : &A;
		Fighter* orange = half ? &A : &B;

		auto arena = RocketSim::Arena::Create(RocketSim::GameMode::SOCCAR);
		arena->AddCar(RocketSim::Team::BLUE);
		arena->AddCar(RocketSim::Team::ORANGE);

		int epsThisHalf = targetEps / 2;
		for (int e = 0; e < epsThisHalf && eps < targetEps; e++) {
			arena->ResetToRandomKickoff();
			blue->ticks = orange->ticks = -1;
			blue->controls = orange->controls = {};
			int scored = -1;      // 0 = blue scored, 1 = orange scored
			int tick = 0;
			const int CAP = 120 * 30; // 30s
			while (scored < 0 && tick < CAP) {
				GameState gs = GameState(arena);
				if (gs.goalScored) {
					// scoring team = the half the ball is NOT in (RS_TEAM_FROM_Y convention)
					scored = (gs.ball.pos.y > 0) ? 0 : 1;
					break;
				}
				for (int ci = 0; ci < 2; ci++) {
					Fighter* f = ci == 0 ? blue : orange;
					Player& p = gs.players[ci];
					p.prevAction = f->controls;
					// GGL_XP_JSONL: emit the SAME per-decision record the RLBot client logs,
					// so research/tools wavedash + fidelity meters run unmodified on SIM play.
					// This is the control for "flips look fine in viz, wrong in game": identical
					// weights, identical meter, only the world differs.
					if (jsonl && ci == 0) {
						fprintf(jsonl,
							"{\"type\":\"decision\",\"t\":%.6f,\"i\":0,\"g\":%d,\"boost\":%.3f,"
							"\"hf\":%d,\"hj\":%d,\"hdj\":%d,\"atsj\":%.4f,\"flip\":%d,"
							"\"p\":[%.3f,%.3f,%.3f],\"v\":[%.3f,%.3f,%.3f],"
							"\"f\":[%.5f,%.5f,%.5f],\"u\":[%.5f,%.5f,%.5f],"
							"\"b\":[%.3f,%.3f,%.3f],"
							"\"act_tuple\":[%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f]}\n",
							tick / 120.0, (int)p.isOnGround, p.boost,
							(int)p.hasFlipped, (int)p.hasJumped, (int)p.hasDoubleJumped,
							p.airTimeSinceJump, (int)p.HasFlipOrJump(),
							p.pos.x, p.pos.y, p.pos.z, p.vel.x, p.vel.y, p.vel.z,
							p.rotMat.forward.x, p.rotMat.forward.y, p.rotMat.forward.z,
							p.rotMat.up.x, p.rotMat.up.y, p.rotMat.up.z,
							gs.ball.pos.x, gs.ball.pos.y, gs.ball.pos.z,
							f->controls[0], f->controls[1], f->controls[2], f->controls[3],
							f->controls[4], f->controls[5], f->controls[6], f->controls[7]);
					}
					if (f->ticks >= f->tickSkip || f->ticks == -1) {
						f->action = f->unit->InferAction(p, gs, /*deterministic=*/true);
						f->ticks = 0;
					}
					if (f->ticks >= f->actionDelay) f->controls = f->action;
					f->ticks++;
					auto* car = arena->_cars[ci];
					CarControls cc{};
					cc.throttle = f->controls[0]; cc.steer = f->controls[1];
					cc.pitch = f->controls[2]; cc.yaw = f->controls[3]; cc.roll = f->controls[4];
					cc.jump = f->controls[5] > 0.5f; cc.boost = f->controls[6] > 0.5f;
					cc.handbrake = f->controls[7] > 0.5f;
					car->controls = cc;
				}
				arena->Step(1);
				tick++;
			}
			if (scored < 0) { capped++; }
			else {
				bool blueScored = (scored == 0);
				bool aScored = (half == 0) ? blueScored : !blueScored;
				if (aScored) aGoals++; else bGoals++;
			}
			eps++;
			if (eps % 20 == 0)
				printf("  %d eps: A %d - %d B (%.1f%%), capped %d\n",
					eps, aGoals, bGoals, 100.0 * aGoals / RS_MAX(aGoals + bGoals, 1), capped);
		}
		delete arena;
	}
	if (jsonl) fclose(jsonl);
	int dec = aGoals + bGoals;
	printf("FINAL: A %d - %d B  -> A share %.1f%% over %d episodes (%d capped)\n",
		aGoals, bGoals, 100.0 * aGoals / RS_MAX(dec, 1), eps, capped);
	return 0;
}
