// Segment replayer against the TUNED vendored RocketSimV3 (the trainer's engine).
// Line protocol on stdin:
//   R px py pz vx vy vz fx fy fz boost grounded   -> restore car state (fresh, angVel 0, roll 0)
//   C thr steer pitch yaw roll jump boost hb      -> set controls, Step(1), print "S px..vz g"
// Every C prints the post-step state, so the driver can score drift at every tick depth.
#include "RocketSim.h"
#include <cstdio>
#include <cmath>

using namespace RocketSim;

int main(int argc, char** argv) {
	Init(argc > 1 ? argv[1] : "collision_meshes", true);
	Arena* arena = Arena::Create(GameMode::SOCCAR);
	Car* car = arena->AddCar(Team::BLUE);

	char op; float a[11];
	while (scanf(" %c", &op) == 1) {
		if (op == 'R') {
			// R p3 v3 fwd3 up3 angVel3 boost grounded hasJumped hasDoubleJumped hasFlipped
			//   isFlipping isJumping airTimeSinceJump   (23 floats) - FULL pose + JUMP/FLIP state.
			// The jump/flip block is REQUIRED for dodge replay: without it the engine thinks the
			// car never jumped, so a dodge press becomes a fresh jump and the trajectory is
			// meaningless. (This is why the first certification silently excluded all dodges.)
			float b[23];
			if (scanf("%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f",
				b,b+1,b+2,b+3,b+4,b+5,b+6,b+7,b+8,b+9,b+10,b+11,b+12,b+13,b+14,b+15,b+16,
				b+17,b+18,b+19,b+20,b+21,b+22) != 23) break;
			CarState cs = car->GetState();
			cs.pos = Vec(b[0], b[1], b[2]);
			cs.vel = Vec(b[3], b[4], b[5]);
			Vec f(b[6], b[7], b[8]), u(b[9], b[10], b[11]);
			f = f.Normalized(); u = u.Normalized();
			Vec r = u.Cross(f).Normalized();
			u = f.Cross(r).Normalized();          // re-orthogonalize
			cs.rotMat = RotMat(f, r, u);
			cs.angVel = Vec(b[12], b[13], b[14]);
			cs.boost = b[15];
			cs.isOnGround = b[16] > 0.5f;
			cs.hasJumped        = b[17] > 0.5f;
			cs.hasDoubleJumped  = b[18] > 0.5f;
			cs.hasFlipped       = b[19] > 0.5f;
			cs.isFlipping       = b[20] > 0.5f;
			cs.isJumping        = b[21] > 0.5f;
			cs.airTimeSinceJump = b[22];
			car->SetState(cs);
		} else if (op == 'C') {
			if (scanf("%f %f %f %f %f %f %f %f",
				a, a+1, a+2, a+3, a+4, a+5, a+6, a+7) != 8) break;
			CarControls cc = {};
			cc.throttle = a[0]; cc.steer = a[1]; cc.pitch = a[2]; cc.yaw = a[3]; cc.roll = a[4];
			cc.jump = a[5] > 0.5f; cc.boost = a[6] > 0.5f; cc.handbrake = a[7] > 0.5f;
			car->controls = cc;
			arena->Step(1);
			CarState o = car->GetState();
			// Trailing 4 columns appended 2026-08-25 (hasFlipped, isFlipping, isJumping,
			// airTimeSinceJump) - existing drivers index the first 10 positionally.
			printf("S %.4f %.4f %.4f %.4f %.4f %.4f %d %.5f %.5f %.5f %d %d %d %.4f\n",
				o.pos.x, o.pos.y, o.pos.z, o.vel.x, o.vel.y, o.vel.z, (int)o.isOnGround,
				o.rotMat.forward.x, o.rotMat.forward.y, o.rotMat.forward.z,
				(int)o.hasFlipped, (int)o.isFlipping, (int)o.isJumping, o.airTimeSinceJump);
		}
	}
	return 0;
}
