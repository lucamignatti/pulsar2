#include "NextoOpponent.h"

#include <RLGymCPP/ActionParsers/DefaultAction.h>
#include <RLGymCPP/CommonValues.h>

using namespace GGL;

// Nexto's lookup table, ported verbatim from rlbot-run/nexto/agent.py
// (make_lookup_table): 8-tuples of [throttle, steer, pitch, yaw, roll, jump,
// boost, handbrake]. Ground first, then aerial - 90 rows total.
static std::vector<std::array<float, 8>> MakeNextoLookup() {
	std::vector<std::array<float, 8>> out;
	// Ground
	for (int throttle = -1; throttle <= 1; throttle++)
		for (int steer = -1; steer <= 1; steer++)
			for (int boost = 0; boost <= 1; boost++)
				for (int handbrake = 0; handbrake <= 1; handbrake++) {
					if (boost == 1 && throttle != 1)
						continue;
					out.push_back({ (float)(throttle != 0 ? throttle : boost), (float)steer, 0.f,
						(float)steer, 0.f, 0.f, (float)boost, (float)handbrake });
				}
	// Aerial
	for (int pitch = -1; pitch <= 1; pitch++)
		for (int yaw = -1; yaw <= 1; yaw++)
			for (int roll = -1; roll <= 1; roll++)
				for (int jump = 0; jump <= 1; jump++)
					for (int boost = 0; boost <= 1; boost++) {
						if (jump == 1 && yaw != 0)
							continue; // only need roll for sideflip
						if (pitch == 0 && roll == 0 && jump == 0)
							continue; // duplicate with ground
						float handbrake = (jump == 1 && (pitch != 0 || yaw != 0 || roll != 0)) ? 1.f : 0.f;
						out.push_back({ (float)boost, (float)yaw, (float)pitch, (float)yaw,
							(float)roll, (float)jump, (float)boost, handbrake });
					}
	return out;
}

GGL::NextoOpponent::NextoOpponent(const std::string& modelPath, torch::Device device)
	: device(device) {

	try {
		model = torch::jit::load(modelPath, device);
		model.eval();
	} catch (const std::exception& e) {
		RG_ERR_CLOSE("NextoOpponent: failed to load TorchScript model from \"" << modelPath
			<< "\": " << e.what());
	}

	lookup = MakeNextoLookup();
	RG_ASSERT(lookup.size() == 90);

	// Map each Nexto row to our DefaultAction row by exact tuple equality. Both
	// tables descend from the same lineage so this should be the identity - but
	// mapping (and hard-failing on a miss) makes that a checked fact, not a hope.
	RLGC::DefaultAction ours = {};
	actionMap.resize(lookup.size());
	for (size_t i = 0; i < lookup.size(); i++) {
		int found = -1;
		for (size_t j = 0; j < ours.actions.size(); j++) {
			const auto& a = ours.actions[j];
			const float v[8] = { a.throttle, a.steer, a.pitch, a.yaw, a.roll, a.jump, a.boost, a.handbrake };
			bool same = true;
			for (int d = 0; d < 8 && same; d++)
				same = (v[d] == lookup[i][d]);
			if (same) {
				found = (int)j;
				break;
			}
		}
		if (found < 0)
			RG_ERR_CLOSE("NextoOpponent: Nexto action row " << i << " has no equal in our "
				"DefaultAction table - the tables have diverged, refusing to serve garbage controls");
		actionMap[i] = found;
	}

	RG_LOG("NextoOpponent: model loaded from " << modelPath
		<< ", all 90 actions mapped onto DefaultAction");
}

void GGL::NextoOpponent::BeginServe(int numPlayers) {
	prevActions.assign(numPlayers, {});
}

// Feature layout of one entity row (nexto_obs.py): [IS_SELF, IS_MATE, IS_OPP,
// IS_BALL, IS_BOOST, pos(3), lin_vel(3), fw(3), up(3), ang_vel(3), BOOST, DEMO,
// ON_GROUND, HAS_FLIP] = 24. q = self row + previous parsed action (8) = 32.
static constexpr int KV_W = 24, Q_W = 32;
static constexpr int F_SELF = 0, F_MATE = 1, F_OPP = 2, F_BALL = 3, F_BOOST = 4;
static constexpr int F_POS = 5, F_VEL = 8, F_FW = 11, F_UP = 14, F_ANGVEL = 17;
static constexpr int F_BOOSTAMT = 20, F_DEMO = 21, F_GROUND = 22, F_FLIP = 23;

void GGL::NextoOpponent::Act(const std::vector<RLGC::GameState>& states,
	const std::vector<bool>& isOld,
	const std::vector<uint8_t>& prevArenaTerminals,
	std::vector<int>& outActionIdx) {

	RG_NO_GRAD;
	using namespace RLGC;

	// Entity counts differ by mode (2/4/6 players + ball + 34 pads), so batch per
	// mode: [0]=1v1, [1]=2v2, [2]=3v3
	std::vector<float> qBuf[3], kvBuf[3];
	std::vector<int> who[3]; // global player index per batch row

	int globalIdx = 0;
	for (int arenaIdx = 0; arenaIdx < (int)states.size(); arenaIdx++) {
		const GameState& gs = states[arenaIdx];
		int n = (int)gs.players.size();
		int mode = n / 2 - 1;
		int nEnt = n + 1 + CommonValues::BOOST_LOCATIONS_AMOUNT;

		bool freshEpisode = arenaIdx < (int)prevArenaTerminals.size()
			&& prevArenaTerminals[arenaIdx] != 0;

		for (int o = 0; o < n; o++, globalIdx++) {
			int g = globalIdx;
			if (!isOld[g])
				continue;
			if (freshEpisode)
				prevActions[g] = {};

			const Player& self = gs.players[o];
			// Orange observers see the mirrored field (negate x, y of every vector)
			const float inv = (self.team == Team::ORANGE) ? -1.f : 1.f;

			auto& kv = kvBuf[mode];
			size_t base = kv.size();
			kv.resize(base + (size_t)nEnt * KV_W, 0.f);
			float* K = kv.data() + base;

			auto fnVec3 = [&](float* row, int at, const Vec& v, float norm) {
				row[at + 0] = inv * v.x / norm;
				row[at + 1] = inv * v.y / norm;
				row[at + 2] = v.z / norm;
			};

			// Player rows (arena order)
			for (int j = 0; j < n; j++) {
				const Player& p = gs.players[j];
				float* R = K + (size_t)j * KV_W;
				R[F_SELF] = (j == o) ? 1.f : 0.f;
				// Post-swap semantics of nexto_obs: mate = same team (self included),
				// opp = other team
				R[F_MATE] = (p.team == self.team) ? 1.f : 0.f;
				R[F_OPP] = (p.team != self.team) ? 1.f : 0.f;
				fnVec3(R, F_POS, p.pos, 2300.f);
				fnVec3(R, F_VEL, p.vel, 2300.f);
				fnVec3(R, F_FW, p.rotMat.forward, 1.f);
				fnVec3(R, F_UP, p.rotMat.up, 1.f);
				fnVec3(R, F_ANGVEL, p.angVel, 5.5f);
				R[F_BOOSTAMT] = p.boost / 100.f;
				R[F_DEMO] = p.isDemoed ? 1.f : 0.f;
				R[F_GROUND] = p.isOnGround ? 1.f : 0.f;
				R[F_FLIP] = p.HasFlipOrJump() ? 1.f : 0.f;
			}

			// Ball row
			{
				float* R = K + (size_t)n * KV_W;
				R[F_BALL] = 1.f;
				fnVec3(R, F_POS, gs.ball.pos, 2300.f);
				fnVec3(R, F_VEL, gs.ball.vel, 2300.f);
				fnVec3(R, F_ANGVEL, gs.ball.angVel, 5.5f);
			}

			// Pad rows: absolute list order; flags stay absolute-order even for the
			// mirrored orange view (nexto_obs ships this pairing; replicate exactly)
			for (int b = 0; b < CommonValues::BOOST_LOCATIONS_AMOUNT; b++) {
				float* R = K + (size_t)(n + 1 + b) * KV_W;
				const Vec& loc = CommonValues::BOOST_LOCATIONS[b];
				R[F_BOOST] = 1.f;
				fnVec3(R, F_POS, loc, 2300.f);
				R[F_BOOSTAMT] = 0.12f + 0.88f * (loc.z > 72.f);
				R[F_DEMO] = gs.boostPads[b] ? 1.f : 0.f;
			}

			// q = self row + previous parsed action
			auto& q = qBuf[mode];
			size_t qBase = q.size();
			q.resize(qBase + Q_W, 0.f);
			float* Q = q.data() + qBase;
			std::copy(K + (size_t)o * KV_W, K + (size_t)o * KV_W + KV_W, Q);
			for (int d = 0; d < 8; d++)
				Q[KV_W + d] = prevActions[g][d];

			// Relative frame: subtract self pos from every kv row, then rotate all
			// (x, y) pairs of pos/vel/fw/up/angvel by theta = atan2(fw.x, fw.y) of
			// self. q itself stays un-rotated (nexto_obs only transforms kv).
			const float sx = Q[F_POS], sy = Q[F_POS + 1], sz = Q[F_POS + 2];
			const float theta = atan2f(Q[F_FW], Q[F_FW + 1]);
			const float ct = cosf(theta), st = sinf(theta);
			for (int e = 0; e < nEnt; e++) {
				float* R = K + (size_t)e * KV_W;
				R[F_POS] -= sx;
				R[F_POS + 1] -= sy;
				R[F_POS + 2] -= sz;
				for (int at : { F_POS, F_VEL, F_FW, F_UP, F_ANGVEL }) {
					float x = R[at], y = R[at + 1];
					R[at] = ct * x - st * y;
					R[at + 1] = st * x + ct * y;
				}
			}

			who[mode].push_back(g);
		}
	}

	// One batched forward per mode present; masks are all-zero (exact player counts)
	for (int mode = 0; mode < 3; mode++) {
		if (who[mode].empty())
			continue;
		int64_t b = (int64_t)who[mode].size();
		int64_t nEnt = 2 * (mode + 1) + 1 + CommonValues::BOOST_LOCATIONS_AMOUNT;
		auto tQ = torch::from_blob(qBuf[mode].data(), { b, 1, Q_W }, torch::kFloat32).to(device);
		auto tKv = torch::from_blob(kvBuf[mode].data(), { b, nEnt, KV_W }, torch::kFloat32).to(device);
		auto tM = torch::zeros({ b, nEnt }, torch::TensorOptions().device(device));

		auto out = model.forward({ std::make_tuple(tQ, tKv, tM) }).toTuple();
		auto logits = out->elements()[0].toTensor(); // [b, 90]
		auto picks = logits.argmax(-1).cpu();
		auto pickAcc = picks.accessor<int64_t, 1>();

		for (int64_t i = 0; i < b; i++) {
			int nextoIdx = (int)pickAcc[i];
			int g = who[mode][i];
			outActionIdx[g] = actionMap[nextoIdx];
			prevActions[g] = lookup[nextoIdx];
		}
	}
}
