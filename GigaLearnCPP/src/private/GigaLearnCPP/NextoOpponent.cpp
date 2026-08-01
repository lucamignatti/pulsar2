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

// Rewrite every literal `Device = prim::Constant[value="cpu"]` in `graph` to `dev`, recursing
// into nested blocks (if/loop bodies). Returns how many it changed.
//
// WHY THIS EXISTS (2026-07-26). Nexto's traced graph contains exactly one such node:
//     %13 : Device = prim::Constant[value="cpu"]()
//     %input.2 = aten::to(%14, dtype=float32, layout, %13, ...)   # Necto/training/agent.py:31
// i.e. a CONSTANT tensor is explicitly sent to CPU mid-graph, and then lands as `mat1` in an
// addmm against CUDA weights - the "Expected all tensors to be on the same device, but got mat1
// is on cpu" that made the CUDA probe fail since the day Nexto was added. It is NOT the weights:
// all 44 params move fine, there are 0 buffers and 0 stray tensor attributes. Neither
// torch::jit::load(path, cuda) nor Module::to(cuda) helps, because both remap TENSORS and this
// is a literal Device value baked into the IR.
//
// Cost of not fixing it (measured over 1147 iterations of the 5.1 run): Nexto-serving iterations
// ran 19.56s of wall vs 4.26s for every other iteration, at a realized 15.6% serve rate - 36% of
// ALL training wall clock spent with the GPU at 3% utilization while a 460k-param transformer
// ground through ~2047 rows/step on the 2 intra-op threads that Learner.cpp:184 deliberately
// caps libtorch to. On GPU it fits inside the ~1.7s of collection slack the pipeline already
// wastes behind the learn pass, so it costs approximately nothing.
static int RetargetDeviceConstants(torch::jit::Block* block, torch::Device dev) {
	int changed = 0;
	for (torch::jit::Node* node : block->nodes()) {
		if (node->kind() == torch::jit::prim::Constant
			&& node->output()->type()->kind() == c10::TypeKind::DeviceObjType
			&& node->hasAttribute(c10::attr::value)) {
			// Device constants are serialized as their string form ("cpu", "cuda:0").
			if (torch::Device(node->s(c10::attr::value)).is_cpu()) {
				node->s_(c10::attr::value, dev.str());
				changed++;
			}
		}
		for (torch::jit::Block* sub : node->blocks())
			changed += RetargetDeviceConstants(sub, dev);
	}
	return changed;
}

// Same, over every method of the module and all of its submodules - the offending constant lives
// inside a submodule's forward, not the top-level graph (the top-level graph has zero of them;
// it only shows up once the graph is inlined).
static int RetargetModuleDeviceConstants(torch::jit::Module& mod, torch::Device dev) {
	int changed = 0;
	for (const auto& sub : mod.named_modules())
		for (auto& method : sub.value.get_methods())
			changed += RetargetDeviceConstants(method.graph()->block(), dev);
	return changed;
}

GGL::NextoOpponent::NextoOpponent(const std::string& modelPath, torch::Device device)
	: device(torch::kCPU) {

	try {
		model = torch::jit::load(modelPath, torch::kCPU);
		model.eval();
	} catch (const std::exception& e) {
		RG_ERR_CLOSE("NextoOpponent: failed to load TorchScript model from \"" << modelPath
			<< "\": " << e.what());
	}

	// Device probe (2026-07-20 crash-loop fix): the trace was authored on CPU and
	// TorchScript graphs can carry baked-in device constants that only surface as
	// a runtime_error at forward - which, on the pipelined collect worker, is an
	// unprintable std::terminate (the exact live-crash signature; the CPU-
	// sequential smoke couldn't see it). Probe the requested device with a dummy
	// forward at BOOT, where failure is loud and cheap, and fall back to CPU.
	if (device.is_cuda()) {
		try {
			auto trial = torch::jit::load(modelPath, device);
			trial.eval();
			torch::NoGradGuard noGrad;

			// Move the baked-in CPU device literal(s) onto the target device. Without this the
			// forward below throws "mat1 is on cpu" and we fall back to CPU, which is exactly
			// what happened on every boot from 2026-07-20 to 2026-07-26.
			int retargeted = RetargetModuleDeviceConstants(trial, device);

			// RANDOM probe input, not zeros: a graph that is still half-on-CPU can survive an
			// all-zeros forward, and zeros make every logit tie so an argmax check would be
			// meaningless. Fixed seed so a failure is reproducible from the log.
			torch::manual_seed(20260726);
			auto tQ = torch::randn({ 64, 1, 32 }, torch::TensorOptions().device(torch::kCPU));
			auto tKv = torch::randn({ 64, 37, 24 }, torch::TensorOptions().device(torch::kCPU));
			auto tM = torch::zeros({ 64, 37 }, torch::TensorOptions().device(torch::kCPU));

			auto outDev = trial.forward({ std::make_tuple(
				tQ.to(device), tKv.to(device), tM.to(device)) }).toTuple();
			auto logitsDev = outDev->elements()[0].toTensor().to(torch::kCPU, torch::kFloat32);
			if (!logitsDev.isfinite().all().item<bool>())
				throw std::runtime_error("non-finite probe logits");

			// EQUIVALENCE GUARD. Being finite is not being correct - the whole point of a graph
			// rewrite is that it could silently change semantics. `model` is still the untouched
			// CPU load, so run it on the SAME input and require agreement. Act() only ever
			// consumes argmax(logits), so argmax agreement is the metric that matters; the value
			// tolerance is deliberately loose because the trainer enables TF32 (~10-bit mantissa
			// on matmul), which is a real numeric difference and NOT a bug.
			auto logitsRef = model.forward({ std::make_tuple(tQ, tKv, tM) })
				.toTuple()->elements()[0].toTensor().to(torch::kFloat32);
			float scale = std::max(1.f, logitsRef.abs().max().item<float>());
			float maxDev = (logitsDev - logitsRef).abs().max().item<float>() / scale;
			float agree = (logitsDev.argmax(-1) == logitsRef.argmax(-1))
				.to(torch::kFloat32).mean().item<float>();
			if (maxDev > 5e-2f)
				throw std::runtime_error("CUDA/CPU logit mismatch (max rel dev "
					+ std::to_string(maxDev) + ")");
			if (agree < 0.95f)
				throw std::runtime_error("CUDA/CPU argmax disagreement (agreement "
					+ std::to_string(agree) + ")");

			model = trial;
			this->device = device;
			RG_LOG("NextoOpponent: serving on " << device << " (retargeted " << retargeted
				<< " baked CPU device constant(s); probe max rel dev " << maxDev
				<< ", argmax agreement " << (agree * 100.f) << "%)");
		} catch (const std::exception& e) {
			// No longer expected: as of 2026-07-26 the baked CPU device constant is retargeted
			// above, so this path should now be genuinely rare. Landing here means either the
			// rewrite missed a constant (new/changed model file) or the equivalence guard
			// rejected the CUDA output - in BOTH cases falling back to CPU is correct and
			// costs only throughput, never correctness. If it fires, the run silently loses
			// ~36% of its wall clock, so treat a "serving on CPU" line as a perf regression
			// worth chasing rather than routine noise.
			//
			// torch::jit's what() carries the entire serialized TorchScript traceback (~40 lines
			// of someone else's Windows paths) which drowned the boot log on every restart. Keep
			// only the last non-empty line - that is the actual root cause - and put the full
			// dump behind GGL_VERBOSE_NEXTO=1.
			std::string msg = e.what();
			if (const char* v = std::getenv("GGL_VERBOSE_NEXTO"); !(v && v[0] && std::string(v) != "0")) {
				size_t end = msg.find_last_not_of(" \t\r\n");
				if (end != std::string::npos) {
					size_t nl = msg.find_last_of('\n', end);
					size_t start = (nl == std::string::npos) ? 0 : nl + 1;
					msg = msg.substr(start, end - start + 1);
				}
			}
			RG_LOG("NextoOpponent: CUDA probe failed (" << msg
				<< ") - serving on CPU instead (expected; set GGL_VERBOSE_NEXTO=1 for the full trace)");
		}
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
				// NOT HasFlipOrJump(). The reference this obs reproduces is rlgym_compat
				// v1/player_data.py, which defines has_flip as exactly
				//     not has_flipped and not has_double_jumped
				//     and air_time_since_jump < DOUBLEJUMP_MAX_DELAY
				// with NO on-ground disjunct. HasFlipOrJump() ORs in isOnGround, so a car
				// mid-ground-flip (hasFlipped already set while the wheels are still down)
				// reports "flip available" where the reference reports spent. Nexto's net was
				// trained against the reference meaning, so feeding it ours puts that feature
				// off-distribution. Compute the reference expression directly.
				R[F_FLIP] = (!p.hasFlipped && !p.hasDoubleJumped
					&& p.airTimeSinceJump < RLConst::DOUBLEJUMP_MAX_DELAY) ? 1.f : 0.f;
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

		// try/catch so a worker-thread failure PRINTS before the process dies -
		// an uncaught exception on the collect worker is a bare std::terminate
		// with no message (learned from the 2026-07-20 crash loop)
		torch::Tensor logits;
		try {
			auto out = model.forward({ std::make_tuple(tQ, tKv, tM) }).toTuple();
			logits = out->elements()[0].toTensor(); // [b, 90]
		} catch (const std::exception& e) {
			RG_ERR_CLOSE("NextoOpponent::Act: forward failed (mode " << mode
				<< ", batch " << b << ", device " << device << "): " << e.what());
		}
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
