#include "DipSearch.h"
#include <private/GigaLearnCPP/PPO/PPOLearner.h>
#include "Timer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <random>
#include <thread>
#include <type_traits>

using namespace GGL;

// ---------------------------------------------------------------------------- bank
int64_t DipSnapshotBank::Push(ArenaSnapshot&& snap) {
	std::lock_guard<std::mutex> lk(mtx);
	frames.push_back(std::move(snap));
	return firstId + (int64_t)frames.size() - 1;
}

bool DipSnapshotBank::Get(int64_t id, ArenaSnapshot& out) const {
	std::lock_guard<std::mutex> lk(mtx);
	if (id < firstId || id >= firstId + (int64_t)frames.size())
		return false;
	out = frames[(size_t)(id - firstId)];
	return true;
}

void DipSnapshotBank::Trim() {
	std::lock_guard<std::mutex> lk(mtx);
	while (frames.size() > capacity) {
		frames.pop_front();
		firstId++;
	}
}

size_t DipSnapshotBank::Size() const {
	std::lock_guard<std::mutex> lk(mtx);
	return frames.size();
}

// ---------------------------------------------------------------------------- bytes
namespace {
	struct SnapHdr {
		uint64_t tickCount, rngState;
		uint32_t nCars, nPads, extraLen;
	};
	static_assert(std::is_trivially_copyable<BallState>::value, "BallState must be memcpy-able");
	static_assert(std::is_trivially_copyable<VizCarState>::value, "VizCarState must be memcpy-able");
	static_assert(std::is_trivially_copyable<CarControls>::value, "CarControls must be memcpy-able");
	static_assert(std::is_trivially_copyable<BallHitInfo>::value, "BallHitInfo must be memcpy-able");
	template<class T> void Put(std::vector<uint8_t>& out, const T& v) {
		const uint8_t* b = reinterpret_cast<const uint8_t*>(&v);
		out.insert(out.end(), b, b + sizeof(T));
	}
	template<class T> bool Get(const uint8_t*& p, const uint8_t* end, T& v) {
		if (p + sizeof(T) > end) return false;
		std::memcpy(&v, p, sizeof(T));
		p += sizeof(T);
		return true;
	}
}

std::vector<uint8_t> GGL::SerializeSnapshot(const ArenaSnapshot& snap) {
	std::vector<uint8_t> out;
	SnapHdr h = {};
	h.tickCount = snap.tickCount;
	h.rngState = snap.rngState;
	h.nCars = (uint32_t)snap.cars.size();
	h.nPads = (uint32_t)snap.pads.size();
	h.extraLen = snap.cars.empty() ? 0u : (uint32_t)snap.cars[0].engineExtra.size();
	Put(out, h);
	Put(out, snap.ball);
	for (auto& c : snap.cars) {
		Put(out, c.id);
		uint8_t team = (uint8_t)c.team;
		Put(out, team);
		Put(out, c.state);
		Put(out, c.controls);
		Put(out, c.ballHitInfo);
		RG_ASSERT(c.engineExtra.size() == h.extraLen);
		out.insert(out.end(), c.engineExtra.begin(), c.engineExtra.end());
	}
	for (auto& pd : snap.pads) {
		uint8_t act = pd.isActive ? 1 : 0;
		Put(out, act);
		Put(out, pd.cooldown);
		Put(out, pd.lockedCarId);
	}
	return out;
}

bool GGL::DeserializeSnapshot(const uint8_t* data, size_t nbytes, ArenaSnapshot& out) {
	const uint8_t* p = data;
	const uint8_t* end = data + nbytes;
	SnapHdr h = {};
	if (!Get(p, end, h)) return false;
	out = ArenaSnapshot();
	out.tickCount = h.tickCount;
	out.rngState = h.rngState;
	if (!Get(p, end, out.ball)) return false;
	out.cars.resize(h.nCars);
	for (auto& c : out.cars) {
		uint8_t team = 0;
		if (!Get(p, end, c.id) || !Get(p, end, team) || !Get(p, end, c.state)
			|| !Get(p, end, c.controls) || !Get(p, end, c.ballHitInfo))
			return false;
		c.team = (Team)team;
		if (p + h.extraLen > end) return false;
		c.engineExtra.assign(p, p + h.extraLen);
		p += h.extraLen;
	}
	out.pads.resize(h.nPads);
	for (auto& pd : out.pads) {
		uint8_t act = 0;
		if (!Get(p, end, act) || !Get(p, end, pd.cooldown) || !Get(p, end, pd.lockedCarId))
			return false;
		pd.isActive = act != 0;
	}
	return p == end;
}

// ---------------------------------------------------------------------------- pool
struct DipSearch::Slot {
	Arena* arena = nullptr;
	RLGC::ObsBuilder* obsBuilder = nullptr;
	RLGC::ActionParser* actionParser = nullptr;
	std::vector<RLGC::WeightedReward> rewards;
	void* userInfo = nullptr;
	RLGC::GameState gs, prevGs;
	std::vector<RLGC::Action> acts;
	// per-job state
	int nCars = 0;
	int job = -1;
	bool alive = false;
	int selfSlot = 0;
	Team selfTeam = Team::BLUE;
	float score = 0;
	int goal = 0;
	std::vector<float> obs;        // numPlayers x obsSize, policy input space
	std::vector<uint8_t> masks;    // numPlayers x numActions
	std::vector<int> curActions;   // numPlayers
	std::vector<int> holdLeft;     // VTS: ticks each car still owes on its hold (0 = due)
	std::vector<uint8_t> due;      // this step: car takes a fresh decision
	std::vector<int> intentZ;      // INTENT CLASS: per-car intent id
	std::vector<int> intentLeft;   // INTENT CLASS: decisions left on the current intent (0 = draw)
	int selfDecisions = 0;         // VTS: decisions the self car has taken so far
	int tickAcc = 0;               // physics ticks advanced so far
	float discount = 1.f;
	bool needLeaf = false;
	float stepReward = 0;
	int stepGoal = 0;
	int stepTicks = 0;
};

struct DipSearch::Job {
	int state = 0;
	int kind = 0;                  // 0 policy, 1 open-loop prefix, 2 policy-at-temperature prefix, 3 INTENT sequence
	int L = 0;                     // prefix length in macros (kind 3: in intent periods)
	float temp = 1.f;
	std::vector<int> actions;      // kind 1: the prefix (macro-action indices)
	std::vector<int> intents;      // kind 3: the prefix (intent ids, one per period)
	bool record = false;
	// outputs
	float score = 0;
	int goal = 0;
	std::vector<int> executed;     // the self player's action at each macro boundary of the prefix
	std::vector<float> recObs;
	std::vector<int32_t> recAct;
	std::vector<uint8_t> recMask;
	std::vector<int32_t> recIntentId;
	std::vector<float> recIntentFeat;
};

DipSearch::DipSearch(const DipSearchConfig& cfg, RLGC::EnvCreateFn envCreateFn,
	int tickSkip, int actionDelay, int obsSize, int numActions, float gamma,
	const VarTickSkipConfig* vtsCfg, int intentDim, int intentPeriod)
	: cfg(cfg), tickSkip(tickSkip), actionDelay(actionDelay), obsSize(obsSize),
	numActions(numActions), gamma(gamma), intentDim(intentDim), intentPeriod(RS_MAX(1, intentPeriod)),
	intentOn(intentDim > 0) {
#if GGL_DIP_HAS_VTS
	if (vtsCfg && vtsCfg->enabled && vtsCfg->innerActions > 0) {
		vts = *vtsCfg;
		vtsOn = true;
		RG_ASSERT(actionDelay == 0);
	}
#else
	(void)vtsCfg;
#endif

	pool.resize((size_t)cfg.poolSize, nullptr);
	// Arena creation is the expensive part (collision meshes); build in parallel like EnvSet.
	ParallelFor(cfg.poolSize, [&](int i) {
		auto r = envCreateFn(i);
		auto* s = new Slot();
		s->arena = r.arena;
		s->obsBuilder = r.obsBuilder;
		s->actionParser = r.actionParser;
		s->rewards = r.rewards;
		s->userInfo = r.userInfo;
		// Terminal conditions and the state setter are never used by the pool: episodes
		// end only on goals and states come from snapshots.
		for (auto* t : r.terminalConditions)
			delete t;
		delete r.stateSetter;
		s->acts.resize(r.arena->_cars.size());
		s->obs.resize(r.arena->_cars.size() * (size_t)obsSize);
		s->masks.resize(r.arena->_cars.size() * (size_t)numActions);
		s->curActions.resize(r.arena->_cars.size());
		s->holdLeft.assign(r.arena->_cars.size(), 0);
		s->due.assign(r.arena->_cars.size(), 1);
		s->intentZ.assign(r.arena->_cars.size(), 0);
		s->intentLeft.assign(r.arena->_cars.size(), 0);
		s->nCars = (int)r.arena->_cars.size();
		pool[(size_t)i] = s;
	});
	// The fleet's EnvCreateFn may be index-dependent (team size by arena index), so the
	// pool built from the same indices carries the same mix; jobs are placed on a slot
	// whose car count matches their snapshot.
	std::map<int, int> mix;
	for (auto* sl : pool) mix[sl->nCars]++;
	numPlayers = pool[0]->nCars;
	std::string mixStr;
	for (auto& kv : mix) mixStr += RS_STR(kv.first << "-car x" << kv.second << " ");
	RG_LOG("DipSearch: pool of " << cfg.poolSize << " arenas (" << mixStr << "), horizon "
		<< cfg.horizon << " steps, macro " << cfg.macro << ", " << cfg.threads << " threads"
		<< (vtsOn ? " [VTS: per-tick gamma, hold buckets]" : "")
		<< (intentOn ? RS_STR(" [INTENT CLASS: " << intentDim << " intents x " << this->intentPeriod << " decisions; candidates = intent sequences]") : ""));
}

DipSearch::~DipSearch() {
	for (auto* s : pool) {
		if (!s) continue;
		delete s->arena;
		delete s->obsBuilder;
		delete s->actionParser;
		for (auto& w : s->rewards)
			delete w.reward;
		delete s;
	}
}

void DipSearch::ParallelFor(int n, const std::function<void(int)>& fn) {
	int nt = RS_CLAMP(cfg.threads, 1, RS_MAX(1, n));
	if (nt == 1) {
		for (int i = 0; i < n; i++) fn(i);
		return;
	}
	std::vector<std::thread> ts;
	ts.reserve((size_t)nt);
	for (int t = 0; t < nt; t++) {
		ts.emplace_back([&, t]() {
			for (int i = t; i < n; i += nt)
				fn(i);
		});
	}
	for (auto& th : ts) th.join();
}

static void BuildRows(DipSearch* self, int obsSize, int numActions, RLGC::ObsBuilder* ob,
	RLGC::ActionParser* ap, RLGC::GameState& gs, std::vector<float>& obs, std::vector<uint8_t>& masks,
	const std::vector<double>* normMean, const std::vector<double>* normStd) {
	(void)self;
	for (size_t p = 0; p < gs.players.size(); p++) {
		FList row = ob->BuildObs(gs.players[p], gs);
		RG_ASSERT((int)row.size() == obsSize);
		float* dst = &obs[p * (size_t)obsSize];
		for (int j = 0; j < obsSize; j++)
			dst[j] = normMean ? (float)((row[j] - (*normMean)[j]) / (*normStd)[j]) : row[j];
		auto m = ap->GetActionMask(gs.players[p], gs);
		RG_ASSERT((int)m.size() == numActions);
		std::copy(m.begin(), m.end(), masks.begin() + (ptrdiff_t)(p * (size_t)numActions));
	}
}

void DipSearch::Restore(Slot& s, const ArenaSnapshot& snap) {
	bool ok = snap.ApplyTo(s.arena);
	if (!ok && snap.cars.size() == s.arena->_cars.size() && snap.pads.size() == s.arena->_boostPads.size()) {
		// Same shape, different car ids (the engine hands out ids per arena or globally
		// depending on the build): remap by index. Team order must agree, or the row's
		// player slot would land on the wrong car.
		ArenaSnapshot re = snap;
		std::map<uint32_t, uint32_t> idMap;
		bool teamsOk = true;
		for (size_t i = 0; i < re.cars.size(); i++) {
			teamsOk &= re.cars[i].team == s.arena->_cars[i]->team;
			idMap[re.cars[i].id] = s.arena->_cars[i]->id;
			re.cars[i].id = s.arena->_cars[i]->id;
		}
		for (auto& pd : re.pads)
			if (pd.lockedCarId) {
				auto it = idMap.find(pd.lockedCarId);
				pd.lockedCarId = it == idMap.end() ? 0 : it->second;
			}
		ok = teamsOk && re.ApplyTo(s.arena);
	}
	if (!ok)
		RG_ERR_CLOSE("DipSearch: snapshot does not match the pool arena (snapshot cars=" << snap.cars.size()
			<< " pads=" << snap.pads.size() << ", arena cars=" << s.arena->_cars.size()
			<< " pads=" << s.arena->_boostPads.size() << ")");
	// The obs's prevAction is the action whose controls the snapshot's cars still hold
	// (pending controls == the previously parsed action), exactly what the banked row saw.
	for (size_t i = 0; i < s.arena->_cars.size(); i++)
		s.acts[i] = RLGC::Action(s.arena->_cars[i]->controls);
	s.gs = RLGC::GameState();
	s.gs.userInfo = s.userInfo;
	s.gs.UpdateFromArena(s.arena, s.acts, NULL);
	s.gs.userInfo = s.userInfo;
	s.obsBuilder->Reset(s.gs);
	for (auto& w : s.rewards)
		w.reward->Reset(s.gs);
	s.prevGs.MakeEmpty();
}

// One lockstep wave: every job restored into a slot, stepped `horizon` decisions or to a
// goal, scored as sum gamma^t r_t (critic units) + gamma^T V(s_T) on survivors.
void DipSearch::RunWave(std::vector<Job>& jobs, const std::vector<DipSearchState>& states,
	PPOLearner* ppo, const std::vector<double>* normMean, const std::vector<double>* normStd,
	float rewardScale, float rewardClip) {

	RG_NO_GRAD;
	RG_ASSERT(jobs.size() <= pool.size());
	const int nJobs = (int)jobs.size();

	// place each job on a free slot whose car count matches its snapshot
	std::vector<int> jobSlot((size_t)nJobs, -1);
	{
		std::map<int, std::vector<int>> freeByCars;
		for (int i = (int)pool.size() - 1; i >= 0; i--)
			freeByCars[pool[(size_t)i]->nCars].push_back(i);
		for (int j = 0; j < nJobs; j++) {
			int need = (int)states[(size_t)jobs[(size_t)j].state].snap.cars.size();
			auto it = freeByCars.find(need);
			if (it == freeByCars.end() || it->second.empty()) {
				jobs[(size_t)j].score = std::numeric_limits<float>::quiet_NaN();
				jobs[(size_t)j].goal = 0;
				lastSkippedJobs++;
				continue;
			}
			jobSlot[(size_t)j] = it->second.back();
			it->second.pop_back();
		}
	}
	std::vector<int> assigned;
	for (int j = 0; j < nJobs; j++)
		if (jobSlot[(size_t)j] >= 0) assigned.push_back(j);

	// restore + first obs (parallel)
	ParallelFor((int)assigned.size(), [&](int ai) {
		const int j = assigned[(size_t)ai];
		Slot& s = *pool[(size_t)jobSlot[(size_t)j]];
		Job& job = jobs[(size_t)j];
		s.job = j;
		const auto& st = states[(size_t)job.state];
		Restore(s, st.snap);
		s.selfSlot = st.slot;
		s.selfTeam = s.gs.players[(size_t)st.slot].team;
		s.alive = true;
		s.score = 0;
		s.goal = 0;
		std::fill(s.holdLeft.begin(), s.holdLeft.end(), 0); // every car is due at the restored step
		std::fill(s.due.begin(), s.due.end(), 1);
		std::fill(s.intentLeft.begin(), s.intentLeft.end(), 0); // every car draws an intent at the restored step
		s.selfDecisions = 0;
		s.tickAcc = 0;
		s.discount = 1.f;
		s.needLeaf = false;
		BuildRows(this, obsSize, numActions, s.obsBuilder, s.actionParser, s.gs, s.obs, s.masks, normMean, normStd);
		job.executed.clear();
		job.recObs.clear(); job.recAct.clear(); job.recMask.clear();
		job.recIntentId.clear(); job.recIntentFeat.clear();
	});

	const bool hp = ppo->config.useHalfPrecision;
	std::mt19937 rng((unsigned)(std::hash<int>()(nJobs) ^ 0x9e3779b9u) ^ (unsigned)lastWaves);

	std::vector<int> aliveIdx;
	std::vector<float> obsBuf;
	std::vector<uint8_t> maskBuf;
	std::vector<float> tempBuf;
	std::vector<float> intentFeatBuf;
	std::vector<int64_t> intentIdBuf;
	std::uniform_int_distribution<int> zRoll(0, RS_MAX(0, intentDim - 1));
	const int horizonTicks = cfg.horizon * RS_MAX(1, tickSkip);
	const int maxSteps = vtsOn ? cfg.horizon * RS_MAX(1, tickSkip) : cfg.horizon; // VTS: 1-tick holds exist

	std::vector<int64_t> rowOff;   // per alive slot: first row in the batch
	for (int t = 0; t < maxSteps; t++) {
		aliveIdx.clear();
		for (int j : assigned)
			if (pool[(size_t)jobSlot[(size_t)j]]->alive)
				aliveIdx.push_back(jobSlot[(size_t)j]);
		if (aliveIdx.empty())
			break;
		rowOff.assign(aliveIdx.size() + 1, 0);
		for (size_t a = 0; a < aliveIdx.size(); a++)
			rowOff[a + 1] = rowOff[a] + pool[(size_t)aliveIdx[a]]->nCars;
		const int64_t rows = rowOff.back();
		obsBuf.resize((size_t)rows * obsSize);
		maskBuf.resize((size_t)rows * numActions);
		tempBuf.assign((size_t)rows, 1.f);
		if (intentOn) {
			intentFeatBuf.assign((size_t)rows * (size_t)(intentDim + 1), 0.f);
			intentIdBuf.assign((size_t)rows, 0);
		}
		for (size_t a = 0; a < aliveIdx.size(); a++) {
			Slot& s = *pool[(size_t)aliveIdx[a]];
			if (intentOn) {
				// collection semantics: a car whose interval expired draws a fresh uniform intent
				// (clock = 1 on the draw step); the self car of a kind-3 job takes the next
				// intent of its sequence while inside the prefix
				const Job& jb = jobs[(size_t)s.job];
				for (int p = 0; p < s.nCars; p++) {
					const bool dueP = vtsOn ? (s.holdLeft[(size_t)p] <= 0) : true;
					if (dueP && s.intentLeft[(size_t)p] <= 0) {
						int z = zRoll(rng);
						if (jb.kind == 3 && p == s.selfSlot) {
							const int pos = s.selfDecisions / intentPeriod;
							if (pos < (int)jb.intents.size()) z = jb.intents[(size_t)pos];
						}
						s.intentZ[(size_t)p] = RS_CLAMP(z, 0, intentDim - 1);
						s.intentLeft[(size_t)p] = intentPeriod;
					}
					float* f = &intentFeatBuf[(size_t)(rowOff[a] + p) * (size_t)(intentDim + 1)];
					f[s.intentZ[(size_t)p]] = 1.f;
					f[intentDim] = (float)s.intentLeft[(size_t)p] / (float)intentPeriod;
					intentIdBuf[(size_t)(rowOff[a] + p)] = s.intentZ[(size_t)p];
				}
			}
			std::copy(s.obs.begin(), s.obs.end(), obsBuf.begin() + (ptrdiff_t)(rowOff[a] * obsSize));
			std::copy(s.masks.begin(), s.masks.end(), maskBuf.begin() + (ptrdiff_t)(rowOff[a] * numActions));
			for (int p = 0; p < s.nCars; p++)
				s.due[(size_t)p] = vtsOn ? (s.holdLeft[(size_t)p] <= 0) : 1;
			const Job& job = jobs[(size_t)s.job];
			const bool selfInPrefix = job.kind == 3 ? (s.selfDecisions < job.L * intentPeriod)
				: vtsOn ? (s.selfDecisions < job.L) : (t < job.L * cfg.macro);
			if (job.kind == 2 && selfInPrefix && s.due[(size_t)s.selfSlot])
				tempBuf[(size_t)(rowOff[a] + s.selfSlot)] = job.temp;
		}

		auto tObs = torch::from_blob(obsBuf.data(), { rows, (int64_t)obsSize }, torch::kFloat32)
			.to(ppo->device, /*non_blocking=*/false, /*copy=*/true);
		auto tMasks = torch::from_blob(maskBuf.data(), { rows, (int64_t)numActions }, torch::kUInt8)
			.to(ppo->device, false, true);
		torch::Tensor tIntentFeat, tIntentIds;
		if (intentOn) {
			tIntentFeat = torch::from_blob(intentFeatBuf.data(), { rows, (int64_t)intentDim + 1 }, torch::kFloat32)
				.to(ppo->device, false, true);
			tIntentIds = torch::from_blob(intentIdBuf.data(), { rows }, torch::kInt64).to(ppo->device, false, true);
		}
		auto probs = PPOLearner::InferPolicyProbsFromModels(
			ppo->models, tObs, tMasks, ppo->config.policyTemperature, hp, {}, nullptr, {}, false,
			tIntentFeat, tIntentIds).to(torch::kFloat32);
		// per-row temperature on the self rows of temp-prefix jobs (probs^(1/T), renormalised)
		auto tTemp = torch::from_blob(tempBuf.data(), { rows }, torch::kFloat32).to(ppo->device, false, true);
		auto pT = probs.pow((1.f / tTemp).unsqueeze(1));
		pT = pT / pT.sum(1, true).clamp_min(1e-12f);
		auto sampled = torch::multinomial(pT, 1, true).flatten().to(torch::kCPU);
		const int64_t* sp = sampled.data_ptr<int64_t>();

		// decide actions for DUE cars (open-loop overrides on the self car), record, then step
		for (size_t a = 0; a < aliveIdx.size(); a++) {
			Slot& s = *pool[(size_t)aliveIdx[a]];
			Job& job = jobs[(size_t)s.job];
			for (int p = 0; p < s.nCars; p++)
				if (s.due[(size_t)p])
					s.curActions[(size_t)p] = (int)sp[rowOff[a] + p];
			if (!s.due[(size_t)s.selfSlot])
				continue;
			const int d = vtsOn ? s.selfDecisions : t;              // self decision index
			const int prefixPos = vtsOn ? d : d / cfg.macro;
			const bool inPrefix = job.kind == 3 ? (s.selfDecisions < job.L * intentPeriod)
				: vtsOn ? (d < job.L) : (t < job.L * cfg.macro);
			if (job.kind == 1 && inPrefix) {
				int act = job.actions[(size_t)prefixPos];
				const uint8_t* m = &s.masks[(size_t)s.selfSlot * numActions];
				if (!m[act]) {
					// invalid under the current mask: uniform among valid, like the study
					std::vector<int> valid;
					for (int k = 0; k < numActions; k++) if (m[k]) valid.push_back(k);
					act = valid.empty() ? act : valid[(size_t)(rng() % valid.size())];
				}
				s.curActions[(size_t)s.selfSlot] = act;
			}
			if (inPrefix && job.L > 0) {
				if (job.kind == 3) {
					if (s.selfDecisions % intentPeriod == 0)
						job.executed.push_back(s.intentZ[(size_t)s.selfSlot]);
				} else if (vtsOn || t % cfg.macro == 0)
					job.executed.push_back(s.curActions[(size_t)s.selfSlot]);
				if (job.record) {
					const float* o = &s.obs[(size_t)s.selfSlot * obsSize];
					job.recObs.insert(job.recObs.end(), o, o + obsSize);
					job.recAct.push_back((int32_t)s.curActions[(size_t)s.selfSlot]);
					const uint8_t* m = &s.masks[(size_t)s.selfSlot * numActions];
					job.recMask.insert(job.recMask.end(), m, m + numActions);
					if (intentOn) {
						const size_t r0 = (size_t)(rowOff[a] + s.selfSlot);
						job.recIntentId.push_back((int32_t)intentIdBuf[r0]);
						const float* f = &intentFeatBuf[r0 * (size_t)(intentDim + 1)];
						job.recIntentFeat.insert(job.recIntentFeat.end(), f, f + intentDim + 1);
					}
				}
			}
			s.selfDecisions++;
		}
		if (intentOn)   // the intent clock runs per decision, like collection
			for (size_t a = 0; a < aliveIdx.size(); a++) {
				Slot& s = *pool[(size_t)aliveIdx[a]];
				for (int p = 0; p < s.nCars; p++)
					if (s.due[(size_t)p]) s.intentLeft[(size_t)p] = RS_MAX(0, s.intentLeft[(size_t)p] - 1);
			}

		ParallelFor((int)aliveIdx.size(), [&](int a) {
			Slot& s = *pool[(size_t)aliveIdx[(size_t)a]];
			// EnvSet::StepFirstHalf / StepSecondHalf parity (VTS: the true-skip executor)
			s.prevGs = s.gs;
			s.gs.ResetBeforeStep();
			int ticks = tickSkip;
			if (vtsOn) {
#if GGL_DIP_HAS_VTS
				for (size_t p = 0; p < s.gs.players.size(); p++) {
					if (!s.due[p]) continue;
					RLGC::Action act = s.actionParser->ParseAction(s.curActions[p], s.gs.players[p], s.gs);
					s.arena->_cars[p]->controls = (CarControls)act;
					s.acts[p] = act;
					s.holdLeft[p] = VtsHoldTicks(s.curActions[p], vts);
				}
#endif
				ticks = RS_MAX(1, *std::min_element(s.holdLeft.begin(), s.holdLeft.end()));
				s.arena->Step(ticks);
				for (auto& h : s.holdLeft) h -= ticks;
			} else {
				s.arena->Step(actionDelay);
				for (size_t p = 0; p < s.gs.players.size(); p++) {
					RLGC::Action act = s.actionParser->ParseAction(s.curActions[p], s.gs.players[p], s.gs);
					s.arena->_cars[p]->controls = (CarControls)act;
					s.acts[p] = act;
				}
				s.arena->Step(tickSkip - actionDelay);
			}
			s.gs.UpdateFromArena(s.arena, s.acts, s.prevGs.IsEmpty() ? NULL : &s.prevGs);
			s.gs.userInfo = s.userInfo;
			s.stepTicks = ticks;

			uint8_t term = RLGC::TerminalType::NOT_TERMINAL;
			int goal = 0;
			if (s.gs.goalScored) {
				term = RLGC::TerminalType::NORMAL;
				goal = (s.selfTeam != RS_TEAM_FROM_Y(s.gs.ball.pos.y)) ? 1 : -1;
			}
			float r = 0;
			for (auto& w : s.rewards) {
				w.reward->PreStep(s.gs);
				FList out = w.reward->GetAllRewards(s.gs, term);
				r += out[(size_t)s.selfSlot] * w.weight;
			}
			r *= rewardScale;
			if (rewardClip > 0)
				r = RS_CLAMP(r, -rewardClip, rewardClip);
			s.stepReward = r;
			s.stepGoal = goal;
			if (goal == 0)
				BuildRows(this, obsSize, numActions, s.obsBuilder, s.actionParser, s.gs, s.obs, s.masks, normMean, normStd);
		});

		for (size_t a = 0; a < aliveIdx.size(); a++) {
			Slot& s = *pool[(size_t)aliveIdx[a]];
			s.score += s.discount * s.stepReward;
			// gamma is per tick under VTS (GAE discounts gamma^ticks), per row otherwise
			s.discount *= vtsOn ? std::pow(gamma, (float)s.stepTicks) : gamma;
			s.tickAcc += s.stepTicks;
			if (s.stepGoal != 0) {
				s.goal = s.stepGoal;
				s.alive = false;
			} else if (s.tickAcc >= horizonTicks) {
				s.alive = false;
				s.needLeaf = true;
			}
		}
	}
	// anything still alive hit maxSteps: bootstrap it too
	for (int j : assigned) {
		Slot& s = *pool[(size_t)jobSlot[(size_t)j]];
		if (s.alive) { s.alive = false; s.needLeaf = true; }
	}

	// leaf values for horizon survivors: V_real of the self player's obs
	aliveIdx.clear();
	for (int j : assigned)
		if (pool[(size_t)jobSlot[(size_t)j]]->needLeaf)
			aliveIdx.push_back(jobSlot[(size_t)j]);
	if (!aliveIdx.empty()) {
		obsBuf.resize(aliveIdx.size() * (size_t)obsSize);
		for (size_t a = 0; a < aliveIdx.size(); a++) {
			Slot& s = *pool[(size_t)aliveIdx[a]];
			std::copy(s.obs.begin() + (ptrdiff_t)((size_t)s.selfSlot * obsSize),
				s.obs.begin() + (ptrdiff_t)((size_t)(s.selfSlot + 1) * obsSize),
				obsBuf.begin() + (ptrdiff_t)(a * (size_t)obsSize));
		}
		auto tObs = torch::from_blob(obsBuf.data(), { (int64_t)aliveIdx.size(), (int64_t)obsSize }, torch::kFloat32)
			.to(ppo->device, false, true);
		auto v = ppo->InferCritic(tObs).to(torch::kCPU, torch::kFloat32);
		const float* vp = v.data_ptr<float>();
		for (size_t a = 0; a < aliveIdx.size(); a++)
			pool[(size_t)aliveIdx[a]]->score += pool[(size_t)aliveIdx[a]]->discount * vp[a];
	}

	for (int j : assigned) {
		Slot& s = *pool[(size_t)jobSlot[(size_t)j]];
		jobs[(size_t)j].score = s.score;
		jobs[(size_t)j].goal = s.goal;
		s.alive = false;
		s.needLeaf = false;
		s.job = -1;
	}
	lastWaves++;
}

// ---------------------------------------------------------------------------- driver
void DipSearch::Run(std::vector<DipSearchState>& states, PPOLearner* ppo,
	const std::vector<double>* normMean, const std::vector<double>* normStd,
	float rewardScale, float rewardClip, std::vector<DipSearchResult>& out, float budgetSecs) {

	Timer timer = {};
	lastWaves = 0;
	lastSkippedJobs = 0;
	out.assign(states.size(), DipSearchResult());
	const int perStateA = cfg.nBaseline + cfg.nRandom + cfg.nTemp;
	const int perStateB = cfg.nFinalists * cfg.kFinal;
	const int perStateC = cfg.kHeldout;
	const int G = RS_MAX(1, cfg.poolSize / RS_MAX(perStateA, RS_MAX(perStateB, perStateC)));
	std::mt19937 rng(1234567u + (unsigned)states.size());

	for (size_t g0 = 0; g0 < states.size(); g0 += (size_t)G) {
		if (timer.Elapsed() > budgetSecs)
			break;
		size_t g1 = RS_MIN(g0 + (size_t)G, states.size());

		// ---- wave A: baseline + candidates
		std::vector<Job> A;
		for (size_t si = g0; si < g1; si++) {
			for (int i = 0; i < cfg.nBaseline; i++) {
				Job j; j.state = (int)si; j.kind = 0; A.push_back(j);
			}
			const int Ls[3] = { 2, 4, 8 };
			const int iLs[3] = { 1, 2, 3 };   // intent sequences: 1-3 periods
			for (int i = 0; i < cfg.nRandom; i++) {
				Job j; j.state = (int)si;
				if (intentOn) {
					j.kind = 3; j.L = iLs[i % 3];
					for (int m = 0; m < j.L; m++) j.intents.push_back((int)(rng() % (unsigned)intentDim));
				} else {
					j.kind = 1; j.L = Ls[i % 3];
					for (int m = 0; m < j.L; m++) j.actions.push_back((int)(rng() % (unsigned)numActions));
				}
				A.push_back(j);
			}
			const float temps[3] = { 2.f, 4.f, 2.f };
			const int tL[3] = { 4, 4, 8 };
			for (int i = 0; i < cfg.nTemp; i++) {
				Job j; j.state = (int)si; j.kind = 2; j.L = tL[i % 3]; j.temp = temps[i % 3];
				A.push_back(j);
			}
		}
		RunWave(A, states, ppo, normMean, normStd, rewardScale, rewardClip);

		// per state: baseline stats, candidate ranking
		struct Cand { std::vector<int> prefix; float score; int kind; };   // kind 1 = actions, 3 = intents
		std::vector<std::vector<Cand>> finalists(g1 - g0);
		for (size_t si = g0; si < g1; si++) {
			DipSearchResult& r = out[si];
			std::vector<float> bs; int bg = 0, bc = 0;
			std::map<std::pair<int, std::vector<int>>, float> best;   // (kind, prefix) -> best single score
			for (auto& j : A) {
				if ((size_t)j.state != si || std::isnan(j.score)) continue;
				if (j.kind == 0) {
					bs.push_back(j.score); bg += j.goal == 1; bc += j.goal == -1;
				} else if (!j.executed.empty()) {
					auto key = std::make_pair(j.kind == 3 ? 3 : 1, j.executed);
					auto it = best.find(key);
					if (it == best.end() || j.score > it->second) best[key] = j.score;
				}
			}
			double m = 0; for (float x : bs) m += x; m /= RS_MAX(1, (int)bs.size());
			double v = 0; for (float x : bs) v += (x - m) * (x - m); v /= RS_MAX(1, (int)bs.size());
			r.searched = !bs.empty();   // no baseline = no slot of this car count in the pool
			r.baseMean = (float)m; r.baseStd = (float)std::sqrt(v);
			r.baseGoal = bg / (float)RS_MAX(1, (int)bs.size());
			r.baseConcede = bc / (float)RS_MAX(1, (int)bs.size());
			std::vector<Cand> cs;
			for (auto& kv : best) cs.push_back({ kv.first.second, kv.second, kv.first.first });
			std::sort(cs.begin(), cs.end(), [](const Cand& a, const Cand& b) { return a.score > b.score; });
			if ((int)cs.size() > cfg.nFinalists) cs.resize((size_t)cfg.nFinalists);
			finalists[si - g0] = cs;
		}

		// ---- wave B: finalists re-evaluated fresh
		std::vector<Job> B;
		for (size_t si = g0; si < g1; si++)
			for (size_t f = 0; f < finalists[si - g0].size(); f++)
				for (int i = 0; i < cfg.kFinal; i++) {
					const Cand& c = finalists[si - g0][f];
					Job j; j.state = (int)si; j.kind = c.kind; j.L = (int)c.prefix.size();
					if (c.kind == 3) j.intents = c.prefix; else j.actions = c.prefix;
					B.push_back(j);
				}
		if (!B.empty())
			RunWave(B, states, ppo, normMean, normStd, rewardScale, rewardClip);

		// pick the winner per state (policy itself is a finalist with its baseline mean)
		std::vector<std::vector<int>> winner(g1 - g0);
		std::vector<int> winnerKind(g1 - g0, 1);
		for (size_t si = g0; si < g1; si++) {
			auto& fs = finalists[si - g0];
			std::vector<double> sum(fs.size(), 0.0); std::vector<int> cnt(fs.size(), 0);
			for (auto& j : B) {
				if ((size_t)j.state != si || std::isnan(j.score)) continue;
				for (size_t f = 0; f < fs.size(); f++)
					if (fs[f].kind == j.kind && fs[f].prefix == (j.kind == 3 ? j.intents : j.actions)) { sum[f] += j.score; cnt[f]++; break; }
			}
			double bestMean = out[si].baseMean; int bestF = -1;
			for (size_t f = 0; f < fs.size(); f++) {
				double mf = cnt[f] ? sum[f] / cnt[f] : -1e9;
				if (mf > bestMean) { bestMean = mf; bestF = (int)f; }
			}
			if (bestF >= 0) { winner[si - g0] = fs[(size_t)bestF].prefix; winnerKind[si - g0] = fs[(size_t)bestF].kind; }
		}

		// ---- wave C: held-out re-evaluation of the winner (+ recorded imitation rollouts)
		std::vector<Job> C;
		for (size_t si = g0; si < g1; si++) {
			if (winner[si - g0].empty()) continue;
			for (int i = 0; i < cfg.kHeldout; i++) {
				Job j; j.state = (int)si; j.kind = winnerKind[si - g0]; j.L = (int)winner[si - g0].size();
				if (j.kind == 3) j.intents = winner[si - g0]; else j.actions = winner[si - g0];
				j.record = i < cfg.imitateRollouts; C.push_back(j);
			}
		}
		if (!C.empty())
			RunWave(C, states, ppo, normMean, normStd, rewardScale, rewardClip);
		for (size_t si = g0; si < g1; si++) {
			DipSearchResult& r = out[si];
			if (winner[si - g0].empty()) {
				r.bestMean = r.baseMean; r.gain = 0; r.bestL = 0; continue;
			}
			double m = 0; int n = 0, g = 0, c = 0;
			for (auto& j : C) {
				if ((size_t)j.state != si || std::isnan(j.score)) continue;
				m += j.score; n++; g += j.goal == 1; c += j.goal == -1;
			}
			r.bestMean = n ? (float)(m / n) : r.baseMean;
			r.bestGoal = g / (float)RS_MAX(1, n);
			r.bestConcede = c / (float)RS_MAX(1, n);
			r.gain = r.bestMean - r.baseMean;
			r.bestL = (int)winner[si - g0].size();
			r.prefix = winner[si - g0];
			r.intentPrefix = winnerKind[si - g0] == 3;
			r.accepted = r.gain >= cfg.minGain;
			if (r.accepted) {
				for (auto& j : C) {
					if ((size_t)j.state != si || !j.record) continue;
					r.obs.insert(r.obs.end(), j.recObs.begin(), j.recObs.end());
					r.actions.insert(r.actions.end(), j.recAct.begin(), j.recAct.end());
					r.masks.insert(r.masks.end(), j.recMask.begin(), j.recMask.end());
					r.intentIds.insert(r.intentIds.end(), j.recIntentId.begin(), j.recIntentId.end());
					r.intentFeat.insert(r.intentFeat.end(), j.recIntentFeat.begin(), j.recIntentFeat.end());
				}
				r.nRows = (int)r.actions.size();
			}
		}
	}
	lastWaveSecs = timer.Elapsed();
}
