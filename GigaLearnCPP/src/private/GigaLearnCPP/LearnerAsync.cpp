#include <GigaLearnCPP/Learner.h>
#include <RLGymCPP/TerminalConditions/TerminalCondition.h>
#include <GigaLearnCPP/PPO/PPOLearner.h>
#include <GigaLearnCPP/PPO/ExperienceBuffer.h>
#include <private/GigaLearnCPP/PPO/ExperienceBuffer.h>
#include <private/GigaLearnCPP/PolicyVersionManager.h>
#include <private/GigaLearnCPP/Async/PolicyPack.h>
#include <private/GigaLearnCPP/Async/WeightDoubleBuffer.h>
#include <private/GigaLearnCPP/Async/Fragment.h>
#include <private/GigaLearnCPP/Async/FragmentFifo.h>
#include <private/GigaLearnCPP/Async/LearnPrep.h>
#include <private/GigaLearnCPP/Util/WelfordStat.h>
#include <GigaLearnCPP/Util/Timer.h>

#ifdef RG_CUDA_SUPPORT
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <thread>
#include <vector>

using namespace RLGC;

void GGL::Learner::StartAsync() {
	dist->enable_async_routing();
	RG_LOG("Learner::StartAsync(): rank=" << DistRank()
		<< (dist->is_learner() ? " LEARNER" : " COLLECTOR")
		<< " nL=" << dist->n_learners() << " nC=" << dist->n_collectors()
		<< " games=" << config.numGames);

	std::atomic<bool> saveQueued = false;
	std::thread keyPressThread;
	StartQuitKeyThread(saveQueued, keyPressThread);

	const int T = config.asyncFragmentTicks > 0 ? config.asyncFragmentTicks : 390;
	const int nL = dist->n_learners();
	// Weak-scale default: 160k rows/learner (2-node = 800k). A fixed 800k global
	// caps fleet SPS once Learn is Optim/seqHalf-bound (10-node ≈ 30-node ~1.05M).
	int64_t globalBatch = config.asyncGlobalBatchSize;
	if (globalBatch <= 0)
		globalBatch = 160000LL * std::max(1, nL);
	const int64_t local_target = globalBatch / std::max(1, nL);
	const int slack = T;

	if (dist->is_collector()) {
		WeightDoubleBuffer wbuf;
		auto pair = ppo->GetPolicyModels();
		wbuf.Init(pair, dist->local_rank());
		wbuf.StartRecvThread(dist);

		envSet->Reset();
		const int numPlayers = envSet->state.numPlayers;
		RG_LOG("[DIST][COLLECT] ready rank=" << DistRank()
			<< " games=" << config.numGames << " players=" << numPlayers);
		struct Ring {
			std::vector<float> states, logProbs, rewards;
			std::vector<int32_t> actions;
			std::vector<int8_t> terminals;
			std::vector<uint8_t> masks;
			std::vector<uint32_t> policyVersions;
			size_t Length() const { return actions.size(); }
			void ErasePrefix(int n, int obsSize, int numActions) {
				if (n <= 0) return;
				states.erase(states.begin(), states.begin() + (size_t)n * obsSize);
				logProbs.erase(logProbs.begin(), logProbs.begin() + n);
				rewards.erase(rewards.begin(), rewards.begin() + n);
				actions.erase(actions.begin(), actions.begin() + n);
				terminals.erase(terminals.begin(), terminals.begin() + n);
				masks.erase(masks.begin(), masks.begin() + (size_t)n * numActions);
				policyVersions.erase(policyVersions.begin(), policyVersions.begin() + n);
			}
			void Clear() {
				states.clear(); logProbs.clear(); rewards.clear(); actions.clear();
				terminals.clear(); masks.clear(); policyVersions.clear();
			}
		};
		std::vector<Ring> rings(numPlayers);
		uint64_t fragment_id = 0;
		uint64_t publishedVersion = 0;
		uint64_t ticks = 0;
		uint64_t nTermSends = 0, nTruncSends = 0;
		auto lastBeat = std::chrono::steady_clock::now();
		std::vector<int> curActions(numPlayers);
		std::vector<float> newLogProbs(numPlayers);
		bool exiting = false;
		bool gotWeights = false;

		auto recvExitIfAny = [&]() {
			size_t n = 0;
			if (!dist->iprobe_host(Dist::Session::kAnySource, Dist::Session::TAG_EXIT, &n))
				return false;
			int dummy = 0;
			dist->recv_host(&dummy, sizeof(dummy), Dist::Session::kAnySource, Dist::Session::TAG_EXIT);
			exiting = true;
			return true;
		};

		auto tryPublish = [&]() {
			if (!wbuf.TryPublish())
				return false;
			publishedVersion = wbuf.pending_version.load();
			uint64_t seq = wbuf.pending_archive_seq.load();
			if (seq != 0)
				wbuf.pending_archive_seq.store(0);
			return true;
		};

		struct PendingSend {
			std::vector<uint8_t> blob;
			Dist::Session::CollectiveRequest req{};
		};
		std::deque<PendingSend> pendingSends;
		constexpr int kMaxPendingSends = 32;
		auto progressSends = [&]() {
			while (!pendingSends.empty() && dist->test_host(&pendingSends.front().req))
				pendingSends.pop_front();
		};

		auto tryHandleGo = [&]() {
			size_t n = 0;
			if (!dist->iprobe_host(Dist::Session::kAnySource, Dist::Session::TAG_WEIGHT_GO, &n))
				return;
			int go = 0;
			dist->recv_host(&go, sizeof(go), Dist::Session::kAnySource, Dist::Session::TAG_WEIGHT_GO);
			// Kick NCCL on the recv thread and keep InferActions on this thread. Waiting
			// here for the previous bcast parked collect in the mid-traj gaps (~118→75
			// ticks/s). First weights still wait so we never roll with an empty pack.
			wbuf.RequestBcast();
			if (!gotWeights) {
				wbuf.WaitBcastDone();
				tryPublish();
				gotWeights = true;
			}
		};

		auto fnParallelFor = [](int n, const std::function<void(int)>& body) {
			if (n <= 0)
				return;
			int nThreads = RS_MAX(1, RLGC::g_ThreadPool.GetNumThreads());
			int numChunks = RS_MIN(n, nThreads);
			int chunkSize = (n + numChunks - 1) / numChunks;
			RLGC::g_ThreadPool.StartBatchedJobs(
				[&body, n, chunkSize](int chunk) {
					int start = chunk * chunkSize;
					int stop = RS_MIN(start + chunkSize, n);
					for (int i = start; i < stop; i++)
						body(i);
				},
				numChunks, false
			);
		};

		auto sendRingPrefix = [&](Ring& ring, int nSend, bool forceLastTrunc, const float* nextObs) {
			if (nSend <= 0) return;
			if (recvExitIfAny())
				return;
			progressSends();
			while ((int)pendingSends.size() >= kMaxPendingSends) {
				if (recvExitIfAny())
					return;
				tryHandleGo();
				progressSends();
				std::this_thread::yield();
			}
			TrajectoryFragment frag;
			frag.hdr.collector_world = DistRank();
			frag.hdr.T = nSend;
			frag.hdr.obsSize = obsSize;
			frag.hdr.numActions = numActions;
			const bool hasNext = nextObs != nullptr;
			frag.hdr.nTrunc = hasNext ? 1 : 0;
			frag.hdr.fragment_id = fragment_id;
			frag.hdr.env_steps = (uint64_t)nSend;
			uint32_t mn = UINT32_MAX, mx = 0;
			for (int i = 0; i < nSend; i++) {
				mn = std::min(mn, ring.policyVersions[i]);
				mx = std::max(mx, ring.policyVersions[i]);
			}
			frag.hdr.min_policy_version = mn;
			frag.hdr.max_policy_version = mx;
			// Pack memcpy's immediately; do not clone. forceLastTrunc mutates a prefix
			// row that ErasePrefix discards after this send.
			frag.states = torch::from_blob(ring.states.data(), { nSend, obsSize }, torch::kFloat32);
			frag.actions = torch::from_blob(ring.actions.data(), { nSend }, torch::kInt32);
			frag.logProbs = torch::from_blob(ring.logProbs.data(), { nSend }, torch::kFloat32);
			frag.rewards = torch::from_blob(ring.rewards.data(), { nSend }, torch::kFloat32);
			frag.terminals = torch::from_blob(ring.terminals.data(), { nSend }, torch::kInt8);
			frag.actionMasks = torch::from_blob(ring.masks.data(), { nSend, numActions }, torch::kUInt8);
			frag.policy_version = torch::from_blob(ring.policyVersions.data(), { nSend }, torch::kInt32);
			if (forceLastTrunc)
				frag.terminals[nSend - 1] = (int8_t)RLGC::TerminalType::TRUNCATED;
			if (hasNext)
				frag.nextStates = torch::from_blob(const_cast<float*>(nextObs), { 1, obsSize }, torch::kFloat32);
			else
				frag.nextStates = torch::empty({ 0, obsSize }, torch::kFloat32);
			pendingSends.emplace_back();
			pendingSends.back().blob = frag.Pack();
			int dest = dist->dest_learner_world(fragment_id);
			dist->isend_host(pendingSends.back().blob.data(), pendingSends.back().blob.size(),
				dest, Dist::Session::TAG_FRAGMENT, &pendingSends.back().req);
			fragment_id++;
			if ((fragment_id & 63) == 0)
				RG_LOG("[DIST][COLLECT] rank=" << DistRank() << " T=" << nSend << " dest=" << dest
					<< " ver=" << publishedVersion << " inflight=" << pendingSends.size());
		};

		torch::Tensor cpu_states, cpu_masks;
		std::vector<float> obsCopy, postObs;
		std::vector<uint8_t> maskCopy;
		auto ensureCpu2d = [](torch::Tensor& buf, int64_t rows, int64_t cols, torch::ScalarType dt) {
			if (!buf.defined() || buf.size(0) < rows || buf.size(1) != cols)
				buf = torch::empty({ rows, cols }, torch::TensorOptions().dtype(dt));
			return buf.narrow(0, 0, rows);
		};

		while (!exiting) {
			if (recvExitIfAny())
				break;
			progressSends();
			tryHandleGo();
			if (config.asyncMidTrajPull)
				tryPublish();

			auto now = std::chrono::steady_clock::now();
			if (now - lastBeat > std::chrono::seconds(2)) {
				lastBeat = now;
				RG_LOG("[DIST][COLLECT] rank=" << DistRank()
					<< " ticks=" << ticks
					<< " gotW=" << gotWeights
					<< " termSend=" << nTermSends
					<< " truncSend=" << nTruncSends
					<< " ver=" << publishedVersion);
			}
			if (!gotWeights) {
				std::this_thread::yield();
				continue;
			}

			ticks++;
			envSet->Reset();

			torch::Tensor tStates_view = torch::from_blob(
				envSet->state.obs.data.data(),
				{ (int64_t)envSet->state.obs.size[0], (int64_t)envSet->state.obs.size[1] },
				torch::kFloat32);
			torch::Tensor tMasks_view = torch::from_blob(
				envSet->state.actionMasks.data.data(),
				{ (int64_t)envSet->state.actionMasks.size[0], (int64_t)envSet->state.actionMasks.size[1] },
				torch::kUInt8);
			const int64_t nrows = tStates_view.size(0);
			auto tStates_cpu = ensureCpu2d(cpu_states, nrows, obsSize, torch::kFloat32);
			auto tMasks_cpu = ensureCpu2d(cpu_masks, nrows, numActions, torch::kUInt8);
			tStates_cpu.copy_(tStates_view);
			tMasks_cpu.copy_(tMasks_view);
			torch::Tensor tdStates = tStates_cpu.to(ppo->device, /*non_blocking=*/true);
			torch::Tensor tdMasks = tMasks_cpu.to(ppo->device, /*non_blocking=*/true);

			envSet->StepFirstHalf(true);

			torch::Tensor tActions, tLogProbs;
			ppo->InferActions(tdStates, tdMasks, &tActions, &tLogProbs, wbuf.InferModels());
			envSet->Sync();
			auto hostActs = tActions.to(torch::kCPU, torch::kInt64);
			auto hostLog = tLogProbs.to(torch::kCPU, torch::kFloat32);
			curActions.resize((size_t)nrows);
			newLogProbs.resize((size_t)nrows);
			{
				const int64_t* p = hostActs.data_ptr<int64_t>();
				for (int64_t i = 0; i < nrows; i++)
					curActions[(size_t)i] = (int)p[i];
			}
			std::memcpy(newLogProbs.data(), hostLog.data_ptr<float>(), newLogProbs.size() * sizeof(float));

			obsCopy.resize((size_t)tStates_cpu.numel());
			std::memcpy(obsCopy.data(), tStates_cpu.data_ptr<float>(), obsCopy.size() * sizeof(float));
			maskCopy.resize((size_t)tMasks_cpu.numel());
			std::memcpy(maskCopy.data(), tMasks_cpu.data_ptr<uint8_t>(), maskCopy.size());

			envSet->StepSecondHalf(curActions, false);

			postObs.resize(obsCopy.size());
			std::memcpy(postObs.data(), envSet->state.obs.data.data(), postObs.size() * sizeof(float));

			std::vector<uint8_t> curTerminals(numPlayers, 0);
			for (int idx = 0; idx < (int)envSet->arenas.size(); idx++) {
				uint8_t tt = envSet->state.terminals[idx];
				if (!tt) continue;
				auto start = envSet->state.arenaPlayerStartIdx[idx];
				int np = (int)envSet->state.gameStates[idx].players.size();
				for (int i = 0; i < np; i++)
					curTerminals[start + i] = tt;
			}

			fnParallelFor(numPlayers, [&](int p) {
				auto& ring = rings[p];
				ring.states.insert(ring.states.end(), obsCopy.begin() + (size_t)p * obsSize,
					obsCopy.begin() + (size_t)(p + 1) * obsSize);
				ring.masks.insert(ring.masks.end(), maskCopy.begin() + (size_t)p * numActions,
					maskCopy.begin() + (size_t)(p + 1) * numActions);
				ring.actions.push_back((int32_t)curActions[p]);
				ring.logProbs.push_back(newLogProbs[p]);
				float rew = 0.f;
				if (!envSet->state.rewards.empty() && p < (int)envSet->state.rewards.size())
					rew = envSet->state.rewards[p];
				ring.rewards.push_back(rew);
				ring.terminals.push_back((int8_t)curTerminals[p]);
				ring.policyVersions.push_back((uint32_t)publishedVersion);
			});
			for (int p = 0; p < numPlayers; p++) {
				if (exiting)
					break;
				auto& ring = rings[p];
				int8_t term = (int8_t)curTerminals[p];
				int L = (int)ring.Length();
				if (term) {
					if (L >= 1) {
						const float* nextPtr = nullptr;
						if (term == (int8_t)RLGC::TerminalType::TRUNCATED)
							nextPtr = postObs.data() + (size_t)p * obsSize;
						sendRingPrefix(ring, L, false, nextPtr);
						nTermSends++;
					}
					ring.Clear();
				} else if (L >= T + 1) {
					sendRingPrefix(ring, T, true, ring.states.data() + (size_t)T * obsSize);
					nTruncSends++;
					ring.ErasePrefix(T, obsSize, numActions);
				}
			}
		}

		while (!pendingSends.empty()) {
			progressSends();
			tryHandleGo();
			std::this_thread::yield();
		}
		int ack = 1;
		Dist::Session::CollectiveRequest ackReq{};
		dist->isend_host(&ack, sizeof(ack), 0, Dist::Session::TAG_EXIT_ACK, &ackReq);
		while (!dist->test_host(&ackReq))
			std::this_thread::yield();
		wbuf.RequestBcast();
		wbuf.WaitBcastDone();
		wbuf.JoinRecvThread();
		wbuf.Shutdown();
		dist->barrier_world();
		return;
	}

	// ---- learners ----
	FragmentFifo fifo;
	fifo.fifoMaxRows = 3 * local_target;
	fifo.maxPolicyLag = config.asyncMaxPolicyLag > 0 ? config.asyncMaxPolicyLag : 16;

	PolicyPack pack;
	pack.InitFrom(ppo->models);
	torch::Tensor packStaging = torch::empty({ pack.totalFloats },
		torch::TensorOptions().dtype(torch::kFloat32).device(ppo->device));
#ifdef RG_CUDA_SUPPORT
	cudaStream_t weightBcastStream = nullptr;
	cudaEvent_t bcastEv = nullptr;
	cudaEvent_t packEv = nullptr;
	bool bcastOutstanding = false;
	if (ppo->device.is_cuda()) {
		c10::cuda::CUDAGuard dg(ppo->device.index());
		cudaStreamCreateWithFlags(&weightBcastStream, cudaStreamNonBlocking);
		cudaEventCreateWithFlags(&bcastEv, cudaEventDisableTiming);
		cudaEventCreateWithFlags(&packEv, cudaEventDisableTiming);
	}
#endif
	uint64_t pending_archive_seq = 0;
	int64_t collected_this_cycle = 0;
	int64_t total_collected = 0, total_consumed = 0;
	int nexto_for_learn = 0, nexto_against_learn = 0;
	double fragRecvTime = 0;
	double weightBcastTime = 0;
	double bcastWaitTime = 0;

	ExperienceBuffer experience(config.randomSeed + DistRank() + 1, ppo->device);

	auto DrainFragments = [&]() {
		size_t nbytes = 0;
		int src = Dist::Session::kAnySource;
		Timer recvTimer = {};
		int nRecv = 0;
		while (nRecv < 64 && dist->iprobe_host(Dist::Session::kAnySource, Dist::Session::TAG_FRAGMENT, &nbytes, &src)) {
			std::vector<uint8_t> blob(nbytes);
			dist->recv_host(blob.data(), nbytes, src, Dist::Session::TAG_FRAGMENT);
			auto frag = TrajectoryFragment::Unpack(blob.data(), blob.size());
			collected_this_cycle += (int64_t)frag.hdr.env_steps;
			nexto_for_learn += frag.hdr.nexto_goals_for;
			nexto_against_learn += frag.hdr.nexto_goals_against;
			fifo.enqueue(std::move(frag));
			nRecv++;
		}
		if (nRecv > 0)
			fragRecvTime += recvTimer.Elapsed();
	};

	auto rejoinLearners = [&]() {
		Dist::Session::CollectiveRequest req{};
		dist->ibarrier_learners(&req);
		while (!dist->test_request(&req))
			DrainFragments();
	};

	auto waitBcastDone = [&]() {
#ifdef RG_CUDA_SUPPORT
		if (!bcastOutstanding || !bcastEv)
			return;
		Timer wt = {};
		while (cudaEventQuery(bcastEv) == cudaErrorNotReady)
			DrainFragments();
		cudaEventSynchronize(bcastEv);
		bcastOutstanding = false;
		bcastWaitTime += wt.Elapsed();
#endif
	};

	auto drainWhileStream = [&]() {
#ifdef RG_CUDA_SUPPORT
		if (ppo->device.is_cuda()) {
			cudaEvent_t ev = nullptr;
			cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
			cudaEventRecord(ev, c10::cuda::getCurrentCUDAStream(ppo->device.index()).stream());
			while (cudaEventQuery(ev) == cudaErrorNotReady)
				DrainFragments();
			cudaEventSynchronize(ev);
			cudaEventDestroy(ev);
			return;
		}
#endif
		DrainFragments();
	};

	auto bcastPackedWeights = [&](uint64_t version, uint64_t seq) {
		Timer bt = {};
		waitBcastDone();
		std::vector<int> ones;
		std::vector<Dist::Session::CollectiveRequest> reqs;
		if (DistRank() == 0 && dist->n_collectors() > 0) {
			ones.assign((size_t)dist->n_collectors(), 1);
			reqs.resize((size_t)dist->n_collectors());
			for (int c = 0; c < dist->n_collectors(); c++)
				dist->isend_host(&ones[(size_t)c], sizeof(int), dist->n_learners() + c,
					Dist::Session::TAG_WEIGHT_GO, &reqs[(size_t)c]);
		}
		DrainFragments();
		pack.Pack(ppo->models, packStaging, version, seq);
		for (;;) {
			bool all = true;
			for (auto& r : reqs) {
				if (!dist->test_host(&r))
					all = false;
			}
			if (all)
				break;
			DrainFragments();
		}
#ifdef RG_CUDA_SUPPORT
		cudaStream_t def = ppo->device.is_cuda()
			? (cudaStream_t)c10::cuda::getCurrentCUDAStream(ppo->device.index()).stream()
			: nullptr;
		if (weightBcastStream && packEv && bcastEv && DistRank() == 0
			&& dist->n_collectors() > 0 && def) {
			cudaEventRecord(packEv, def);
			cudaStreamWaitEvent(weightBcastStream, packEv, 0);
			dist->bcast_weights(packStaging.data_ptr<float>(), (size_t)packStaging.numel(),
				(Dist::Session::Stream)weightBcastStream, /*wait=*/false);
			cudaEventRecord(bcastEv, weightBcastStream);
			bcastOutstanding = true;
		} else {
			Dist::Session::Stream wt = (Dist::Session::Stream)def;
			dist->bcast_weights(packStaging.data_ptr<float>(), (size_t)packStaging.numel(), wt, /*wait=*/false);
			drainWhileStream();
		}
#else
		dist->bcast_weights(packStaging.data_ptr<float>(), (size_t)packStaging.numel(), nullptr, /*wait=*/false);
		drainWhileStream();
#endif
		weightBcastTime += bt.Elapsed();
	};

	Report pendingDisplay;
	bool havePendingDisplay = false;
	auto flushDisplay = [&]() {
		if (!havePendingDisplay || DistRank() != 0)
			return;
		havePendingDisplay = false;
		pendingDisplay.Finish();
		if (metricSender)
			metricSender->Send(pendingDisplay);
		pendingDisplay.Display({
			"Average Step Reward",
			"Policy Entropy",
			"Mean KL Divergence",
			"Rating/1v1",
			"Rating/2v2",
			"Rating/3v3",
			"Ref/Share",
			"Ref/Oldest Share",
			"",
			"Async/VersionDiff Min",
			"Async/VersionDiff Avg",
			"Async/VersionDiff Max",
			"Async/FreshFrac",
			"Async/MuChangeFrac",
			"Async/MidTraj Pulls Avg",
			"Async/MidTraj Pulls Max",
			"Async/MidTraj Span Avg",
			"Async/FIFO Rows",
			"Async/Drop Stale",
			"Async/Drop Overflow",
			"Async/Drop Rate",
			"Async/Collected Steps",
			"Async/Consumed Steps",
			"Async/Total Collected Steps",
			"Async/Total Consumed Steps",
			"Async/N Used",
			"Async/MinCut Slack",
			"Async/Collector SPS",
			"Async/Learner SPS",
			"Async/Weight Bcast Time",
			"Async/Bcast Wait Time",
			"Async/Epoch Hook Time",
			"Async/Minibatch Drain Time",
			"Async/Frag Recv Time",
			"Async/Policy Version",
			"Async/Ready Wait Time",
			"Async/Cycle Wall",
			"Async/Unaccounted Time",
			"Async/FIFO Depth",
			"PPO/AllReduce Time",
			"PPO/Clip Time",
			"PPO/Optim Time",
			"PPO/StepOk Min Time",
			"PPO/Upload Time",
			"Async/Prep Other Time",
			"Plasticity Time",
			"GPU/Allocated MB",
			"GPU/Reserved MB",
			"",
			"Collection Steps/Second",
			"Consumption Steps/Second",
			"Overall Steps/Second",
			"",
			"PPO Learn Time",
			"Value Pred Time",
			"GAE Time",
			"Prep Time",
			"Collected Timesteps",
			"Total Timesteps",
			"Total Iterations"
		});
	};
	double epochHookTime = 0;
	double minibatchDrainTime = 0;
	auto drainOnly = [&]() { DrainFragments(); };

	ppo->onMinibatchEnd = [&]() {
		Timer dt = {};
		DrainFragments();
		minibatchDrainTime += dt.Elapsed();
	};
	ppo->onBeforeStepOptims = [&]() { waitBcastDone(); };
	int epochCounter = 0;
	ppo->onEpochEnd = [&](bool stepOk) {
		Timer ht = {};
		epochCounter++;
		if (stepOk)
			ppo->policyVersion++;
		fifo.learner_version = ppo->policyVersion;
		const bool lastEpoch = epochCounter >= std::max(1, config.ppo.epochs);
		const bool doBcast = config.asyncMidTrajPull || lastEpoch;
		if (doBcast && DistRank() == 0 && dist->n_collectors() > 0) {
			uint64_t seq = stepOk ? pending_archive_seq : 0;
			bcastPackedWeights(ppo->policyVersion, seq);
			if (stepOk)
				pending_archive_seq = 0;
		}
		epochHookTime += ht.Elapsed();
	};

	if (DistRank() == 0 && dist->n_collectors() > 0) {
		bcastPackedWeights(ppo->policyVersion, 0);
		RG_LOG("[DIST] initial weights kicked floats=" << pack.totalFloats);
	}
	waitBcastDone();
	rejoinLearners();

	bool isFirst = true;
	auto lastHb = std::chrono::steady_clock::now();
	for (;;) {
		fragRecvTime = 0;
		weightBcastTime = 0;
		bcastWaitTime = 0;
		epochHookTime = 0;
		minibatchDrainTime = 0;
		Timer cycleTimer = {};
		Timer readyTimer = {};
		double readyWait = 0;
		for (;;) {
			DrainFragments();
			flushDisplay();
			int local_rows = (int)fifo.rows();
			int min_rows = local_rows;
			Dist::Session::CollectiveRequest minReq{};
			dist->imin_host(&min_rows, 1, &minReq);
			while (!dist->test_request(&minReq))
				DrainFragments();
			if (DistRank() == 0) {
				auto now = std::chrono::steady_clock::now();
				if (now - lastHb > std::chrono::seconds(2)) {
					lastHb = now;
					RG_LOG("[DIST][LEARN] wait fifo=" << local_rows
						<< " min=" << min_rows
						<< " target=" << local_target
						<< " collected_cycle=" << collected_this_cycle);
				}
			}
			if (min_rows >= (int)local_target - slack)
				break;
			std::this_thread::yield();
		}
		readyWait = readyTimer.Elapsed();

		int local_rows = (int)fifo.rows();
		int min_rows = local_rows;
		{
			Dist::Session::CollectiveRequest minReq{};
			dist->imin_host(&min_rows, 1, &minReq);
			while (!dist->test_request(&minReq))
				DrainFragments();
		}
		int n_used = std::min(min_rows, (int)local_target);
		auto frags = fifo.pop_for_learn(n_used);
		n_used = 0;
		for (auto& f : frags)
			n_used += (int)f.rows();
		const int ingestedFrags = (int)frags.size();
		const int dropStale = fifo.drop_stale;
		const int dropOverflow = fifo.drop_overflow;

		Report report;
		experience.data = {};
		Timer prepTimer = {};
		PrepareExperienceBatched(
			ppo, config, frags, experience.data, report, returnStat, dist, obsSize, drainOnly);
		const float prepS = prepTimer.Elapsed();
		report["Prep Time"] = prepS;

		report["Async/N Used"] = (float)n_used;
		report["Async/FIFO Rows"] = (float)fifo.rows();
		report["Async/FIFO Depth"] = (float)fifo.fragment_count();
		report["Async/Drop Stale"] = (float)dropStale;
		report["Async/Drop Overflow"] = (float)dropOverflow;
		{
			int den = dropStale + dropOverflow + ingestedFrags;
			report["Async/Drop Rate"] = den > 0 ? (float)(dropStale + dropOverflow) / (float)den : 0.f;
		}
		report["Async/Policy Version"] = (float)ppo->policyVersion;
		report["Async/MinCut Slack"] = (float)(local_target - n_used);
		report["Async/Ready Wait Time"] = (float)readyWait;
		report["Async/Local Target"] = (float)local_target;

		ppo->config.batchSize = n_used;
		epochCounter = 0;
		Timer learnTimer = {};
		ppo->Learn(experience, report, isFirst);
		isFirst = false;
		const float learnS = learnTimer.Elapsed();
		report["PPO Learn Time"] = learnS;

		int64_t consumed = (int64_t)n_used * nL;
		int64_t collected = collected_this_cycle;
		dist->sum_host(&collected, 1);
		int64_t prevTimesteps = (int64_t)totalTimesteps;
		total_collected += collected;
		total_consumed += consumed;
		totalTimesteps = (uint64_t)total_consumed;
		totalIterations++;
		report["Async/Collected Steps"] = (float)collected;
		report["Async/Consumed Steps"] = (float)consumed;
		report["Async/Ingested Rows"] = (float)consumed;
		report["Async/Total Collected Steps"] = (float)total_collected;
		report["Async/Total Consumed Steps"] = (float)total_consumed;
		report["Async/Collector Env Steps"] = (float)collected;
		report["Collected Timesteps"] = (float)collected;
		report["Consumed Timesteps"] = (float)consumed;
		report["Total Timesteps"] = (float)totalTimesteps;
		report["Total Iterations"] = (float)totalIterations;
		int nf = nexto_for_learn, na = nexto_against_learn;
		dist->sum_host(&nf, 1);
		dist->sum_host(&na, 1);
		nextoGoalsFor += nf;
		nextoGoalsAgainst += na;
		nexto_for_learn = nexto_against_learn = 0;
		collected_this_cycle = 0;
		fifo.drop_stale = fifo.drop_overflow = 0;

		double cycleWall = cycleTimer.Elapsed();
		double consumeWall = (double)prepS + (double)learnS;
		if (cycleWall > 1e-9) {
			report["Async/Collector SPS"] = (float)((double)collected / cycleWall);
			report["Collection Steps/Second"] = report["Async/Collector SPS"];
			report["Overall Steps/Second"] = (float)((double)consumed / cycleWall);
		}
		if (consumeWall > 1e-9) {
			report["Async/Learner SPS"] = (float)((double)consumed / consumeWall);
			report["Consumption Steps/Second"] = report["Async/Learner SPS"];
		}
		report["Async/Weight Bcast Time"] = (float)weightBcastTime;
		report["Async/Bcast Wait Time"] = (float)bcastWaitTime;
		report["Async/Epoch Hook Time"] = (float)epochHookTime;
		report["Async/Minibatch Drain Time"] = (float)minibatchDrainTime;
		report["Async/Frag Recv Time"] = (float)fragRecvTime;
		report["Async/Cycle Wall"] = (float)cycleWall;
		report["Async/Unaccounted Time"] = (float)(cycleWall - readyWait - consumeWall);
		{
			float vp = report["Value Pred Time"];
			float gae = report["GAE Time"];
			report["Async/Prep Other Time"] = prepS - vp - gae;
		}
		report["Consumption Time"] = (float)consumeWall;
		report["Collection Time"] = (float)readyWait;
#ifdef RG_CUDA_SUPPORT
		if (ppo->device.is_cuda()) {
			const int idx = ppo->device.index();
			auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(idx);
			constexpr size_t agg = 0;
			report["GPU/Allocated MB"] = (float)(stats.allocated_bytes[agg].current / (1024 * 1024));
			report["GPU/Reserved MB"] = (float)(stats.reserved_bytes[agg].current / (1024 * 1024));
		}
#endif

		if (!config.checkpointFolder.empty()
			&& totalIterations % (uint64_t)std::max(1, (int)config.iterPerSave) == 0)
			Save();
		if (DistRank() == 0 && versionMgr) {
			versionMgr->OnIteration(ppo, report, (int64_t)totalTimesteps, prevTimesteps);
			if ((totalTimesteps / versionMgr->tsPerVersion > (uint64_t)prevTimesteps / versionMgr->tsPerVersion)
				|| prevTimesteps == 0)
				pending_archive_seq = totalTimesteps;
		}
		if (DistRank() == 0) {
			RG_LOG("ASYNC iter=" << totalIterations
				<< " n_used=" << n_used
				<< " collected=" << collected
				<< " consumed=" << consumed
				<< " fifo=" << fifo.rows()
				<< " ver=" << ppo->policyVersion
				<< " vdAvg=" << report["Async/VersionDiff Avg"]
				<< " fresh=" << report["Async/FreshFrac"]
				<< " learn_s=" << report["PPO Learn Time"]
				<< " hook_s=" << epochHookTime
				<< " ar_s=" << report["PPO/AllReduce Time"]
				<< " clip_s=" << report["PPO/Clip Time"]
				<< " drain_s=" << minibatchDrainTime);
			pendingDisplay = report;
			havePendingDisplay = true;
		}
		rejoinLearners();

		int exitFlag = (saveQueued || exitRequested) ? 1 : 0;
		dist->max_host(&exitFlag, 1);
		if (exitFlag) {
			flushDisplay();
			if (DistRank() == 0) {
				std::vector<int> ones((size_t)dist->n_collectors(), 1);
				std::vector<Dist::Session::CollectiveRequest> exitReqs((size_t)dist->n_collectors());
				for (int c = 0; c < dist->n_collectors(); c++)
					dist->isend_host(&ones[(size_t)c], sizeof(int), dist->n_learners() + c, Dist::Session::TAG_EXIT, &exitReqs[(size_t)c]);
				for (;;) {
					bool all = true;
					for (auto& r : exitReqs) {
						if (!dist->test_host(&r))
							all = false;
					}
					if (all)
						break;
					DrainFragments();
				}
				int nAck = 0;
				while (nAck < dist->n_collectors()) {
					DrainFragments();
					size_t nb = 0;
					if (dist->iprobe_host(Dist::Session::kAnySource, Dist::Session::TAG_EXIT_ACK, &nb)) {
						int ackv = 0;
						dist->recv_host(&ackv, sizeof(ackv), Dist::Session::kAnySource, Dist::Session::TAG_EXIT_ACK);
						nAck++;
					}
				}
			}
			rejoinLearners();
			int had = 0;
			do {
				had = dist->iprobe_host(Dist::Session::kAnySource, Dist::Session::TAG_FRAGMENT, nullptr) ? 1 : 0;
				if (had)
					DrainFragments();
				dist->min_host(&had, 1);
			} while (had);
			if (DistRank() == 0 && dist->n_collectors() > 0) {
				pack.Pack(ppo->models, packStaging, UINT64_MAX, 0);
				dist->bcast_weights(packStaging.data_ptr<float>(), (size_t)packStaging.numel(), nullptr);
			}
			rejoinLearners();
			if (!config.checkpointFolder.empty())
				Save();
			dist->barrier_world();
			return;
		}
	}
}
