#pragma once
#include "PolicyPack.h"
#include <GigaLearnCPP/Distributed/Session.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>

namespace GGL {

class WeightDoubleBuffer {
public:
	ModelSet buf[2];
	torch::Tensor packStaging;
	PolicyPack pack;
	Dist::Session::Stream weightStream = nullptr;
#ifdef RG_CUDA_SUPPORT
	cudaEvent_t readyEvent = nullptr;
	cudaStream_t rawStream = nullptr;
#endif
	int deviceIndex = 0;

	std::atomic<int> read_idx{ 0 };
	std::atomic<int> pending_idx{ -1 };
	std::atomic<uint64_t> pending_version{ 0 };
	std::atomic<uint64_t> pending_archive_seq{ 0 };
	std::atomic<int> recv_busy{ 0 };
	std::atomic<int> stop{ 0 };
	std::atomic<int> nccl_go{ 0 };
	std::atomic<uint64_t> bcast_issued{ 0 };
	std::atomic<uint64_t> bcast_done{ 0 };
	std::mutex mu;
	std::condition_variable cv;

	void Init(ModelSet& livePolicyPair, int deviceIndex);
	void Shutdown();

	ModelSet* InferModels() { return &buf[read_idx.load()]; }

	bool TryPublish();
	void RecvLoop(Dist::Session* dist);
	void RequestBcast();
	void WaitBcastDone();
	void RequestStop();

	std::thread recvThread;
	void StartRecvThread(Dist::Session* dist);
	void JoinRecvThread();
};

}
