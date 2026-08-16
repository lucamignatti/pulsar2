#include "WeightDoubleBuffer.h"

#ifdef RG_CUDA_SUPPORT
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#endif

namespace GGL {

void WeightDoubleBuffer::Init(ModelSet& livePolicyPair, int deviceIndex_) {
	deviceIndex = deviceIndex_;
	buf[0] = PolicyPack::ClonePolicyPair(livePolicyPair);
	buf[1] = PolicyPack::ClonePolicyPair(livePolicyPair);
	pack.InitFrom(livePolicyPair);
	packStaging = torch::empty({ pack.totalFloats },
		torch::TensorOptions().dtype(torch::kFloat32).device(torch::Device(torch::kCUDA, deviceIndex)));
#ifdef RG_CUDA_SUPPORT
	cudaStreamCreateWithFlags(&rawStream, cudaStreamNonBlocking);
	weightStream = rawStream;
	cudaEventCreateWithFlags(&readyEvent, cudaEventDisableTiming);
#endif
	read_idx = 0;
	pending_idx = -1;
	recv_busy = 0;
	stop = 0;
	nccl_go = 0;
	bcast_issued = 0;
	bcast_done = 0;
}

void WeightDoubleBuffer::Shutdown() {
#ifdef RG_CUDA_SUPPORT
	if (readyEvent) {
		cudaEventDestroy(readyEvent);
		readyEvent = nullptr;
	}
	if (rawStream) {
		cudaStreamDestroy(rawStream);
		rawStream = nullptr;
		weightStream = nullptr;
	}
#endif
	buf[0].Free();
	buf[1].Free();
}

bool WeightDoubleBuffer::TryPublish() {
	if (recv_busy.load() != 0)
		return false;
	int pend = pending_idx.load();
	int rd = read_idx.load();
	if (pend < 0 || pend == rd)
		return false;
#ifdef RG_CUDA_SUPPORT
	if (readyEvent) {
		cudaError_t q = cudaEventQuery(readyEvent);
		if (q == cudaErrorNotReady)
			return false;
		if (q != cudaSuccess)
			cudaEventSynchronize(readyEvent);
	}
#endif
	read_idx.store(pend);
	return true;
}

void WeightDoubleBuffer::RequestBcast() {
	bcast_issued.fetch_add(1);
	{
		std::lock_guard<std::mutex> g(mu);
		nccl_go.store(1);
	}
	cv.notify_one();
}

void WeightDoubleBuffer::WaitBcastDone() {
	const uint64_t target = bcast_issued.load();
	std::unique_lock<std::mutex> lk(mu);
	cv.wait(lk, [&] { return bcast_done.load() >= target || stop.load(); });
}

void WeightDoubleBuffer::RequestStop() {
	{
		std::lock_guard<std::mutex> g(mu);
		stop.store(1);
	}
	cv.notify_all();
}

void WeightDoubleBuffer::RecvLoop(Dist::Session* dist) {
#ifdef RG_CUDA_SUPPORT
	cudaSetDevice(deviceIndex);
	c10::cuda::set_device(deviceIndex);
	c10::cuda::CUDAGuard guard(deviceIndex);
#endif
	// NCCL-only after main posts RequestBcast. Sitting in ncclBcast holds the GPU
	// and blocks InferActions (FUNNELED: GO is recvd on collector main).
	for (;;) {
		{
			std::unique_lock<std::mutex> lk(mu);
			cv.wait(lk, [&] { return stop.load() || nccl_go.load(); });
			if (stop.load() && !nccl_go.load())
				return;
			nccl_go.store(0);
		}
		recv_busy.store(1);
		int write = 1 - read_idx.load();
#ifdef RG_CUDA_SUPPORT
		c10::cuda::CUDAStream wtStream = c10::cuda::getStreamFromExternal(rawStream, deviceIndex);
		c10::cuda::CUDAStreamGuard streamGuard(wtStream);
#endif
		dist->bcast_weights(packStaging.data_ptr<float>(), (size_t)packStaging.numel(), weightStream);
		PolicyPackHeader hdr{};
		pack.Unpack(packStaging, buf[write], &hdr);
#ifdef RG_CUDA_SUPPORT
		if (rawStream && readyEvent)
			cudaEventRecord(readyEvent, rawStream);
#endif
		if (hdr.policy_version == UINT64_MAX) {
			recv_busy.store(0);
			bcast_done.fetch_add(1);
			cv.notify_all();
			return;
		}
		pending_idx.store(write);
		pending_version.store(hdr.policy_version);
		pending_archive_seq.store(hdr.archive_seq);
		recv_busy.store(0);
		bcast_done.fetch_add(1);
		cv.notify_all();
	}
}

void WeightDoubleBuffer::StartRecvThread(Dist::Session* dist) {
	recvThread = std::thread([this, dist]() { RecvLoop(dist); });
}

void WeightDoubleBuffer::JoinRecvThread() {
	if (recvThread.joinable())
		recvThread.join();
}

}
