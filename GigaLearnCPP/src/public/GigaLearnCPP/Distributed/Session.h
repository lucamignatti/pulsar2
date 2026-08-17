#pragma once

#include <GigaLearnCPP/Framework.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace GGL {
namespace Dist {

// Compile-time MPI+NCCL session. CMake selects SessionMpiNccl.cpp vs SessionStub.cpp.
// Public header does not include mpi.h / nccl.h / cuda_runtime.h.
class RG_IMEXPORT Session {
public:
	// cudaStream_t without pulling CUDA into every TU. Null is the default stream.
	using Stream = void*;

	static constexpr int kAnySource = -1;
	static constexpr int TAG_FRAGMENT = 100;
	static constexpr int TAG_EXIT = 101;
	static constexpr int TAG_WEIGHT_GO = 102;
	static constexpr int TAG_EXIT_ACK = 103;

	// Opaque MPI_Request storage (header stays MPI-free).
	struct CollectiveRequest {
		static constexpr size_t kStorage = 32;
		alignas(8) unsigned char storage[kStorage]{};
	};

	struct GradRef {
		float* ptr;
		size_t n;
	};

	// MPI_Init (if needed) + NCCL comm, then collective self-test. Aborts on mismatch.
	static Session Init(int& argc, char**& argv);

	Session(Session&&) noexcept;
	Session& operator=(Session&&) noexcept;
	~Session();

	Session(const Session&) = delete;
	Session& operator=(const Session&) = delete;

	int rank() const;
	int world() const;
	int local_rank() const;
	bool distributed() const { return world() > 1; }

	bool is_learner() const;
	bool is_collector() const;
	int n_learners() const;
	int n_collectors() const;
	int learner_rank() const;      // 0..nL-1 or -1
	int collector_rank() const;    // 0..nC-1 or -1
	int dest_learner_world(uint64_t fragment_id) const;

	// The rank/size of the COLLECTIVE GROUP this rank's allreduce/bcast actually run
	// on: the learner group under async routing, the world otherwise. Anything that
	// partitions work by rank ownership around a collective (Muon NS sharding) MUST
	// use these — global rank/world under async routing would assign owners to
	// collector ranks that never step, silently leaving their params stale.
	int group_rank() const;
	int group_world() const;

	// Expert-parallel learn support (MOE_SPEED.md Stage 2). Both run on the GROUP
	// (learner group under async routing): a host allgather for routing counts and
	// a variable-row fp16 device all-to-all (row counts/displacements are HOST
	// arrays of group_world() entries; self-traffic is a local device copy —
	// send-to-self inside an NCCL group was unreliable on this stack, measured in
	// moe-bench).
	// Broadcast on the GROUP comm (learner group under async routing). bcast_device
	// uses the WORLD comm — wrong for anything only learners execute.
	void bcast_device_group(float* ptr, size_t n, int root = 0, Stream stream = nullptr);
	void allgather_host_group(const int* send, int* recv, int perRank);
	void alltoall_rows_f16_group(
		const void* send, void* recv,
		const int* sendRows, const int* sendDisp,
		const int* recvRows, const int* recvDisp,
		int width, Stream stream = nullptr);

	void enable_async_routing();

	void barrier();
	void barrier_world();
	void barrier_learners();
	void ibarrier_learners(CollectiveRequest* req);
	bool test_request(CollectiveRequest* req);

	void sum_host(int* buf, size_t n);
	void sum_host(int64_t* buf, size_t n);
	void sum_host(float* buf, size_t n);
	void sum_host(double* buf, size_t n);
	void min_host(int* buf, size_t n);
	void imin_host(int* buf, size_t n, CollectiveRequest* req);
	void min_host_learners(int* buf, size_t n);
	void max_host(int* buf, size_t n);
	void max_host(double* buf, size_t n);
	void avg_host(float* buf, size_t n);
	void bcast_host(void* buf, size_t nbytes, int root = 0);
	void bcast_host_learners(void* buf, size_t nbytes, int root = 0);

	void sum_host_world(int* buf, size_t n);
	void max_host_world(int* buf, size_t n);

	void allreduce_sum_device(float* ptr, size_t n, Stream stream = nullptr);
	void allreduce_avg_device(float* ptr, size_t n, Stream stream = nullptr);
	void bcast_device(float* ptr, size_t n, int root = 0, Stream stream = nullptr);
	void bcast_weights(float* ptr, size_t n, Stream stream = nullptr, bool wait = true);

	void send_host(void* p, size_t nbytes, int dest_world, int tag);
	void ssend_host(void* p, size_t nbytes, int dest_world, int tag);
	void isend_host(void* p, size_t nbytes, int dest_world, int tag, CollectiveRequest* req);
	void recv_host(void* p, size_t nbytes, int src_world, int tag);
	bool iprobe_host(int src_world_or_any, int tag, size_t* nbytesOut = nullptr, int* srcOut = nullptr);
	bool test_host(CollectiveRequest* req);

	// Pack small CUDA float32 grads into ~25 MiB buckets (one ncclAvg each), oversized
	// tensors in-place, single stream sync at the end. Group size <=1 is a no-op.
	void allreduce_avg_grads(const GradRef* grads, size_t nGrads, Stream stream = nullptr,
		size_t bucketBytes = 25 * 1024 * 1024);
	void allreduce_avg_grads(const std::vector<GradRef>& grads, Stream stream = nullptr,
		size_t bucketBytes = 25 * 1024 * 1024) {
		allreduce_avg_grads(grads.data(), grads.size(), stream, bucketBytes);
	}

private:
	Session();
	void RunSelfTest();
	void RunAsyncSelfTest();
	friend struct SessionHelpers;

	struct Impl;
	std::unique_ptr<Impl> impl;
};

}
}
