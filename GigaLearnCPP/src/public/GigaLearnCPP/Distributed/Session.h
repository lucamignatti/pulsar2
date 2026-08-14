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

	void barrier();

	void sum_host(int* buf, size_t n);
	void sum_host(int64_t* buf, size_t n);
	void sum_host(float* buf, size_t n);
	void sum_host(double* buf, size_t n);
	void min_host(int* buf, size_t n);
	void max_host(int* buf, size_t n);
	void max_host(double* buf, size_t n);
	void avg_host(float* buf, size_t n);
	void bcast_host(void* buf, size_t nbytes, int root = 0);

	void allreduce_sum_device(float* ptr, size_t n, Stream stream = nullptr);
	void allreduce_avg_device(float* ptr, size_t n, Stream stream = nullptr);
	void bcast_device(float* ptr, size_t n, int root = 0, Stream stream = nullptr);

	// Pack small CUDA float32 grads into ~25 MiB buckets (one ncclAvg each), oversized
	// tensors in-place, single stream sync at the end. world<=1 is a no-op.
	void allreduce_avg_grads(const GradRef* grads, size_t nGrads, Stream stream = nullptr,
		size_t bucketBytes = 25 * 1024 * 1024);
	void allreduce_avg_grads(const std::vector<GradRef>& grads, Stream stream = nullptr,
		size_t bucketBytes = 25 * 1024 * 1024) {
		allreduce_avg_grads(grads.data(), grads.size(), stream, bucketBytes);
	}

private:
	Session();
	void RunSelfTest();

	struct Impl;
	std::unique_ptr<Impl> impl;
};

}
}
