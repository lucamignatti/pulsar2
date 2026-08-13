#pragma once

#include <GigaLearnCPP/Framework.h>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace GGL {
namespace Dist {

// Compile-time MPI+NCCL session. CMake selects SessionMpiNccl.cpp vs SessionStub.cpp.
// Public header does not include mpi.h / nccl.h / cuda_runtime.h.
class RG_IMEXPORT Session {
public:
	// cudaStream_t without pulling CUDA into every TU. Null is the default stream.
	using Stream = void*;

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
	void max_host(double* buf, size_t n);
	void avg_host(float* buf, size_t n);
	void bcast_host(void* buf, size_t nbytes, int root = 0);

	void allreduce_sum_device(float* ptr, size_t n, Stream stream = nullptr);
	void allreduce_avg_device(float* ptr, size_t n, Stream stream = nullptr);
	void bcast_device(float* ptr, size_t n, int root = 0, Stream stream = nullptr);

private:
	Session();
	void RunSelfTest();

	struct Impl;
	std::unique_ptr<Impl> impl;
};

}
}
