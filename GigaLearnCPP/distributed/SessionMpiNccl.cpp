#include <GigaLearnCPP/Distributed/Session.h>

#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace GGL {
namespace Dist {

#define GGL_MPI_CHECK(cmd) do { \
	int _e = (cmd); \
	if (_e != MPI_SUCCESS) { \
		char _es[MPI_MAX_ERROR_STRING]; int _el = 0; \
		MPI_Error_string(_e, _es, &_el); \
		std::cerr << "MPI error " << __FILE__ << ":" << __LINE__ << " " << _es << std::endl; \
		MPI_Abort(MPI_COMM_WORLD, 1); \
	} \
} while (0)

#define GGL_CUDA_CHECK(cmd) do { \
	cudaError_t _e = (cmd); \
	if (_e != cudaSuccess) { \
		std::cerr << "CUDA error " << __FILE__ << ":" << __LINE__ << " " \
		          << cudaGetErrorString(_e) << std::endl; \
		MPI_Abort(MPI_COMM_WORLD, 1); \
	} \
} while (0)

#define GGL_NCCL_CHECK(cmd) do { \
	ncclResult_t _e = (cmd); \
	if (_e != ncclSuccess) { \
		std::cerr << "NCCL error " << __FILE__ << ":" << __LINE__ << " " \
		          << ncclGetErrorString(_e) << std::endl; \
		MPI_Abort(MPI_COMM_WORLD, 1); \
	} \
} while (0)

struct Session::Impl {
	int rank = 0;
	int world = 1;
	int local_rank = 0;
	ncclComm_t comm = nullptr;
	bool finalize_mpi = false;
};

static cudaStream_t AsCudaStream(Session::Stream stream) {
	return static_cast<cudaStream_t>(stream);
}

static void SyncStream(Session::Stream stream) {
	if (!stream)
		GGL_CUDA_CHECK(cudaDeviceSynchronize());
	else
		GGL_CUDA_CHECK(cudaStreamSynchronize(AsCudaStream(stream)));
}

Session::Session() = default;
Session::Session(Session&&) noexcept = default;
Session& Session::operator=(Session&&) noexcept = default;

Session::~Session() {
	if (!impl)
		return;
	if (impl->comm) {
		if (impl->world > 1)
			GGL_MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
		ncclCommDestroy(impl->comm);
		impl->comm = nullptr;
	}
	if (impl->finalize_mpi) {
		int flag = 0;
		MPI_Finalized(&flag);
		if (!flag)
			MPI_Finalize();
	}
}

Session Session::Init(int& argc, char**& argv) {
	Session s;
	s.impl = std::make_unique<Impl>();

	int already = 0;
	GGL_MPI_CHECK(MPI_Initialized(&already));
	if (!already) {
		int provided = 0;
		GGL_MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
		s.impl->finalize_mpi = true;
	}

	GGL_MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &s.impl->rank));
	GGL_MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &s.impl->world));

	MPI_Comm local = MPI_COMM_NULL;
	GGL_MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, s.impl->rank,
		MPI_INFO_NULL, &local));
	int node_rank = 0;
	GGL_MPI_CHECK(MPI_Comm_rank(local, &node_rank));
	GGL_MPI_CHECK(MPI_Comm_free(&local));

	int ndev = 0;
	GGL_CUDA_CHECK(cudaGetDeviceCount(&ndev));
	if (ndev <= 0) {
		std::cerr << "[rank " << s.impl->rank << "] no CUDA devices\n";
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	const int dev = (ndev == 1) ? 0 : (node_rank % ndev);
	GGL_CUDA_CHECK(cudaSetDevice(dev));
	s.impl->local_rank = dev;

	ncclUniqueId id;
	std::memset(&id, 0, sizeof(id));
	if (s.impl->rank == 0)
		GGL_NCCL_CHECK(ncclGetUniqueId(&id));
	GGL_MPI_CHECK(MPI_Bcast(&id, static_cast<int>(sizeof(id)), MPI_BYTE, 0, MPI_COMM_WORLD));
	GGL_NCCL_CHECK(ncclCommInitRank(&s.impl->comm, s.impl->world, id, s.impl->rank));

	if (s.impl->rank == 0) {
		std::cout << "[DIST] NCCL comm up  world=" << s.impl->world
		          << "  (self-test next)" << std::endl;
	}

	s.RunSelfTest();
	return s;
}

int Session::rank() const { return impl->rank; }
int Session::world() const { return impl->world; }
int Session::local_rank() const { return impl->local_rank; }

void Session::barrier() {
	GGL_MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
}

void Session::sum_host(int* buf, size_t n) {
	if (n == 0) return;
	GGL_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, buf, static_cast<int>(n), MPI_INT, MPI_SUM, MPI_COMM_WORLD));
}

void Session::sum_host(int64_t* buf, size_t n) {
	if (n == 0) return;
	GGL_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, buf, static_cast<int>(n), MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD));
}

void Session::sum_host(float* buf, size_t n) {
	if (n == 0) return;
	GGL_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, buf, static_cast<int>(n), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD));
}

void Session::sum_host(double* buf, size_t n) {
	if (n == 0) return;
	GGL_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, buf, static_cast<int>(n), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
}

void Session::min_host(int* buf, size_t n) {
	if (n == 0) return;
	GGL_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, buf, static_cast<int>(n), MPI_INT, MPI_MIN, MPI_COMM_WORLD));
}

void Session::max_host(double* buf, size_t n) {
	if (n == 0) return;
	GGL_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, buf, static_cast<int>(n), MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));
}

void Session::avg_host(float* buf, size_t n) {
	if (n == 0 || impl->world <= 1) return;
	sum_host(buf, n);
	const float inv = 1.0f / static_cast<float>(impl->world);
	for (size_t i = 0; i < n; ++i)
		buf[i] *= inv;
}

void Session::bcast_host(void* buf, size_t nbytes, int root) {
	if (nbytes == 0) return;
	GGL_MPI_CHECK(MPI_Bcast(buf, static_cast<int>(nbytes), MPI_BYTE, root, MPI_COMM_WORLD));
}

void Session::allreduce_sum_device(float* ptr, size_t n, Stream stream) {
	if (n == 0 || impl->world <= 1) return;
	GGL_NCCL_CHECK(ncclAllReduce(ptr, ptr, n, ncclFloat, ncclSum, impl->comm, AsCudaStream(stream)));
	SyncStream(stream);
}

void Session::allreduce_avg_device(float* ptr, size_t n, Stream stream) {
	if (n == 0 || impl->world <= 1) return;
	GGL_NCCL_CHECK(ncclAllReduce(ptr, ptr, n, ncclFloat, ncclAvg, impl->comm, AsCudaStream(stream)));
	SyncStream(stream);
}

void Session::bcast_device(float* ptr, size_t n, int root, Stream stream) {
	if (n == 0 || impl->world <= 1) return;
	GGL_NCCL_CHECK(ncclBroadcast(ptr, ptr, n, ncclFloat, root, impl->comm, AsCudaStream(stream)));
	SyncStream(stream);
}

void Session::RunSelfTest() {
	const int r = rank();
	const int w = world();

	auto fail = [&](const std::string& msg) {
		std::cerr << "[rank " << r << "] DIST SELFTEST FAILED: " << msg << std::endl;
		MPI_Abort(MPI_COMM_WORLD, 1);
	};
	auto pass = [&](const char* name) {
		if (r == 0)
			std::cout << "[DIST] " << name << " PASSED" << std::endl;
	};

	barrier();
	pass("MPI barrier");

	{
		int v = r * 10 + 5;
		sum_host(&v, 1);
		const int expected = 10 * (w * (w - 1) / 2) + 5 * w;
		if (v != expected)
			fail("sum_host(int)");
	}
	pass("MPI sum int");

	{
		int64_t v = static_cast<int64_t>(r) * 100 + 3;
		sum_host(&v, 1);
		const int64_t expected = 100 * (static_cast<int64_t>(w) * (w - 1) / 2) + 3 * w;
		if (v != expected)
			fail("sum_host(int64)");
	}
	pass("MPI sum int64");

	{
		double v = static_cast<double>(r) + 0.25;
		sum_host(&v, 1);
		const double expected = static_cast<double>(w) * (w - 1) / 2.0 + 0.25 * w;
		if (std::abs(v - expected) > 1e-6)
			fail("sum_host(double)");
	}
	pass("MPI sum double");

	{
		int v = r * 10 + 5;
		min_host(&v, 1);
		if (v != 5)
			fail("min_host(int)");
	}
	pass("MPI min int");

	{
		double v = static_cast<double>(r) + 1.5;
		max_host(&v, 1);
		if (std::abs(v - (static_cast<double>(w - 1) + 1.5)) > 1e-12)
			fail("max_host(double)");
	}
	pass("MPI max double");

	{
		float v = static_cast<float>(r) * 10.0f + 5.0f;
		avg_host(&v, 1);
		const float meanRank = (static_cast<float>(w) - 1.0f) / 2.0f;
		const float expected = meanRank * 10.0f + 5.0f;
		if (std::abs(v - expected) > 1e-3f)
			fail("avg_host(float)");
	}
	pass("MPI avg float");

	{
		unsigned char buf[8];
		if (r == 0) {
			for (int i = 0; i < 8; ++i)
				buf[i] = static_cast<unsigned char>(0xA0 + i);
		} else {
			std::memset(buf, 0xFF, sizeof(buf));
		}
		bcast_host(buf, sizeof(buf), 0);
		for (int i = 0; i < 8; ++i) {
			if (buf[i] != static_cast<unsigned char>(0xA0 + i))
				fail("bcast_host");
		}
	}
	pass("MPI bcast bytes");

	const float meanRank = (static_cast<float>(w) - 1.0f) / 2.0f;

	{
		float* d = nullptr;
		GGL_CUDA_CHECK(cudaMalloc(&d, sizeof(float)));
		float h = static_cast<float>(r);
		GGL_CUDA_CHECK(cudaMemcpy(d, &h, sizeof(float), cudaMemcpyHostToDevice));
		allreduce_avg_device(d, 1, nullptr);
		GGL_CUDA_CHECK(cudaMemcpy(&h, d, sizeof(float), cudaMemcpyDeviceToHost));
		GGL_CUDA_CHECK(cudaFree(d));
		if (std::abs(h - meanRank) > 1e-3f) {
			std::ostringstream ss;
			ss << "NCCL avg 1-float got " << h << " expected " << meanRank;
			fail(ss.str());
		}
	}
	pass("NCCL allreduce avg (1)");

	{
		const size_t n = 1u << 20;
		std::vector<float> h(n);
		for (size_t i = 0; i < n; ++i)
			h[i] = static_cast<float>(r) + static_cast<float>(i) * 0.001f;
		float* d = nullptr;
		GGL_CUDA_CHECK(cudaMalloc(&d, n * sizeof(float)));
		GGL_CUDA_CHECK(cudaMemcpy(d, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
		allreduce_avg_device(d, n, nullptr);
		GGL_CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost));
		GGL_CUDA_CHECK(cudaFree(d));
		for (size_t i = 0; i < n; i += n / 16) {
			const float expected = meanRank + static_cast<float>(i) * 0.001f;
			if (std::abs(h[i] - expected) > 1e-2f) {
				std::ostringstream ss;
				ss << "NCCL avg 1e6 mismatch at " << i << " got " << h[i] << " expected " << expected;
				fail(ss.str());
			}
		}
	}
	pass("NCCL allreduce avg (1e6)");

	{
		float* d = nullptr;
		GGL_CUDA_CHECK(cudaMalloc(&d, sizeof(float)));
		float h = static_cast<float>(r + 1);
		GGL_CUDA_CHECK(cudaMemcpy(d, &h, sizeof(float), cudaMemcpyHostToDevice));
		allreduce_sum_device(d, 1, nullptr);
		GGL_CUDA_CHECK(cudaMemcpy(&h, d, sizeof(float), cudaMemcpyDeviceToHost));
		GGL_CUDA_CHECK(cudaFree(d));
		const float expected = static_cast<float>(w) * (w + 1) / 2.0f;
		if (std::abs(h - expected) > 1e-3f)
			fail("NCCL allreduce sum");
	}
	pass("NCCL allreduce sum (1)");

	{
		float* d = nullptr;
		GGL_CUDA_CHECK(cudaMalloc(&d, sizeof(float)));
		float h = (r == 0) ? 42.0f : -1.0f;
		GGL_CUDA_CHECK(cudaMemcpy(d, &h, sizeof(float), cudaMemcpyHostToDevice));
		bcast_device(d, 1, 0, nullptr);
		GGL_CUDA_CHECK(cudaMemcpy(&h, d, sizeof(float), cudaMemcpyDeviceToHost));
		GGL_CUDA_CHECK(cudaFree(d));
		if (std::abs(h - 42.0f) > 1e-6f)
			fail("NCCL bcast");
	}
	pass("NCCL bcast");

	{
		float mpi_v = static_cast<float>(r);
		sum_host(&mpi_v, 1);
		float* d = nullptr;
		GGL_CUDA_CHECK(cudaMalloc(&d, sizeof(float)));
		float nccl_v = static_cast<float>(r);
		GGL_CUDA_CHECK(cudaMemcpy(d, &nccl_v, sizeof(float), cudaMemcpyHostToDevice));
		allreduce_sum_device(d, 1, nullptr);
		GGL_CUDA_CHECK(cudaMemcpy(&nccl_v, d, sizeof(float), cudaMemcpyDeviceToHost));
		GGL_CUDA_CHECK(cudaFree(d));
		if (std::abs(mpi_v - nccl_v) > 1e-3f)
			fail("MPI vs NCCL sum cross-check");
	}
	pass("MPI vs NCCL sum cross-check");

	barrier();
	if (r == 0)
		std::cout << "[DIST] ALL SELFTESTS PASSED  world=" << w << std::endl;
}

}
}
