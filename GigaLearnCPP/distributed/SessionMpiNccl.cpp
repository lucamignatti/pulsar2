#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <GigaLearnCPP/Distributed/Session.h>

#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sched.h>
#include <unistd.h>
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
	int nL = 1;
	int nC = 0;
	bool async_routing = false;
	ncclComm_t comm = nullptr;
	ncclComm_t learners_nccl = nullptr;
	ncclComm_t weights_nccl = nullptr;
	MPI_Comm learners_mpi = MPI_COMM_NULL;
	MPI_Comm collectors_mpi = MPI_COMM_NULL;
	MPI_Comm weights_mpi = MPI_COMM_NULL;
	bool finalize_mpi = false;
	float* bucket_ws = nullptr;
	size_t bucket_ws_n = 0;
};

struct SessionHelpers {
	static MPI_Comm HostCommOrAbort(Session::Impl* impl, const char* what);
	static int HostGroup(Session::Impl* impl);
	static ncclComm_t DeviceComm(Session::Impl* impl);
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

static MPI_Request* AsReq(Session::CollectiveRequest* r) {
	static_assert(sizeof(MPI_Request) <= Session::CollectiveRequest::kStorage,
		"CollectiveRequest storage too small for MPI_Request");
	return reinterpret_cast<MPI_Request*>(r->storage);
}

static bool EnvTruthy(const char* name) {
	const char* e = std::getenv(name);
	return e && e[0] && std::strcmp(e, "0") != 0;
}

static void FreeIfSplit(MPI_Comm& c) {
	if (c != MPI_COMM_NULL && c != MPI_COMM_WORLD) {
		GGL_MPI_CHECK(MPI_Comm_free(&c));
		c = MPI_COMM_NULL;
	}
}

Session::Session() = default;
Session::Session(Session&&) noexcept = default;
Session& Session::operator=(Session&&) noexcept = default;

Session::~Session() {
	if (!impl)
		return;
	if (impl->bucket_ws) {
		cudaFree(impl->bucket_ws);
		impl->bucket_ws = nullptr;
		impl->bucket_ws_n = 0;
	}
	if (impl->weights_nccl) {
		ncclCommDestroy(impl->weights_nccl);
		impl->weights_nccl = nullptr;
	}
	if (impl->learners_nccl) {
		ncclCommDestroy(impl->learners_nccl);
		impl->learners_nccl = nullptr;
	}
	FreeIfSplit(impl->weights_mpi);
	FreeIfSplit(impl->learners_mpi);
	FreeIfSplit(impl->collectors_mpi);
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

MPI_Comm SessionHelpers::HostCommOrAbort(Session::Impl* impl, const char* what) {
	if (impl->async_routing && impl->nC > 0) {
		if (impl->learners_mpi == MPI_COMM_NULL) {
			std::cerr << "[rank " << impl->rank << "] " << what
			          << ": collectors must not enter Learners collectives\n";
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		return impl->learners_mpi;
	}
	return MPI_COMM_WORLD;
}

int SessionHelpers::HostGroup(Session::Impl* impl) {
	if (impl->async_routing && impl->nC > 0)
		return impl->nL;
	return impl->world;
}

ncclComm_t SessionHelpers::DeviceComm(Session::Impl* impl) {
	if (impl->async_routing && impl->nC > 0)
		return impl->learners_nccl;
	return impl->comm;
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
	// One physical GPU per local rank. Compute nodes have 6 V100s; the fen may
	// expose fewer — always trust cudaGetDeviceCount, never a hardcoded 4/6.
	// Do not set CUDA_VISIBLE_DEVICES here (libtorch will not reliably pick it up).
	const int dev = (ndev == 1) ? 0 : (node_rank % ndev);
	GGL_CUDA_CHECK(cudaSetDevice(dev));
	s.impl->local_rank = dev;
	{
		cudaDeviceProp prop;
		std::memset(&prop, 0, sizeof(prop));
		GGL_CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
		int runtimeDev = -1;
		GGL_CUDA_CHECK(cudaGetDevice(&runtimeDev));
		size_t memFree = 0, memTotal = 0;
		GGL_CUDA_CHECK(cudaMemGetInfo(&memFree, &memTotal));
		std::string cpus = "?";
		{
			std::ifstream st("/proc/self/status");
			std::string line;
			while (std::getline(st, line)) {
				const char* k = "Cpus_allowed_list:";
				if (line.compare(0, std::strlen(k), k) == 0) {
					cpus = line.substr(std::strlen(k));
					while (!cpus.empty() && (cpus[0] == ' ' || cpus[0] == '\t'))
						cpus.erase(cpus.begin());
					break;
				}
			}
		}
		int gpuNuma = -1;
		{
			char pciPath[96];
			std::snprintf(pciPath, sizeof(pciPath),
				"/sys/bus/pci/devices/%04x:%02x:%02x.0/numa_node",
				prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
			std::ifstream nf(pciPath);
			if (nf)
				nf >> gpuNuma;
		}
		const int cpu = sched_getcpu();
		const bool mismatch = (gpuNuma == 0 && cpu >= 80) || (gpuNuma == 8 && cpu < 80);
		std::cout << "[DIST][GPU] rank=" << s.impl->rank
		          << " pid=" << getpid()
		          << " node_rank=" << node_rank
		          << " ndev=" << ndev
		          << " device=" << dev
		          << " cudaGetDevice=" << runtimeDev
		          << " name=" << prop.name
		          << " pci=" << std::hex << prop.pciDomainID << ":"
		          << prop.pciBusID << ":" << prop.pciDeviceID << std::dec
		          << " gpu_numa=" << gpuNuma
		          << " cpu=" << cpu
		          << " cpus=" << cpus
		          << " mem_free_mb=" << (memFree / (1024 * 1024))
		          << " mem_total_mb=" << (memTotal / (1024 * 1024))
		          << (mismatch ? " NUMA_MISMATCH" : "")
		          << std::endl;
	}

	s.impl->nL = s.impl->world;
	s.impl->nC = 0;
	if (EnvTruthy("GGL_ASYNC")) {
		const char* nl = std::getenv("GGL_ASYNC_N_LEARNERS");
		if (!nl || !nl[0]) {
			std::cerr << "[DIST] GGL_ASYNC=1 requires GGL_ASYNC_N_LEARNERS\n";
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		s.impl->nL = std::atoi(nl);
		s.impl->nC = s.impl->world - s.impl->nL;
		if (s.impl->nL < 1 || s.impl->nC < 1) {
			std::cerr << "[DIST] GGL_ASYNC_N_LEARNERS=" << s.impl->nL
			          << " world=" << s.impl->world << " need 1<=nL<world\n";
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
	}

	ncclUniqueId id;
	std::memset(&id, 0, sizeof(id));
	if (s.impl->rank == 0)
		GGL_NCCL_CHECK(ncclGetUniqueId(&id));
	GGL_MPI_CHECK(MPI_Bcast(&id, static_cast<int>(sizeof(id)), MPI_BYTE, 0, MPI_COMM_WORLD));
	GGL_NCCL_CHECK(ncclCommInitRank(&s.impl->comm, s.impl->world, id, s.impl->rank));

	if (s.impl->nC > 0) {
		const int r = s.impl->rank;
		const int nL = s.impl->nL;
		const int nC = s.impl->nC;
		int lcolor = (r < nL) ? 1 : MPI_UNDEFINED;
		int ccolor = (r >= nL) ? 2 : MPI_UNDEFINED;
		int wcolor = (r == 0 || r >= nL) ? 3 : MPI_UNDEFINED;
		GGL_MPI_CHECK(MPI_Comm_split(MPI_COMM_WORLD, lcolor, r, &s.impl->learners_mpi));
		GGL_MPI_CHECK(MPI_Comm_split(MPI_COMM_WORLD, ccolor, r, &s.impl->collectors_mpi));
		GGL_MPI_CHECK(MPI_Comm_split(MPI_COMM_WORLD, wcolor, r, &s.impl->weights_mpi));

		if (s.impl->learners_mpi != MPI_COMM_NULL) {
			ncclUniqueId lid;
			std::memset(&lid, 0, sizeof(lid));
			if (r == 0)
				GGL_NCCL_CHECK(ncclGetUniqueId(&lid));
			GGL_MPI_CHECK(MPI_Bcast(&lid, static_cast<int>(sizeof(lid)), MPI_BYTE, 0, s.impl->learners_mpi));
			GGL_NCCL_CHECK(ncclCommInitRank(&s.impl->learners_nccl, nL, lid, r));
		}
		if (s.impl->weights_mpi != MPI_COMM_NULL) {
			ncclUniqueId wid;
			std::memset(&wid, 0, sizeof(wid));
			if (r == 0)
				GGL_NCCL_CHECK(ncclGetUniqueId(&wid));
			GGL_MPI_CHECK(MPI_Bcast(&wid, static_cast<int>(sizeof(wid)), MPI_BYTE, 0, s.impl->weights_mpi));
			const int wnccl = (r == 0) ? 0 : (1 + (r - nL));
			GGL_NCCL_CHECK(ncclCommInitRank(&s.impl->weights_nccl, 1 + nC, wid, wnccl));
		}
	}

	if (s.impl->rank == 0) {
		std::cout << "[DIST] NCCL comm up  world=" << s.impl->world
		          << " nL=" << s.impl->nL << " nC=" << s.impl->nC
		          << "  (self-test next)" << std::endl;
	}

	s.RunSelfTest();
	if (s.impl->nC > 0)
		s.RunAsyncSelfTest();
	return s;
}

int Session::rank() const { return impl->rank; }
int Session::world() const { return impl->world; }
int Session::local_rank() const { return impl->local_rank; }

bool Session::is_learner() const { return impl->nC == 0 || impl->rank < impl->nL; }
bool Session::is_collector() const { return impl->nC > 0 && impl->rank >= impl->nL; }
int Session::n_learners() const { return impl->nL; }
int Session::n_collectors() const { return impl->nC; }
int Session::learner_rank() const {
	return is_learner() ? impl->rank : -1;
}
int Session::collector_rank() const {
	return is_collector() ? (impl->rank - impl->nL) : -1;
}
int Session::dest_learner_world(uint64_t fragment_id) const {
	const int c = collector_rank();
	if (c < 0) {
		std::cerr << "[rank " << impl->rank << "] dest_learner_world on non-collector\n";
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	return static_cast<int>((static_cast<uint64_t>(c) + fragment_id) % static_cast<uint64_t>(impl->nL));
}

void Session::enable_async_routing() {
	if (impl->nC > 0)
		impl->async_routing = true;
}

void Session::barrier() {
	GGL_MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
}
void Session::barrier_world() { barrier(); }

void Session::barrier_learners() {
	if (impl->learners_mpi == MPI_COMM_NULL)
		return;
	GGL_MPI_CHECK(MPI_Barrier(impl->learners_mpi));
}

void Session::ibarrier_learners(CollectiveRequest* req) {
	if (!req)
		return;
	if (impl->learners_mpi == MPI_COMM_NULL) {
		int done = 1;
		std::memcpy(req->storage, &done, sizeof(done));
		return;
	}
	GGL_MPI_CHECK(MPI_Ibarrier(impl->learners_mpi, AsReq(req)));
}

bool Session::test_request(CollectiveRequest* req) {
	if (!req)
		return true;
	if (impl->learners_mpi == MPI_COMM_NULL)
		return true;
	int done = 0;
	GGL_MPI_CHECK(MPI_Test(AsReq(req), &done, MPI_STATUS_IGNORE));
	return done != 0;
}

void Session::sum_host(int* buf, size_t n) {
	if (n == 0) return;
	GGL_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, buf, static_cast<int>(n), MPI_INT, MPI_SUM,
		SessionHelpers::HostCommOrAbort(impl.get(), "sum_host")));
}
void Session::sum_host(int64_t* buf, size_t n) {
	if (n == 0) return;
	GGL_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, buf, static_cast<int>(n), MPI_INT64_T, MPI_SUM,
		SessionHelpers::HostCommOrAbort(impl.get(), "sum_host")));
}
void Session::sum_host(float* buf, size_t n) {
	if (n == 0) return;
	GGL_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, buf, static_cast<int>(n), MPI_FLOAT, MPI_SUM,
		SessionHelpers::HostCommOrAbort(impl.get(), "sum_host")));
}
void Session::sum_host(double* buf, size_t n) {
	if (n == 0) return;
	GGL_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, buf, static_cast<int>(n), MPI_DOUBLE, MPI_SUM,
		SessionHelpers::HostCommOrAbort(impl.get(), "sum_host")));
}
void Session::min_host(int* buf, size_t n) {
	if (n == 0) return;
	GGL_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, buf, static_cast<int>(n), MPI_INT, MPI_MIN,
		SessionHelpers::HostCommOrAbort(impl.get(), "min_host")));
}
void Session::imin_host(int* buf, size_t n, CollectiveRequest* req) {
	if (!req)
		return;
	if (n == 0) {
		MPI_Request r = MPI_REQUEST_NULL;
		std::memcpy(req->storage, &r, sizeof(r));
		return;
	}
	GGL_MPI_CHECK(MPI_Iallreduce(MPI_IN_PLACE, buf, static_cast<int>(n), MPI_INT, MPI_MIN,
		SessionHelpers::HostCommOrAbort(impl.get(), "imin_host"), AsReq(req)));
}
void Session::min_host_learners(int* buf, size_t n) {
	if (n == 0 || impl->learners_mpi == MPI_COMM_NULL) return;
	GGL_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, buf, static_cast<int>(n), MPI_INT, MPI_MIN,
		impl->learners_mpi));
}
void Session::max_host(int* buf, size_t n) {
	if (n == 0) return;
	GGL_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, buf, static_cast<int>(n), MPI_INT, MPI_MAX,
		SessionHelpers::HostCommOrAbort(impl.get(), "max_host")));
}
void Session::max_host(double* buf, size_t n) {
	if (n == 0) return;
	GGL_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, buf, static_cast<int>(n), MPI_DOUBLE, MPI_MAX,
		SessionHelpers::HostCommOrAbort(impl.get(), "max_host")));
}
void Session::avg_host(float* buf, size_t n) {
	const int g = SessionHelpers::HostGroup(impl.get());
	if (n == 0 || g <= 1) return;
	sum_host(buf, n);
	const float inv = 1.0f / static_cast<float>(g);
	for (size_t i = 0; i < n; ++i)
		buf[i] *= inv;
}
void Session::bcast_host(void* buf, size_t nbytes, int root) {
	if (nbytes == 0) return;
	GGL_MPI_CHECK(MPI_Bcast(buf, static_cast<int>(nbytes), MPI_BYTE, root, MPI_COMM_WORLD));
}
void Session::bcast_host_learners(void* buf, size_t nbytes, int root) {
	if (nbytes == 0 || impl->learners_mpi == MPI_COMM_NULL) return;
	GGL_MPI_CHECK(MPI_Bcast(buf, static_cast<int>(nbytes), MPI_BYTE, root, impl->learners_mpi));
}
void Session::sum_host_world(int* buf, size_t n) {
	if (n == 0) return;
	GGL_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, buf, static_cast<int>(n), MPI_INT, MPI_SUM, MPI_COMM_WORLD));
}
void Session::max_host_world(int* buf, size_t n) {
	if (n == 0) return;
	GGL_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, buf, static_cast<int>(n), MPI_INT, MPI_MAX, MPI_COMM_WORLD));
}

void Session::allreduce_sum_device(float* ptr, size_t n, Stream stream) {
	if (n == 0 || SessionHelpers::HostGroup(impl.get()) <= 1) return;
	GGL_NCCL_CHECK(ncclAllReduce(ptr, ptr, n, ncclFloat, ncclSum, SessionHelpers::DeviceComm(impl.get()), AsCudaStream(stream)));
	SyncStream(stream);
}
void Session::allreduce_avg_device(float* ptr, size_t n, Stream stream) {
	if (n == 0 || SessionHelpers::HostGroup(impl.get()) <= 1) return;
	GGL_NCCL_CHECK(ncclAllReduce(ptr, ptr, n, ncclFloat, ncclAvg, SessionHelpers::DeviceComm(impl.get()), AsCudaStream(stream)));
	SyncStream(stream);
}
void Session::bcast_device(float* ptr, size_t n, int root, Stream stream) {
	if (n == 0 || impl->world <= 1) return;
	GGL_NCCL_CHECK(ncclBroadcast(ptr, ptr, n, ncclFloat, root, impl->comm, AsCudaStream(stream)));
	SyncStream(stream);
}
void Session::bcast_weights(float* ptr, size_t n, Stream stream, bool wait) {
	if (n == 0 || !impl->weights_nccl) return;
	cudaStream_t s = AsCudaStream(stream);
	GGL_NCCL_CHECK(ncclBroadcast(ptr, ptr, n, ncclFloat, 0, impl->weights_nccl, s));
	if (wait)
		GGL_CUDA_CHECK(cudaStreamSynchronize(s));
}

void Session::send_host(void* p, size_t nbytes, int dest_world, int tag) {
	GGL_MPI_CHECK(MPI_Send(p, static_cast<int>(nbytes), MPI_BYTE, dest_world, tag, MPI_COMM_WORLD));
}
void Session::ssend_host(void* p, size_t nbytes, int dest_world, int tag) {
	GGL_MPI_CHECK(MPI_Ssend(p, static_cast<int>(nbytes), MPI_BYTE, dest_world, tag, MPI_COMM_WORLD));
}
void Session::isend_host(void* p, size_t nbytes, int dest_world, int tag, CollectiveRequest* req) {
	if (!req)
		return;
	GGL_MPI_CHECK(MPI_Isend(p, static_cast<int>(nbytes), MPI_BYTE, dest_world, tag, MPI_COMM_WORLD, AsReq(req)));
}
void Session::recv_host(void* p, size_t nbytes, int src_world, int tag) {
	const int src = (src_world == kAnySource) ? MPI_ANY_SOURCE : src_world;
	GGL_MPI_CHECK(MPI_Recv(p, static_cast<int>(nbytes), MPI_BYTE, src, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
}
bool Session::iprobe_host(int src_world_or_any, int tag, size_t* nbytesOut, int* srcOut) {
	int flag = 0;
	MPI_Status st;
	const int src = (src_world_or_any == kAnySource) ? MPI_ANY_SOURCE : src_world_or_any;
	GGL_MPI_CHECK(MPI_Iprobe(src, tag, MPI_COMM_WORLD, &flag, &st));
	if (!flag)
		return false;
	if (nbytesOut) {
		int count = 0;
		GGL_MPI_CHECK(MPI_Get_count(&st, MPI_BYTE, &count));
		*nbytesOut = static_cast<size_t>(count);
	}
	if (srcOut)
		*srcOut = st.MPI_SOURCE;
	return true;
}

bool Session::test_host(CollectiveRequest* req) {
	if (!req)
		return true;
	int done = 0;
	GGL_MPI_CHECK(MPI_Test(AsReq(req), &done, MPI_STATUS_IGNORE));
	return done != 0;
}

void Session::allreduce_avg_grads(const GradRef* grads, size_t nGrads, Stream stream, size_t bucketBytes) {
	if (!grads || nGrads == 0 || SessionHelpers::HostGroup(impl.get()) <= 1)
		return;

	ncclComm_t nc = SessionHelpers::DeviceComm(impl.get());
	const size_t bucketFloats = (bucketBytes / sizeof(float) < 1) ? 1 : (bucketBytes / sizeof(float));
	cudaStream_t cs = AsCudaStream(stream);

	auto ensureWs = [&](size_t n) {
		if (impl->bucket_ws_n >= n)
			return;
		if (impl->bucket_ws)
			GGL_CUDA_CHECK(cudaFree(impl->bucket_ws));
		GGL_CUDA_CHECK(cudaMalloc(&impl->bucket_ws, n * sizeof(float)));
		impl->bucket_ws_n = n;
	};
	ensureWs(bucketFloats);

	size_t packed = 0;
	std::vector<GradRef> pending;
	pending.reserve(64);

	auto flush = [&]() {
		if (packed == 0)
			return;
		GGL_NCCL_CHECK(ncclAllReduce(impl->bucket_ws, impl->bucket_ws, packed,
			ncclFloat, ncclAvg, nc, cs));
		size_t off = 0;
		for (const GradRef& g : pending) {
			GGL_CUDA_CHECK(cudaMemcpyAsync(g.ptr, impl->bucket_ws + off,
				g.n * sizeof(float), cudaMemcpyDeviceToDevice, cs));
			off += g.n;
		}
		pending.clear();
		packed = 0;
	};

	for (size_t i = 0; i < nGrads; i++) {
		const GradRef& g = grads[i];
		if (!g.ptr || g.n == 0)
			continue;
		if (g.n > bucketFloats) {
			flush();
			GGL_NCCL_CHECK(ncclAllReduce(g.ptr, g.ptr, g.n, ncclFloat, ncclAvg, nc, cs));
			continue;
		}
		if (packed + g.n > bucketFloats)
			flush();
		GGL_CUDA_CHECK(cudaMemcpyAsync(impl->bucket_ws + packed, g.ptr,
			g.n * sizeof(float), cudaMemcpyDeviceToDevice, cs));
		pending.push_back(g);
		packed += g.n;
	}
	flush();
	// Caller (clip_grad / AmpGradsFinite / epoch hook) stream-syncs when it needs the
	// reduced grads. CPU-sync here blocked fragment drain during ~30ms of NCCL.
}

void Session::RunAsyncSelfTest() {
	const int r = rank();
	auto fail = [&](const std::string& msg) {
		std::cerr << "[rank " << r << "] ASYNC SELFTEST FAILED: " << msg << std::endl;
		MPI_Abort(MPI_COMM_WORLD, 1);
	};

	if (is_learner()) {
		int v = learner_rank();
		min_host_learners(&v, 1);
		if (v != 0)
			fail("min_host_learners learner_rank");
	}

	if (impl->weights_nccl) {
		std::vector<float> h(1024, 0.f);
		if (r == 0) {
			for (int i = 0; i < 1024; ++i)
				h[i] = 0.5f * static_cast<float>(i);
		}
		float* d = nullptr;
		GGL_CUDA_CHECK(cudaMalloc(&d, h.size() * sizeof(float)));
		GGL_CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));
		bcast_weights(d, h.size(), nullptr);
		GGL_CUDA_CHECK(cudaMemcpy(h.data(), d, h.size() * sizeof(float), cudaMemcpyDeviceToHost));
		GGL_CUDA_CHECK(cudaFree(d));
		if (is_collector()) {
			for (int i = 0; i < 1024; ++i) {
				if (std::abs(h[i] - 0.5f * static_cast<float>(i)) > 1e-4f)
					fail("weights bcast pattern");
			}
		}
	}

	if (is_collector() && collector_rank() == 0) {
		std::vector<float> blob(4096);
		for (size_t i = 0; i < blob.size(); ++i)
			blob[i] = static_cast<float>(i) * 0.001f;
		send_host(blob.data(), blob.size() * sizeof(float), dest_learner_world(0), TAG_FRAGMENT);
	}
	if (r == 0) {
		std::vector<float> blob(4096, -1.f);
		recv_host(blob.data(), blob.size() * sizeof(float), kAnySource, TAG_FRAGMENT);
		for (size_t i = 0; i < blob.size(); ++i) {
			if (std::abs(blob[i] - static_cast<float>(i) * 0.001f) > 1e-4f)
				fail("fragment ping");
		}
	}

	barrier_world();
	if (r == 0)
		std::cout << "[DIST] ASYNC SELFTEST PASSED nL=" << impl->nL << " nC=" << impl->nC << std::endl;
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
