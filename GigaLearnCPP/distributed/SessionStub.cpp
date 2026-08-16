#include <GigaLearnCPP/Distributed/Session.h>

#include <cstdlib>
#include <cstring>
#include <iostream>

namespace GGL {
namespace Dist {

struct Session::Impl {
	int rank = 0;
	int world = 1;
	int local_rank = 0;
	int nL = 1;
	int nC = 0;
	bool async_routing = false;
};

Session::Session() = default;
Session::Session(Session&&) noexcept = default;
Session& Session::operator=(Session&&) noexcept = default;
Session::~Session() = default;

Session Session::Init(int& argc, char**& argv) {
	(void)argc;
	(void)argv;
	Session s;
	s.impl = std::make_unique<Impl>();
	s.RunSelfTest();
	return s;
}

int Session::rank() const { return impl->rank; }
int Session::world() const { return impl->world; }
int Session::local_rank() const { return impl->local_rank; }

bool Session::is_learner() const { return true; }
bool Session::is_collector() const { return false; }
int Session::n_learners() const { return 1; }
int Session::n_collectors() const { return 0; }
int Session::learner_rank() const { return 0; }
int Session::collector_rank() const { return -1; }
int Session::dest_learner_world(uint64_t) const { return 0; }
void Session::enable_async_routing() {}

void Session::barrier() {}
void Session::barrier_world() {}
void Session::barrier_learners() {}
void Session::ibarrier_learners(CollectiveRequest*) {}
bool Session::test_request(CollectiveRequest*) { return true; }

void Session::sum_host(int*, size_t) {}
void Session::sum_host(int64_t*, size_t) {}
void Session::sum_host(float*, size_t) {}
void Session::sum_host(double*, size_t) {}
void Session::min_host(int*, size_t) {}
void Session::imin_host(int*, size_t, CollectiveRequest*) {}
void Session::min_host_learners(int*, size_t) {}
void Session::max_host(int*, size_t) {}
void Session::max_host(double*, size_t) {}
void Session::avg_host(float*, size_t) {}
void Session::bcast_host(void*, size_t, int) {}
void Session::bcast_host_learners(void*, size_t, int) {}
void Session::sum_host_world(int*, size_t) {}
void Session::max_host_world(int*, size_t) {}

void Session::allreduce_sum_device(float*, size_t, Stream) {}
void Session::allreduce_avg_device(float*, size_t, Stream) {}
void Session::bcast_device(float*, size_t, int, Stream) {}
void Session::bcast_weights(float*, size_t, Stream, bool) {}
void Session::send_host(void*, size_t, int, int) {}
void Session::ssend_host(void*, size_t, int, int) {}
void Session::isend_host(void*, size_t, int, int, CollectiveRequest*) {}
void Session::recv_host(void*, size_t, int, int) {}
bool Session::iprobe_host(int, int, size_t*, int*) { return false; }
bool Session::test_host(CollectiveRequest*) { return true; }
void Session::allreduce_avg_grads(const GradRef*, size_t, Stream, size_t) {}

void Session::RunAsyncSelfTest() {}

void Session::RunSelfTest() {
	if (rank() != 0 || world() != 1 || local_rank() != 0) {
		std::cerr << "[DIST] stub self-test failed: expected rank0/world1\n";
		std::exit(1);
	}
	int v = 7;
	sum_host(&v, 1);
	if (v != 7) {
		std::cerr << "[DIST] stub self-test failed: identity sum\n";
		std::exit(1);
	}
	std::cout << "[DIST] stub self-test PASSED (rank 0/1)\n";
}

}
}
