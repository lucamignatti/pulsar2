#include <GigaLearnCPP/Distributed/Session.h>

#include <cstdlib>
#include <iostream>

namespace GGL {
namespace Dist {

struct Session::Impl {
	int rank = 0;
	int world = 1;
	int local_rank = 0;
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

void Session::barrier() {}

void Session::sum_host(int*, size_t) {}
void Session::sum_host(int64_t*, size_t) {}
void Session::sum_host(float*, size_t) {}
void Session::sum_host(double*, size_t) {}
void Session::min_host(int*, size_t) {}
void Session::max_host(double*, size_t) {}
void Session::avg_host(float*, size_t) {}
void Session::bcast_host(void*, size_t, int) {}

void Session::allreduce_sum_device(float*, size_t, Stream) {}
void Session::allreduce_avg_device(float*, size_t, Stream) {}
void Session::bcast_device(float*, size_t, int, Stream) {}

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
