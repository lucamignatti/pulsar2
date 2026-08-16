#pragma once
#include "Fragment.h"
#include <deque>
#include <vector>

namespace GGL {

class FragmentFifo {
public:
	int64_t fifoMaxRows = 0;
	int maxPolicyLag = 16;
	uint64_t learner_version = 0;
	int drop_stale = 0;
	int drop_overflow = 0;

	int64_t rows() const { return rowCount; }
	int fragment_count() const { return (int)q.size(); }

	void enqueue(TrajectoryFragment frag);
	std::vector<TrajectoryFragment> pop_for_learn(int64_t n_used);

private:
	std::deque<TrajectoryFragment> q;
	int64_t rowCount = 0;
};

}
