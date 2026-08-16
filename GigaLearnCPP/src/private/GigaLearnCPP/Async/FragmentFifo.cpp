#include "FragmentFifo.h"

namespace GGL {

void FragmentFifo::enqueue(TrajectoryFragment frag) {
	if (learner_version > frag.hdr.max_policy_version
		&& (learner_version - frag.hdr.max_policy_version) >= (uint64_t)maxPolicyLag) {
		drop_stale++;
		return;
	}
	rowCount += frag.rows();
	q.push_back(std::move(frag));
	while (rowCount > fifoMaxRows && !q.empty()) {
		rowCount -= q.front().rows();
		q.pop_front();
		drop_overflow++;
	}
}

std::vector<TrajectoryFragment> FragmentFifo::pop_for_learn(int64_t n_used) {
	std::vector<TrajectoryFragment> out;
	int64_t got = 0;
	while (!q.empty() && got < n_used) {
		TrajectoryFragment f = std::move(q.front());
		q.pop_front();
		rowCount -= f.rows();
		int64_t need = n_used - got;
		if (f.rows() <= need) {
			got += f.rows();
			out.push_back(std::move(f));
		} else {
			TrajectoryFragment prefix = f.SlicePrefix(need);
			TrajectoryFragment suffix = f.SliceSuffix(need);
			got += prefix.rows();
			out.push_back(std::move(prefix));
			rowCount += suffix.rows();
			q.push_front(std::move(suffix));
		}
	}
	return out;
}

}
