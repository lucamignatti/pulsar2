#pragma once
#include <cstdint>
#include <vector>

namespace GGL {

	// Keep a LOG-SPACED span when there are more candidates than we can hold: repeatedly drop the
	// one whose removal least widens the largest gap in the retained timeline. Never drops the
	// oldest or the newest. Returns indices into `ts` (which must be ascending).
	//
	// Salvaged from the deleted LeagueArchive on 2026-07-25, where the rationale was written and
	// still holds: FIFO eviction would keep a moving RECENT window, which is precisely the pool
	// myopia a permanent reference set exists to cure. Keeping the OLDEST forever is what makes
	// Ref/Oldest Share a fixed yardstick rather than another treadmill.
	//
	// Header-only and dependency-free on purpose - it is pure integer logic, so it does not belong
	// behind the torch-dependent private headers where it would only be reachable from a smoke run.
	inline std::vector<size_t> DecimateSpaced(const std::vector<uint64_t>& ts, size_t keep) {
		std::vector<size_t> idx(ts.size());
		for (size_t i = 0; i < idx.size(); i++)
			idx[i] = i;
		while (idx.size() > keep && idx.size() > 2) {
			size_t bestPos = 1;
			double bestCost = 1e300;
			for (size_t p = 1; p + 1 < idx.size(); p++) { // never drop the oldest or newest
				double cost = (double)(ts[idx[p + 1]] - ts[idx[p - 1]]);
				if (cost < bestCost) {
					bestCost = cost;
					bestPos = p;
				}
			}
			idx.erase(idx.begin() + bestPos);
		}
		return idx;
	}
}
