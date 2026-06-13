#pragma once
#include <RLGymCPP/BasicTypes/Lists.h>
#include <nlohmann/json.hpp>

#include <cmath>
#include <vector>

namespace GGL {

	// EMA-decayed fixed-bin histogram over [0, maxValue], for the adaptive-target
	// quantile trackers (Feature C). Update() once per learner iteration: existing mass
	// decays by `decay`, this iteration's samples are added at weight 1 each — so the
	// quantile estimate tracks a ~1/(1-decay)-iteration horizon of achieved values.
	// Samples beyond maxValue land in the last bin (only distorts quantiles above the
	// last bin edge, far past the tracked quantiles).
	struct EMAHistogram {
		int numBins;
		float maxValue;
		double decay;
		std::vector<double> counts;

		EMAHistogram(int numBins = 50, float maxValue = 6000, double decay = 0.999) :
			numBins(numBins), maxValue(maxValue), decay(decay) {
			counts.assign(numBins, 0.0);
		}

		void Update(const RLGC::FList& samples) {
			for (double& c : counts)
				c *= decay;
			float binWidth = maxValue / numBins;
			for (float sample : samples) {
				if (!std::isfinite(sample))
					continue;
				int bin = (int)(RS_MAX(sample, 0.0f) / binWidth);
				counts[RS_MIN(bin, numBins - 1)] += 1.0;
			}
		}

		double TotalWeight() const {
			double total = 0;
			for (double c : counts)
				total += c;
			return total;
		}

		// Weighted quantile with linear interpolation within the bin. Returns 0 on an
		// empty histogram (callers hold the target on cold/sparse data anyway).
		double Quantile(double q) const {
			double total = TotalWeight();
			if (total <= 0)
				return 0;

			double target = RS_CLAMP(q, 0.0, 1.0) * total;
			double binWidth = (double)maxValue / numBins;
			double cum = 0;
			for (int i = 0; i < numBins; i++) {
				if (cum + counts[i] >= target) {
					double frac = counts[i] > 0 ? (target - cum) / counts[i] : 0;
					return (i + frac) * binWidth;
				}
				cum += counts[i];
			}
			return maxValue;
		}

		nlohmann::json ToJSON() const {
			nlohmann::json result = {};
			result["num_bins"] = numBins;
			result["max_value"] = maxValue;
			result["counts"] = counts;
			return result;
		}

		void ReadFromJSON(const nlohmann::json& j) {
			// Only restore if the binning matches; otherwise keep the fresh histogram
			// (the seeded target + min-samples hold make a cold restart safe).
			if (j.contains("num_bins") && (int)j["num_bins"] == numBins &&
				j.contains("max_value") && (float)j["max_value"] == maxValue &&
				j.contains("counts")) {
				counts = j["counts"].get<std::vector<double>>();
				counts.resize(numBins, 0.0);
			}
		}
	};
}
