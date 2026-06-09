#pragma once
#include "../FrameworkTorch.h"
#include <GigaLearnCPP/Util/Utils.h>
#include <nlohmann/json.hpp>

namespace GGL {
	struct WelfordStat {
		double runningMean = 0, runningVariance = 0;

		int64_t count = 0;

		WelfordStat() {};

		void Increment(const FList& samples) {
			for (float sample : samples) {
				double delta = (double)sample - runningMean;
				double deltaN = delta / (count + 1);

				runningMean += deltaN;
				runningVariance += delta * deltaN * count;
				count++;
			}
		}

		void Reset() {
			*this = WelfordStat();
		}

		double GetMean() const {
			if (count < 2)
				return 0;

			return runningMean;
		}

		double GetSTD() const {
			if (count < 2)
				return 1;

			double curVar = runningVariance / (count - 1);
			if (curVar == 0)
				curVar = 1;
			return sqrt(curVar);
		}

		nlohmann::json ToJSON() const {
			nlohmann::json result = {};
			result["mean"] = runningMean;
			result["var"] = runningVariance;
			result["count"] = count;
			return result;
		}

		void ReadFromJSON(const nlohmann::json& json) {
			runningMean = json["mean"];
			runningVariance = json["var"];
			count = json["count"];
		}
	};

	struct BatchedWelfordStat {
		int width;
		std::vector<double> runningMeans, runningVariances;

		int64_t count = 0;

		BatchedWelfordStat(int width) : width(width) {
			runningMeans.resize(width);
			runningVariances.resize(width);
		};

		void IncrementRow(float* samples) {
			for (int i = 0; i < width; i++) {
				double delta = samples[i] - runningMeans[i];
				double deltaN = delta / (count + 1);
				runningMeans[i] += deltaN;
				runningVariances[i] += delta * deltaN * count;
			}
			count++;
		}

		void Reset() {
			*this = BatchedWelfordStat(width);
		}

		const std::vector<double>& GetMean() {
			return runningMeans;
		}

		std::vector<double> GetSTD() {
			if (count < 2)
				return std::vector<double>(width, 1);

			std::vector<double> result = runningVariances;
			for (double& d : result) {
				d /= (count - 1);
				if (d == 0)
					d = 1;

				d = sqrt(d);
			}
			
			return result;
		}

		nlohmann::json ToJSON() const {
			nlohmann::json result = {};
			result["means"] = Utils::MakeJSONArray<double>(runningMeans);
			result["vars"] = Utils::MakeJSONArray<double>(runningVariances);
			result["count"] = count;
			return result;
		}

		void ReadFromJSON(const nlohmann::json& json) {
			runningMeans = Utils::MakeVecFromJSON<double>(json["means"]);
			runningVariances = Utils::MakeVecFromJSON<double>(json["vars"]);
			count = json["count"];
		}
	};

	// ── Shared obs standardization ───────────────────────────────────────────
	// Single source of truth used by both the PPO collection loop and the ES rollouts, so
	// members and baseline always see obs normalized exactly the way the policy was trained.

	// Compute per-dim (offset, invStd) from running stats, with the same clamping the trainer uses.
	inline void ComputeObsNorm(
		BatchedWelfordStat* stat, float minSTD, float maxMeanRange,
		std::vector<float>& outOffset, std::vector<float>& outInvStd) {

		const std::vector<double>& meanVec = stat->GetMean();
		std::vector<double> stdVec = stat->GetSTD();
		int w = stat->width;
		outOffset.resize(w);
		outInvStd.resize(w);
		for (int j = 0; j < w; j++) {
			outOffset[j] = RS_CLAMP((float)meanVec[j], -maxMeanRange, maxMeanRange);
			outInvStd[j] = 1.0f / RS_MAX((float)stdVec[j], minSTD);
		}
	}

	// Apply a precomputed (offset, invStd) normalization in-place to a row-major [rows, width] obs buffer.
	inline void ApplyObsNorm(
		float* obsData, int rows, int width,
		const std::vector<float>& offset, const std::vector<float>& invStd) {

		for (int i = 0; i < rows; i++) {
			float* row = obsData + (size_t)i * width;
			for (int j = 0; j < width; j++)
				row[j] = (row[j] - offset[j]) * invStd[j];
		}
	}
}