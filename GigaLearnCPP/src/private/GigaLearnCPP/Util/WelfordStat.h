#pragma once
#include "../FrameworkTorch.h"
#include <GigaLearnCPP/Util/Utils.h>
#include <GigaLearnCPP/Distributed/Session.h>
#include <nlohmann/json.hpp>

namespace GGL {

	inline void ChanCombine(double& mean, double& m2, int64_t& n, double meanB, double m2B, int64_t nB) {
		if (nB <= 0)
			return;
		if (n <= 0) {
			mean = meanB;
			m2 = m2B;
			n = nB;
			return;
		}
		double delta = meanB - mean;
		int64_t nA = n;
		n = nA + nB;
		mean = mean + delta * (double)nB / (double)n;
		m2 = m2 + m2B + delta * delta * (double)nA * (double)nB / (double)n;
	}

	struct WelfordStat {
		double runningMean = 0, runningVariance = 0;
		int64_t count = 0;

		// This-iteration moments. Increment() only touches these; SyncAcrossRanks merges
		// them (allreduced when distributed) into the running stats via Chan's method.
		double localSum = 0, localSumSq = 0;
		int64_t localCount = 0;

		WelfordStat() {};

		void Increment(const FList& samples) {
			for (float sample : samples) {
				localSum += (double)sample;
				localSumSq += (double)sample * (double)sample;
				localCount++;
			}
		}

		void Reset() {
			*this = WelfordStat();
		}

		// Must be called after Increment() before GetMean/GetSTD should see the new
		// samples — including world==1 / dist==nullptr, which still fold locals in.
		void SyncAcrossRanks(Dist::Session* dist) {
			double sum = localSum, sumsq = localSumSq;
			int64_t n = localCount;
			if (dist && dist->distributed()) {
				dist->sum_host(&sum, 1);
				dist->sum_host(&sumsq, 1);
				dist->sum_host(&n, 1);
			}
			if (n > 0) {
				double meanB = sum / (double)n;
				double m2B = sumsq - sum * meanB;
				ChanCombine(runningMean, runningVariance, count, meanB, m2B, n);
			}
			localSum = localSumSq = 0;
			localCount = 0;
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
			localSum = localSumSq = 0;
			localCount = 0;
		}
	};

	struct BatchedWelfordStat {
		int width;
		std::vector<double> runningMeans, runningVariances;
		int64_t count = 0;

		std::vector<double> localSums, localSumSqs;
		int64_t localCount = 0;

		BatchedWelfordStat(int width) : width(width) {
			runningMeans.resize(width);
			runningVariances.resize(width);
			localSums.assign(width, 0);
			localSumSqs.assign(width, 0);
		};

		void IncrementRow(float* samples) {
			for (int i = 0; i < width; i++) {
				localSums[i] += samples[i];
				localSumSqs[i] += (double)samples[i] * samples[i];
			}
			localCount++;
		}

		void Reset() {
			*this = BatchedWelfordStat(width);
		}

		void SyncAcrossRanks(Dist::Session* dist) {
			std::vector<double> sums = localSums;
			std::vector<double> sumsqs = localSumSqs;
			int64_t n = localCount;
			if (dist && dist->distributed()) {
				dist->sum_host(sums.data(), (size_t)width);
				dist->sum_host(sumsqs.data(), (size_t)width);
				dist->sum_host(&n, 1);
			}
			if (n > 0) {
				int64_t newCount = count;
				for (int i = 0; i < width; i++) {
					int64_t c = count;
					double meanB = sums[i] / (double)n;
					double m2B = sumsqs[i] - sums[i] * meanB;
					ChanCombine(runningMeans[i], runningVariances[i], c, meanB, m2B, n);
					newCount = c;
				}
				count = newCount;
			}
			std::fill(localSums.begin(), localSums.end(), 0);
			std::fill(localSumSqs.begin(), localSumSqs.end(), 0);
			localCount = 0;
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
			// Keys must match ToJSON above ("means"/"vars") or the round-trip throws on resume
			runningMeans = Utils::MakeVecFromJSON<double>(json["means"]);
			runningVariances = Utils::MakeVecFromJSON<double>(json["vars"]);
			count = json["count"];
			localSums.assign(width, 0);
			localSumSqs.assign(width, 0);
			localCount = 0;
		}
	};
}
