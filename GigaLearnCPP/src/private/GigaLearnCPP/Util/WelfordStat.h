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
			runningMeans = Utils::MakeVecFromJSON<double>(json["mean"]);
			runningVariances = Utils::MakeVecFromJSON<double>(json["var"]);
			count = json["count"];
		}
	};
}