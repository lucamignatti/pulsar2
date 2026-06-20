#pragma once
#include "../Framework.h"

#include "Utils.h"

namespace GGL {
	struct Report {
		typedef double Val;
		std::unordered_map<std::string, Val> data;

		struct Avg {
			Val total = 0;
			uint64_t count = 0;
			Avg() = default;
		};

		std::unordered_map<std::string, Avg> avgs;
		
		Report() = default;

		Val& operator[](const std::string& key) {
			return data[key];
		}

		Val operator[](const std::string& key) const {
			return data.at(key);
		}

		bool Has(const std::string& key) const {
			return data.find(key) != data.end();
		}

		void Add(const std::string& key, Val val) {
			if (Has(key)) {
				data[key] += val;
			} else {
				data[key] = val;
			}
		}

		void AddAvg(const std::string& key, Val val) {
			auto& avg = avgs[key];
			avg.total += val;
			avg.count++;
		}

		void FinishAvg(const std::string& key) {
			auto itr = avgs.find(key);
			if (itr == avgs.end())
				RG_ERR_CLOSE("Cannot call Report::FinishAvg() on non-existent average \"" << key << "\"!");
			if (itr->second.count == 0)
				RG_ERR_CLOSE("Cannot finish average metric \"" << key << "\" with zero samples!");

			data[key] = itr->second.total / (Val)itr->second.count;

			avgs.erase(itr);
		}

		std::string SingleToString(const std::string& key, bool digitCommas = false) const {
			Val val = (*this)[key];
			return key + ": "  + Utils::NumToStr(val);
		}

		std::string ToString(bool digitCommas = false, const std::string& prefix = {}) const {
			std::stringstream stream;
			for (auto pair : data) {
				stream << prefix << SingleToString(pair.first, digitCommas) << std::endl;
			}
			return stream.str();
		}

		void Finish() {
			for (auto& pair : avgs) {
				if (pair.second.count == 0)
					RG_ERR_CLOSE("Cannot finish average metric \"" << pair.first << "\" with zero samples!");
				data[pair.first] = pair.second.total / (Val)pair.second.count;
			}
			avgs.clear();
		}

		void Clear() {
			for (auto& pair : avgs) {
				RG_LOG(
					"WARNING: Unfinished average metric \"" << pair.first << "\", " <<
					"please call FinishAvg(\"" << pair.first << "\")/Finish() before the metrics report is cleared."
				);
			}
			avgs.clear();

			*this = Report();
		}

		Report operator+(const Report& other) const {
			Report newReport = *this;
			// insert() would silently keep our own value on key collisions; the merge
			// semantics we want are "the right-hand report wins", so assign explicitly.
			for (const auto& pair : other.data)
				newReport.data[pair.first] = pair.second;
			return newReport;
		}

		Report& operator+=(const Report& other) {
			*this = *this + other;
			return *this;
		}

		void Display(std::vector<std::string> keyRows) const;
	};
}
