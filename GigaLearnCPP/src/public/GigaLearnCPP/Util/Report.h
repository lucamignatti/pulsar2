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

		// LIVE: called (unqualified) by Report::Display in Report.cpp. The old signature took a
		// `digitCommas` flag that the body never read, and Display passed it `true` - so the
		// formatting it implied never happened. Parameter dropped 2026-07-25; the function stays.
		std::string SingleToString(const std::string& key) const {
			Val val = (*this)[key];
			return key + ": "  + Utils::NumToStr(val);
		}

		void AddAvg(const std::string& key, Val val) {
			auto& avg = avgs[key];
			avg.total += val;
			avg.count++;
		}

		void Finish() {
			for (auto& pair : avgs)
				data[pair.first] = pair.second.total / (Val)pair.second.count;
			avgs.clear();
		}

		void Clear() {
			for (auto& pair : avgs) {
				RG_LOG(
					"WARNING: Unfinished average metric \"" << pair.first << "\", " <<
					"please call Finish() before the metrics report is cleared."
				);
			}
			avgs.clear();

			*this = Report();
		}

		void Display(std::vector<std::string> keyRows) const;
	};
}