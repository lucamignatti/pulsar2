#pragma once

#include "Report.h"

void GGL::Report::Display(std::vector<std::string> keyRows) const {
	std::stringstream stream;
	stream << std::string(8, '\n');
	stream << RG_DIVIDER << std::endl;
	for (std::string row : keyRows) {
		if (!row.empty()) {

			int indentLevel = 0;
			while (row[0] == '-') {
				indentLevel++;
				row.erase(row.begin());
			}

			std::string prefix = {};
			if (indentLevel > 0) {
				prefix += std::string((indentLevel - 1) * 3, ' ');
				prefix += " - ";
			}
			if (Has(row)) {
				stream << prefix << SingleToString(row, true) << std::endl;
			} else {
				continue;
			}
		} else {
			stream << std::endl;
		}
	}

	stream << std::string(4, '\n');

	std::cout << stream.str() << std::flush;
}
