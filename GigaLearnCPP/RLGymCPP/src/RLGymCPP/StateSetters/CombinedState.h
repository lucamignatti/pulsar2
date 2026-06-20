#pragma once

#include "StateSetter.h"

#include "../Math.h"

namespace RLGC {
	// Combine state setters with weights
	// On env reset, one of the child state setters will be chosen at random, weighted by their corresponding weight
	class CombinedState : public StateSetter {
	private:
		std::vector<StateSetter*> setters = {};
		std::vector<float> cumulativeWeights = {};
		float totalWeight;

	public:
		CombinedState(const std::vector<std::pair<StateSetter*, float>>& setters) {

			RG_ASSERT(!setters.empty());

			totalWeight = 0;
			for (auto& pair : setters) {
				this->setters.push_back(pair.first);

				if (pair.second < 0 || isnan(pair.second) || isinf(pair.second)) {
					RG_ERR_CLOSE("CombinedState: State setter \"" << typeid(*pair.first).name() << "\" has a negative or invalid weight of " << pair.second);
				}

				totalWeight += pair.second;
				cumulativeWeights.push_back(totalWeight);
			}

			if (totalWeight <= 0)
				RG_ERR_CLOSE("CombinedState: Total state setter weight must be positive");
		}

		void ResetArena(Arena* arena) override {
			float f = RocketSim::Math::RandFloat(0, totalWeight);

			float prevWeight = 0;
			for (int i = 0; i < setters.size(); i++) {
				bool hasWeight = cumulativeWeights[i] > prevWeight;
				if (hasWeight && f < cumulativeWeights[i]) {
					setters[i]->ResetArena(arena);
					return;
				}
				prevWeight = cumulativeWeights[i];
			}

			for (int i = (int)setters.size() - 1; i >= 0; i--) {
				float prev = i > 0 ? cumulativeWeights[i - 1] : 0;
				if (cumulativeWeights[i] > prev) {
					setters[i]->ResetArena(arena);
					return;
				}
			}

			RG_ERR_CLOSE("CombinedState ran out of positive-weight setters before the matching cumulative weight was found (this should never happen)");
		}
	};
}
