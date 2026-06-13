#pragma once
#include "ActionParser.h"
#include "../Gamestates/StateUtil.h"

namespace RLGC {

	// Actions match DiscreteAction in RLGymPPO_CPP and lookup_act.py in Python RLGym, but also has action masking
	class DefaultAction : public ActionParser {
	public:

		std::vector<Action> actions;
		std::vector<uint8_t> groundMask, airMask, jumpMask, boostMask;

		DefaultAction();

		virtual Action ParseAction(int index, const Player& player, const GameState& state) override {
			return MirrorActionX(actions[index], ShouldMirrorXForPlayer(player));
		}

		virtual int GetActionAmount() override {
			return actions.size();
		}

		virtual std::vector<uint8_t> GetActionMask(const Player& player, const GameState& state) override;
		virtual void GetActionMaskInto(std::vector<uint8_t>& result, const Player& player, const GameState& state) override;
	};
}
