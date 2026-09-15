#pragma once
// Rain (Rou_Ad/RichardsWorld) obs builder, vendored from its shipped source.
// 231-dim variant of AdvancedObsPadded (ours is 230: theirs adds goal features).
// Kept as a separate class so our live obs is untouched.
#include "ObsBuilder.h"
#include <vector>

namespace RLGC {
    class AdvancedObsRain : public ObsBuilder {
    public:
        int maxPlayersPerTeam;

        // Immer mit dem Maximum initialisieren (z.B. 3 für bis zu 3v3)
        AdvancedObsRain(int maxPlayersPerTeam = 3);

        virtual FList BuildObs(const Player& player, const GameState& state) override;

    private:
        void AddPlayerToObs(FList& obs, const Player& player, bool inv, const PhysState& ball);
    };
}