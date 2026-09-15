#include "AdvancedObsRain.h"
#include "AdvancedObs.h"
#include <RLGymCPP/Gamestates/StateUtil.h>
#include <vector>

namespace RLGC {

// (their shipped file defined a file-scope global instance here; removed — we
//  construct our own in the cross-play harness)

AdvancedObsRain::AdvancedObsRain(int maxPlayersPerTeam)
    : maxPlayersPerTeam(maxPlayersPerTeam) {}

void AdvancedObsRain::AddPlayerToObs(FList& obs, const Player& player, bool inv, const PhysState& ball) {
    auto phys = InvertPhys(player, inv);
    obs += phys.pos * AdvancedObs::POS_COEF;
    obs += phys.rotMat.forward;
    obs += phys.rotMat.up;
    obs += phys.vel * AdvancedObs::VEL_COEF;
    obs += phys.angVel * AdvancedObs::ANG_VEL_COEF;
    obs += phys.rotMat.Dot(phys.angVel) * AdvancedObs::ANG_VEL_COEF;
    obs += phys.rotMat.Dot(ball.pos - phys.pos) * AdvancedObs::POS_COEF;
    obs += phys.rotMat.Dot(ball.vel - phys.vel) * AdvancedObs::VEL_COEF;
    obs += player.boost / 100.0f;
    obs += player.isOnGround ? 1.0f : 0.0f;
    obs += player.HasFlipOrJump() ? 1.0f : 0.0f;
    obs += player.isDemoed ? 1.0f : 0.0f;
    obs += player.hasJumped ? 1.0f : 0.0f;
}

FList AdvancedObsRain::BuildObs(const Player& player, const GameState& state) {
    FList obs = {};
    bool inv = player.team == Team::ORANGE;

    auto ball = InvertPhys(state.ball, inv);
    auto& pads = state.GetBoostPads(inv);
    auto& padTimers = state.GetBoostPadTimers(inv);

    // Ball
    obs += ball.pos * AdvancedObs::POS_COEF;
    obs += ball.vel * AdvancedObs::VEL_COEF;
    obs += ball.angVel * AdvancedObs::ANG_VEL_COEF;

    // FIX 1: Torpositionen hinzufügen
    // Blue-Tor ist immer das eigene Tor für Blue, Orange-Tor für Orange
    Vec ownGoal   = inv ? CommonValues::ORANGE_GOAL_BACK : CommonValues::BLUE_GOAL_BACK;
    Vec enemyGoal = inv ? CommonValues::BLUE_GOAL_BACK   : CommonValues::ORANGE_GOAL_BACK;

    obs += (ownGoal   - ball.pos) * AdvancedObs::POS_COEF;  // Vektor: Ball → eigenes Tor
    obs += (enemyGoal - ball.pos) * AdvancedObs::POS_COEF;  // Vektor: Ball → gegnerisches Tor

    // Vorherige Aktion
    for (int i = 0; i < player.prevAction.ELEM_AMOUNT; i++)
        obs += player.prevAction[i];

    // Boost Pads
    for (int i = 0; i < CommonValues::BOOST_LOCATIONS_AMOUNT; i++) {
        if (pads[i]) {
            obs += 1.f;
        } else {
            obs += 1.f / (1.f + padTimers[i]);
        }
    }

    // Eigener Spieler
    FList selfObs = {};
    AddPlayerToObs(selfObs, player, inv, ball);
    obs += selfObs;
    size_t playerObsSize = selfObs.size();

    // Teammates & Opponents sammeln
    std::vector<FList> teammates, opponents;
    for (const auto& other : state.players) {
        if (other.carId == player.carId) continue;
        FList relObs = {};
        AddPlayerToObs(relObs, other, inv, ball);
        if (other.team == player.team)
            teammates.push_back(relObs);
        else
            opponents.push_back(relObs);
    }

    // FIX 2: Einheitliches Padding auf maxPlayersPerTeam
    // (bei obsBuilder(3) immer 2 Teammates + 3 Opponents = konstante Obs-Größe)
    while (teammates.size() > (size_t)(maxPlayersPerTeam - 1))
        teammates.pop_back();
    while (teammates.size() < (size_t)(maxPlayersPerTeam - 1))
        teammates.push_back(FList(playerObsSize, 0.0f));

    while (opponents.size() > (size_t)maxPlayersPerTeam)
        opponents.pop_back();
    while (opponents.size() < (size_t)maxPlayersPerTeam)
        opponents.push_back(FList(playerObsSize, 0.0f));

    for (const auto& t : teammates)
        obs += t;
    for (const auto& o : opponents)
        obs += o;

    return obs;
}

} // namespace RLGC