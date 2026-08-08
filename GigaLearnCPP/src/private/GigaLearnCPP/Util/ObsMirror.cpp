#include "ObsMirror.h"
#include <RLGymCPP/CommonValues.h>
#include <RLGymCPP/BasicTypes/Action.h>

namespace GGL::ObsMirror {

	static void Vec3(std::vector<float>& sign, int at, float sx, float sy, float sz) {
		sign[at + 0] = sx; sign[at + 1] = sy; sign[at + 2] = sz;
	}

	Map Build(int maxPlayersPerTeam, int obsSize) {
		using namespace RLGC;

		constexpr int BALL = 9;                 // pos, vel, angVel
		const int ACT = (int)Action::ELEM_AMOUNT;   // throttle steer pitch yaw roll jump boost handbrake
		constexpr int PADS = CommonValues::BOOST_LOCATIONS_AMOUNT;
		constexpr int PLAYER = 29;              // AdvancedObs::AddPlayerToObs element count
		const int tmSlots = maxPlayersPerTeam - 1, opSlots = maxPlayersPerTeam;
		const int players = 1 + tmSlots + opSlots;
		const int presence = tmSlots + opSlots;
		const int expected = BALL + ACT + PADS + PLAYER * players + presence;
		if (!(expected == obsSize)) RG_ERR_CLOSE("ObsMirror: layout math " << expected << " != runtime obs size " << obsSize
			<< " -- the obs builder changed; rebuild this map before trusting it");

		std::vector<int64_t> perm(obsSize);
		std::vector<float> sign(obsSize, 1.f);
		for (int i = 0; i < obsSize; i++) perm[i] = i;

		int at = 0;
		// ball: pos, vel regular; angVel pseudo
		Vec3(sign, at + 0, -1, 1, 1);
		Vec3(sign, at + 3, -1, 1, 1);
		Vec3(sign, at + 6, 1, -1, -1);
		at += BALL;

		// prevAction: negate steer(1), yaw(3), roll(4)
		sign[at + 1] = -1.f;
		sign[at + 3] = -1.f;
		sign[at + 4] = -1.f;
		at += ACT;

		// boost pads: permute each pad to its x-mirrored partner. Nearest-match with a
		// 10uu tolerance: the vendored table carries a 2uu upstream asymmetry
		// (pad {-940, 3310} vs its partner {940, 3308}), so exact matching rejects a
		// physically-symmetric pair.
		for (int i = 0; i < PADS; i++) {
			const Vec& p = CommonValues::BOOST_LOCATIONS[i];
			int partner = -1; float bestD = 10.f;
			for (int j = 0; j < PADS; j++) {
				const Vec& q = CommonValues::BOOST_LOCATIONS[j];
				float d = std::abs(q.x + p.x) + std::abs(q.y - p.y) + std::abs(q.z - p.z);
				if (d < bestD) { bestD = d; partner = j; }
			}
			if (!(partner >= 0)) RG_ERR_CLOSE("ObsMirror: pad " << i << " has no mirror partner");
			perm[at + i] = at + partner;
		}
		at += PADS;

		// player blocks: pos, forward, up, vel (regular world); angVel (pseudo world);
		// local angVel (pseudo, f/r/u); local ball pos + vel (regular, f/r/u); 5 scalars
		for (int pIdx = 0; pIdx < players; pIdx++) {
			int b = at + pIdx * PLAYER;
			Vec3(sign, b + 0, -1, 1, 1);    // pos
			Vec3(sign, b + 3, -1, 1, 1);    // forward
			Vec3(sign, b + 6, -1, 1, 1);    // up
			Vec3(sign, b + 9, -1, 1, 1);    // vel
			Vec3(sign, b + 12, 1, -1, -1);  // angVel (pseudo, world)
			Vec3(sign, b + 15, -1, 1, -1);  // local angVel (pseudo, f/r/u)
			Vec3(sign, b + 18, 1, -1, 1);   // local ball pos (regular, f/r/u)
			Vec3(sign, b + 21, 1, -1, 1);   // local ball vel (regular, f/r/u)
			// b+24..b+28 scalars: unchanged
		}
		at += PLAYER * players;
		at += presence;                     // unchanged
		RG_ASSERT(at == obsSize);

		// structural checks: the map must be an involution (mirror twice = identity)
		for (int i = 0; i < obsSize; i++) {
			int j = (int)perm[i];
			if (!((int)perm[j] == i)) RG_ERR_CLOSE("ObsMirror: perm is not an involution at " << i);
			if (!(sign[i] == sign[j])) RG_ERR_CLOSE("ObsMirror: sign mismatch across perm pair " << i);
		}

		Map map;
		map.obsSize = obsSize;
		map.perm = torch::tensor(perm, torch::kLong);
		map.sign = torch::tensor(sign, torch::kFloat32);
		return map;
	}

	torch::Tensor Apply(const Map& map, const torch::Tensor& obs) {
		RG_ASSERT(map.IsValid() && obs.size(-1) == map.obsSize);
		auto perm = map.perm.to(obs.device());
		auto sign = map.sign.to(obs.device(), obs.scalar_type());
		return obs.index_select(-1, perm) * sign;
	}
}
