#pragma once

#ifdef RG_ROCKETSIM_V3
// v3 Rust engine behind the v2-compatible facade (see RocketSimV3/README.md)
#include "../RocketSimV3/compat/RocketSim.h"
#include "../RocketSimV3/compat/GameEventTracker.h"
#else
#include "../RocketSim/src/RocketSim.h"
#include "../RocketSim/src/Sim/GameEventTracker/GameEventTracker.h"
#endif

// Use RocketSim namespace
using namespace RocketSim;

// Define our own log
#define RG_LOG(s) { std::cout << s << std::endl; }

#define RG_NO_COPY(className) \
className(const className&) = delete;  \
className& operator= (const className&) = delete

#define RG_ERR_CLOSE(s) { \
std::string _errorStr = RS_STR("RG FATAL ERROR: " << s); \
RG_LOG(_errorStr); \
throw std::runtime_error(_errorStr); \
exit(EXIT_FAILURE); \
}

#ifndef RG_UNSAFE
#define RG_ASSERT(cond) { if (!(cond)) { RG_ERR_CLOSE("Assertion failed: " << #cond); } }
#else
#define RG_ASSERT(cond) {}
#endif

#define RG_DIVIDER std::string(40, '=')