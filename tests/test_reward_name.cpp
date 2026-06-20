#include "TestHarness.h"

#include <RLGymCPP/Rewards/Reward.h>
#include <RLGymCPP/Rewards/CommonRewards.h>

// Reward::GetName() tests.
//
// This method was recently fixed (commit bdd9def era) to use abi::__cxa_demangle
// on Itanium-ABI compilers (GCC/Clang), stripping the leading namespace prefix
// while preserving template arguments.
//
// Latent risks:
//   - Name still contains "::" after trimming (namespace not fully stripped)
//   - Template args truncated or dropped
//   - Caching broken: first and second call return different strings
//   - Two different subclasses returning the same name (collision via caching)
//   - Crash via double-free if __cxa_demangle returns nullptr

static void reward_name_strips_namespace_prefix() {
    RLGC::GoalReward r;
    std::string name = r.GetName();
    // Must not contain "::" anywhere (namespace fully stripped)
    TCHECK(name.find("::") == std::string::npos);
    // Must not start with a digit (mangled Itanium names start with digit counts)
    TCHECK(!name.empty() && !std::isdigit((unsigned char)name[0]));
    // Must contain "GoalReward" (the simple class name)
    TCHECK(name.find("GoalReward") != std::string::npos);
}

static void reward_name_velocity_ball_to_goal() {
    RLGC::VelocityBallToGoalReward r;
    std::string name = r.GetName();
    TCHECK(name.find("::") == std::string::npos);
    TCHECK(name.find("VelocityBallToGoalReward") != std::string::npos);
}

static void reward_name_template_reward_preserves_bracket() {
    // PlayerDataEventReward<&PlayerEventState::goal, false> is a template
    // instantiation; the type name after demangling contains '<'.
    // The namespace-stripping logic strips only the prefix BEFORE '<', so
    // the class name itself has no '::' but the template args (e.g.
    // &RLGC::PlayerEventState::goal) may still contain '::' — that's correct.
    RLGC::PlayerGoalReward r;  // typedef for PlayerDataEventReward<..., false>
    std::string name = r.GetName();

    // Must contain the template bracket (not stripped)
    TCHECK(name.find('<') != std::string::npos);
    // Must begin with the bare class name, not a mangled prefix
    TCHECK(!name.empty() && !std::isdigit((unsigned char)name[0]));
    // The class name part (before '<') must not contain '::'
    size_t tStart = name.find('<');
    std::string classNamePart = name.substr(0, tStart);
    TCHECK(classNamePart.find("::") == std::string::npos);
}

static void reward_name_caching_returns_same_string() {
    RLGC::GoalReward r;
    std::string first  = r.GetName();
    std::string second = r.GetName();
    // Exact bit-for-bit equality (cached value)
    TCHECK(first == second);
}

static void reward_name_different_classes_have_distinct_names() {
    // Verify the caching is per-instance/per-type, not shared across types.
    RLGC::GoalReward gr;
    RLGC::VelocityReward vr;
    TCHECK(gr.GetName() != vr.GetName());
}

static void reward_name_template_instantiations_are_distinct() {
    // This is the actual latent bug GetName() was fixed for: DemoReward and
    // PlayerGoalReward are BOTH PlayerDataEventReward<&..., false> and differ
    // ONLY in the template argument (a member pointer).  If the namespace trim
    // ate the template args (or stripped past '<'), these would collide and
    // wandb reward logs would silently merge two distinct rewards.
    RLGC::PlayerGoalReward goal;
    RLGC::DemoReward demo;
    RLGC::AssistReward assist;

    TCHECK(goal.GetName() != demo.GetName());
    TCHECK(goal.GetName() != assist.GetName());
    TCHECK(demo.GetName() != assist.GetName());
}

static void reward_name_no_crash_on_repeated_calls() {
    // If __cxa_demangle had a double-free or use-after-free, this would crash
    // under ASAN.  Just call it many times on the same instance.
    RLGC::GoalReward r;
    for (int i = 0; i < 100; i++)
        TCHECK(!r.GetName().empty());
}

static void reward_name_subclass_gets_own_name() {
    // A user-defined subclass must get ITS type name, not the base Reward name.
    struct MyCustomReward : RLGC::Reward {
        virtual float GetReward(const RLGC::Player&, const RLGC::GameState&, bool) override { return 0; }
    };

    MyCustomReward r;
    std::string name = r.GetName();
    TCHECK(!name.empty());
    TCHECK(name.find("::") == std::string::npos);
    // Should NOT be the base "Reward" name
    TCHECK(name != "Reward");
}

void RunRewardNameTests() {
    RUN_SUITE("reward_name::strips_namespace_prefix", reward_name_strips_namespace_prefix);
    RUN_SUITE("reward_name::velocity_ball_to_goal", reward_name_velocity_ball_to_goal);
    RUN_SUITE("reward_name::template_reward_preserves_bracket", reward_name_template_reward_preserves_bracket);
    RUN_SUITE("reward_name::caching_returns_same_string", reward_name_caching_returns_same_string);
    RUN_SUITE("reward_name::different_classes_have_distinct_names", reward_name_different_classes_have_distinct_names);
    RUN_SUITE("reward_name::template_instantiations_are_distinct", reward_name_template_instantiations_are_distinct);
    RUN_SUITE("reward_name::no_crash_on_repeated_calls", reward_name_no_crash_on_repeated_calls);
    RUN_SUITE("reward_name::subclass_gets_own_name", reward_name_subclass_gets_own_name);
}
