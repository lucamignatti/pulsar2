#include "TestHarness.h"
#include <cstdio>

void RunWelfordTests();
void RunGAETests();
void RunExperienceBufferTests();
void RunRewardNameTests();

int main() {
    printf("=== GigaLearn unit tests ===\n\n");

    RunWelfordTests();
    RunGAETests();
    RunExperienceBufferTests();
    RunRewardNameTests();

    return TestHarness::Summary();
}
