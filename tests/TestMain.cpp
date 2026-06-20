#include "TestHarness.h"
#include <cstdio>

void RunWelfordTests();
void RunGAETests();
void RunExperienceBufferTests();
void RunRewardNameTests();
void RunTrainingBatchTests();
void RunMetricSinkTests();

int main() {
    printf("=== GigaLearn unit tests ===\n\n");

    RunWelfordTests();
    RunGAETests();
    RunExperienceBufferTests();
    RunRewardNameTests();
    RunTrainingBatchTests();
    RunMetricSinkTests();

    return TestHarness::Summary();
}
