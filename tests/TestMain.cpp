#include "TestHarness.h"
#include <cstdio>

void RunWelfordTests();
void RunGAETests();
void RunExperienceBufferTests();
void RunRewardNameTests();
void RunTrainingBatchTests();
void RunMetricSinkTests();
void RunMuonTests();

int main() {
    printf("=== GigaLearn unit tests ===\n\n");

    RunWelfordTests();
    RunGAETests();
    RunExperienceBufferTests();
    RunRewardNameTests();
    RunTrainingBatchTests();
    RunMetricSinkTests();
    RunMuonTests();

    return TestHarness::Summary();
}
