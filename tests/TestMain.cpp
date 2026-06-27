#include "TestHarness.h"
#include <cstdio>

void RunWelfordTests();
void RunGAETests();
void RunExperienceBufferTests();
void RunRewardNameTests();
void RunTrainingBatchTests();
void RunMetricSinkTests();
void RunMuonTests();
void RunRSNormTests();
void RunTDContrastiveTests();

int main() {
    printf("=== GigaLearn unit tests ===\n\n");

    RunWelfordTests();
    RunGAETests();
    RunExperienceBufferTests();
    RunRewardNameTests();
    RunTrainingBatchTests();
    RunMetricSinkTests();
    RunMuonTests();
    RunRSNormTests();
    RunTDContrastiveTests();

    return TestHarness::Summary();
}
