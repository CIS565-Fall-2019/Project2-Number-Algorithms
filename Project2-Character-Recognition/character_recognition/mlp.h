#pragma once

#include "common.h"

namespace CharacterRecognition {
    Common::PerformanceTimer& timer();

    void makeWeightMat(int n, float* data);

    // TODO: implement required elements for MLP sections 1 and 2 here
    float mlpTrain(int i, int j, int k, float* odata, float* idata, float* wkj, float* wji, float* target);

    void mlpRun(int i, int j, int k, float* odata, float* idata, float* wkj, float* wji);
}
