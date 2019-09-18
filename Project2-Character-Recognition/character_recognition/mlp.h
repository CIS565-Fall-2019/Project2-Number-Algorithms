#pragma once

#include "common.h"

namespace CharacterRecognition {
    Common::PerformanceTimer& timer();

    // TODO: implement required elements for MLP sections 1 and 2 here
    //void initSimulation(int n, int output_num);
    //void SGD();
    void MLP_calculation(int n, int output_num,  float* idata, float* weight_matrix, float* odata);
}
