#pragma once

#include "common.h"

namespace CharacterRecognition {
    Common::PerformanceTimer& timer();
	void createAndTrainNN(int n, int h, int m, int d, float *idata, float *hidden, float *odata, float *weightsIH, float *weightsHO, float *actualOutput);
    // TODO: implement required elements for MLP sections 1 and 2 here
}
