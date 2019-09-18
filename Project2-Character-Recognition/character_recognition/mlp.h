#pragma once

#include "common.h"

namespace CharacterRecognition {
    Common::PerformanceTimer& timer();

    // TODO: implement required elements for MLP sections 1 and 2 here
	void initializeW(float* weights, int *layerSizes, int layerNum);
	float computeCostGrad(int *layerSizes, int layerNum, int batchSize, float *weights, float *grad, float *data, float *label);
	void updateWeights(int n, float *weight, float *weightgrad, float alpha);
}
