#pragma once

#include "common.h"

namespace CharacterRecognition {
    Common::PerformanceTimer& timer();

    // TODO: implement required elements for MLP sections 1 and 2 here
	void fillRandomWeights(int n, float *data, float seed);

	void updateWeights(int n, int *input, float *weights, const float *patialErrorDeriv, float error);

	float mlp(int inputSize, int numHiddenLayers, float expectedValue, 
		const float *weights1, const float *weights2, 
		const float *idata, float *adjustedWeights1, float *adjustedWeights2, float *partialDerivatives1, float *partialDerivatives2);
}
