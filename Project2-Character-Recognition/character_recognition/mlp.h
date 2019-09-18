#pragma once

#include "common.h"

namespace CharacterRecognition {
    Common::PerformanceTimer& timer();

    // TODO: implement required elements for MLP sections 1 and 2 here
	void fillRandomWeights(int n, float *data, float seed);

	float mlp(int inputSize, int numHiddenLayers, float expectedValue, 
		const float *weights1, const float *weights2, 
		const float *idata, float *partialDerivatives1, float *partialDerivatives2);

	float mlpNoError(int inputSize, int numHiddenLayers, float expectedValue,
		const float *weights1, const float *weights2,
		const float *idata);

	void updateWeights(int numWeights, float accumulatedError, const float *partials,
		float *weights);

}
