#pragma once

#include "common.h"
#include <vector>

namespace CharacterRecognition {
    Common::PerformanceTimer& timer();

    // TODO: implement required elements for MLP sections 1 and 2 here
	//void getWeights(int numWeights, float *weights, bool training);

	//float runOneInput(int numInput, int numHidden, const float *input, const float *weights, float *weightsErrors, float expected, bool training);

	//void adjustWeights(int numWeights, float *weights);




	void train(int numInput, std::vector<float*> inputs, std::vector<int> expected);

	void run();


}
