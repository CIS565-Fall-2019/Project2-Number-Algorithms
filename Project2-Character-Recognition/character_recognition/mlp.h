#pragma once

#include "common.h"

namespace CharacterRecognition {
    Common::PerformanceTimer& timer();

	void init();
	void train(float lambda);
	void evaluate(float *input, float *output);
	void end();
}
