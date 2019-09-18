#pragma once

#include "common.h"

namespace CharacterRecognition {
    Common::PerformanceTimer& timer();

	void init();
	void train();
	void evaluate(float *input);
	void end();
}
