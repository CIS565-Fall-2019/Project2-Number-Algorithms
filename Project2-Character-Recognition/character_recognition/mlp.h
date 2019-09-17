#pragma once

#include "common.h"
#include <vector>

namespace CharacterRecognition {
    Common::PerformanceTimer& timer();

	void train(int numInput, std::vector<float*> inputs, std::vector<int> expected);

	void run();


}
