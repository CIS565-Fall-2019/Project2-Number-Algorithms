#pragma once

#include "common.h"
#include <vector>

namespace CharacterRecognition {
    Common::PerformanceTimer& timer();

	void train(int numInput, std::vector<float*> inputs, std::vector<float> expected, std::string filename);

	void run(int numInput, std::vector<float*> inputs, std::vector<float> expected, std::string filename);

}
