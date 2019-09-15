#pragma once

#include "common.h"

namespace CharacterRecognition {
    Common::PerformanceTimer& timer();
	// Initializes 
	// 1. CUBLAS Handle
	// 2. Weight Matrices
	void init(int input_size, int hidden_size, int output_size);

	void train(float* idata, float* ilabel, int num_instances, int epochs, float learning_rate);

	// Frees memory and destroys CUBLAS handle
	void free();
}
