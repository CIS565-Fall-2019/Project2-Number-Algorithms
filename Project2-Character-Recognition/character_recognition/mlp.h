#pragma once

#include "common.h"

namespace CharacterRecognition {
    Common::PerformanceTimer& timer();
	/*
	Initializes the model framework
	1. Allocates memory for all Weight Matrices, Gradient Matrics and Hidden Layers
	2. Initializes the weight matrices with random numbers in the range of [-1, 1]
	*/
	void init(int input_size, int hidden_size, int output_size);

	/*
	Trains the model
	1. Creats reqiured device buffers
	2. Trains the model ( Forward Pass, Backward Pass, Loss calculation) epoch number of times
	3. Saves the model weights
	*/
	void train(float* idata, float* ilabel, int num_instances, int epochs, float learning_rate, std::string model_file);
	
	/*
	Predicts ouptut for num of instances in idata variable
	and logs the target variable and the predicted variable
	along with the confidence (prediction probability)
	*/
	void test(float* idata, float* true_label, int num_instances);

	/*
	Clears all model matrices, buffers and destroys the handles
	*/
	void free();
}
