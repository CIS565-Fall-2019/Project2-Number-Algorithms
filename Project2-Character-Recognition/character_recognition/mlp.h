#pragma once

#include "common.h"

namespace CharacterRecognition {
    Common::PerformanceTimer& timer();

    // TODO: implement required elements for MLP sections 1 and 2 here
	/*void forward_pass(float* input);
	float compute_loss(int* true_output, float* predicted_output);
	void compute_gradients(int* true_output, float* predicted_output);
	void update_weights(int learning_rate);
	void initialize_network(int classes, int hidden_size, int lr);
	float train(float* input, int* true_labels, int number_of_epochs);
	float test(float* test_input);*/
	void initialize_network(int instances, int features, int classes, int hidden_size, float lr);
	void train(float* input, float* true_labels, int number_of_epochs);
	void test(float* test_input);
}
