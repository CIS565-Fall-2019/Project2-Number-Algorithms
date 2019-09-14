#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "mlp.h"

int hidden_layer_size;
int number_of_classes;
int learning_rate;

float *weight_input_hidden;
float* weight_hidden_output;

float *weight_input_hidden_gradient;
float* weight_hidden_output_gradient;

float* output;
float* hidden;

float* output_non_linear;
float* hidden_non_linear;

namespace CharacterRecognition {
    using Common::PerformanceTimer;
    PerformanceTimer& timer()
    {
        static PerformanceTimer timer;
        return timer;
    }
    
	//Performs one complete forward pass. Updates arrays - hidden and output
	void forward_pass(float* input) {

	}

	//Returns the loss computed for the given iteration
	float compute_loss(int* true_output, float* predicted_output) {
		return 0;
	}

	//Computes the gradient for the current pass. Updates - weight_input_hidden_gradient and weight_hidden_output_gradient
	void compute_gradients(int* true_output, float* predicted_output) {

	}

	//Updates the weights according to the learning rate. Updates - weight_input_hidden and weight_hidden_output
	void update_weights(int learning_rate) {

	}

	//To initialize network parameters like size of hidden and output layers and initialize weight matrices.
	void initialize_network(int classes, int hidden_size, int lr) {
		number_of_classes = classes;
		hidden_layer_size = hidden_size;
		learning_rate = lr;

		//Randomnly initialize weights
	}

	//Returns training accuracy
	float train(float* input, int* true_labels, int number_of_epochs) {
		float loss;
		for (int i = 0; i < number_of_epochs; i++) {
			//1. Forward Pass through network
			forward_pass(input);

			//2. Compute Loss
			loss = compute_loss(true_labels, output);

			//3. Compute Gradients for all weight matrices
			compute_gradients(true_labels, output);

			//4. Update weights
			update_weights(learning_rate);
		}
		return loss;
	}

	//Returns test acccuracy
	float test(float* test_input) {
		return 0;
	}
    // TODO: __global__

    /**
        * Example of use case (follow how you did it in stream compaction)
        */
    /*void scan(int n, int *odata, const int *idata) {
        timer().startGpuTimer();
        // TODO
        timer().endGpuTimer();
    }
    */

	// TODO: implement required elements for MLP sections 1 and 2 here
}
