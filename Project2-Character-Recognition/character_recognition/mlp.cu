#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include "common.h"
#include "mlp.h"
#include "device_launch_parameters.h"

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

#define blockSize 128

int number_of_instances;
int number_of_features;
int hidden_layer_size;
int number_of_classes;
int learning_rate;

float *weight_input_hidden;
float* weight_hidden_output;

float *weight_input_hidden_gradient;
float* weight_hidden_output_gradient;

float* dev_input;
float* dev_true_labels;

float* output;
float* hidden;

float* output_non_linear;
float* hidden_non_linear;

// REF: https://solarianprogrammer.com/2012/05/31/matrix-multiplication-cuda-cublas-curand-thrust/
// Fill the array A(nr_rows_A, nr_cols_A) with random numbers on GPU
void GPU_fill_rand(float *A, int nr_rows_A, int nr_cols_A) {
	// Create a pseudo-random number generator
	curandGenerator_t prng;
	curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

	// Set the seed for the random number generator using the system clock
	curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

	// Fill the array with random numbers on the device
	curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A);

}

// Multiply the arrays A and B on GPU and save the result in C
// C(m,n) = A(m,k) * B(k,n)
void gpu_blas_mmul(cublasHandle_t &handle, const float *A, const float *B, float *C, const int m, const int k, const int n) {
	int lda = m, ldb = k, ldc = m;
	const float alf = 1;
	const float bet = 0;
	const float *alpha = &alf;
	const float *beta = &bet;
	
	// Do the actual multiplication
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

__global__ void sigmoid(int n, float const *input, float *output) {
	/*  Computes the value of the sigmoid function f(x) = 1/(1 + e^-x).
	 Inputs:
	 input: array
	 output: array, the results of the computation are to be stored here
	*/

	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= n) {
		return;
	}

	output[index] = 1.0 / (1.0 + std::exp(-input[index]));
}

//Print matrix A(nr_rows_A, nr_cols_A) storage in column-major format
void print_matrix(const float *A, int nr_rows_A, int nr_cols_A) {
	for (int i = 0; i < nr_rows_A; ++i) {
		for (int j = 0; j < nr_cols_A; ++j) {
			printf("%f ", A[j * nr_rows_A + i]);
		}
		printf("\n");
	}
	printf("\n");
}

namespace CharacterRecognition {
    using Common::PerformanceTimer;
    PerformanceTimer& timer()
    {
        static PerformanceTimer timer;
        return timer;
    }
    
	//Performs one complete forward pass. Updates arrays - hidden and output
	void forward_pass(cublasHandle_t &handle, float* input, int instance_number) {
		//Compute hidden layer
		float* current_input;
		cudaMalloc((void**)&current_input, number_of_features * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc current_input failed!");
		cudaMemcpy(current_input, input + instance_number, sizeof(float) * number_of_features, cudaMemcpyDeviceToDevice);

		gpu_blas_mmul(handle, current_input, weight_input_hidden, hidden, 1, number_of_features, hidden_layer_size);

		//Compute sigmoid if hidden layer
		dim3 fullBlocksPerGrid((hidden_layer_size + blockSize - 1) / blockSize);
		sigmoid << <fullBlocksPerGrid, blockSize >> > (hidden_layer_size, hidden, hidden_non_linear);

		//Compute output layer
		gpu_blas_mmul(handle, hidden_non_linear, weight_hidden_output, output, 1, hidden_layer_size, number_of_classes);

		//Compute softmax of output layer

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
	void initialize_network(int instances, int features, int classes, int hidden_size, int lr) {
		number_of_instances = instances;
		number_of_features = features;
		number_of_classes = classes;
		hidden_layer_size = hidden_size;
		learning_rate = lr;

		printf("%d %d %d \n", number_of_classes, hidden_layer_size, learning_rate);

		//Allocate memory for weight matrices on device
		cudaMalloc((void**)&weight_input_hidden, number_of_features * hidden_layer_size * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc weight_input_hidden failed!");

		cudaMalloc((void**)&weight_hidden_output, hidden_layer_size * number_of_classes * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc weight_hidden_output failed!");

		//Randomnly initialize weights
		GPU_fill_rand(weight_input_hidden, number_of_features, hidden_layer_size);
		GPU_fill_rand(weight_hidden_output, hidden_layer_size, number_of_classes);

		float* weight1 = (float *)malloc(number_of_features * hidden_layer_size * sizeof(float));
		float* weight2 = (float *)malloc(hidden_layer_size * number_of_classes * sizeof(float));

		cudaMemcpy(weight1, weight_input_hidden, sizeof(float) * number_of_features * hidden_layer_size, cudaMemcpyDeviceToHost);
		print_matrix(weight1, number_of_features, hidden_layer_size);
		cudaMemcpy(weight2, weight_hidden_output , sizeof(float) * hidden_layer_size * number_of_classes, cudaMemcpyDeviceToHost);
		print_matrix(weight2, hidden_layer_size, number_of_classes);

		//Allocate memory for hidden layer and output on device
		cudaMalloc((void**)&hidden,  hidden_layer_size * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc hidden failed!");

		cudaMalloc((void**)&output, number_of_classes * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc output failed!");

		//Allocate memory for output of non-linear functions on device
		cudaMalloc((void**)&hidden_non_linear, hidden_layer_size * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc hidden_non_linear failed!");

		cudaMalloc((void**)&output_non_linear, number_of_classes * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc output_non_linear failed!");

		//Allocate memory for gradients on device
		cudaMalloc((void**)&weight_input_hidden_gradient, number_of_features * hidden_layer_size * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc weight_input_hidden_gradient failed!");

		cudaMalloc((void**)&weight_hidden_output_gradient, hidden_layer_size * number_of_classes * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc weight_hidden_output_gradient failed!");
	}

	//Returns training accuracy
	float train(float* input, int* true_labels, int number_of_epochs) {
		float loss;
		 
		//Allocate memory for input and copy data on device
		cudaMalloc((void**)&dev_input, number_of_instances * number_of_features * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_input failed!");
		cudaMemcpy(dev_input, input, sizeof(float) * number_of_instances * number_of_features, cudaMemcpyHostToDevice);

		//Allocate memory for true labels and copy data on device
		cudaMalloc((void**)&dev_true_labels, number_of_instances * number_of_classes * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_true_labels failed!");
		cudaMemcpy(dev_true_labels, true_labels, sizeof(float) * number_of_instances * number_of_classes, cudaMemcpyHostToDevice);

		// Create a handle for CUBLAS
		cublasHandle_t handle;
		cublasCreate(&handle);

		for (int i = 0; i < number_of_epochs; i++) {
			for (int j = 0; j < number_of_instances; j++) {
				//1. Forward Pass through network
				forward_pass(handle, dev_input, j * number_of_features);

				//2. Compute Loss
				loss = compute_loss(true_labels, output);

				//3. Compute Gradients for all weight matrices
				compute_gradients(true_labels, output);

				//4. Update weights
				update_weights(learning_rate);
			}
		}

		// Destroy the handle
		cublasDestroy(handle);
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
