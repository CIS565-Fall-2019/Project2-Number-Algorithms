#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include "common.h"
#include "mlp.h"
#include "device_launch_parameters.h"
#include <thrust/device_vector.h>
#include <thrust/fill.h>

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

#define blockSize 128

int number_of_instances;
int number_of_features;
int hidden_layer_size;
int number_of_classes;
float* learning_rate;

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

float* temp_loss;
float* loss_per_epoch;
float* all_losses;

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


// REF: https://solarianprogrammer.com/2012/05/31/matrix-multiplication-cuda-cublas-curand-thrust/
// Fill the array A(nr_rows_A, nr_cols_A) with random numbers on GPU
void GPU_fill_rand(float *A, int nr_rows_A, int nr_cols_A) {
	// Create a pseudo-random number generator


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

__global__ void exponential(int n, float* input, float* output) {
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= n) {
		return;
	}
	output[index] = std::exp(input[index]);
}

__global__ void upSweepOptimized(int n, int d, float* A) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);


	int other_index = 1 << d;
	int stride = other_index * 2;

	int new_index = stride * index;
	if (new_index >= n) {
		return;
	}
	A[new_index + stride - 1] += A[new_index + other_index - 1];
}

__global__ void intermediate_calculation(int n, float* temp, float* h) {
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= n) {
		return;
	}
	temp[index] *= (h[index] * (1 - h[index]));
}

__global__ void matrix_subtraction(int n, float* A, float* B, float* C) {
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= n) {
		return;
	}
	C[index] = A[index] - B[index];
}

__global__ void multiply_by_constant(int n, float* A, float* x) {
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= n) {
		return;
	}
	A[index] *= (*x);
}

void getArraySum(int n, float* input, float* sum) {
	float* padded_idata;
	int padded_size = 1 << (ilog2ceil(n));

	cudaMalloc((void**)&padded_idata, padded_size * sizeof(float));
	checkCUDAErrorWithLine("cudaMalloc padded_idata failed!");

	cudaMemset(padded_idata, 0, padded_size * sizeof(float));
	cudaMemcpy(padded_idata, input, sizeof(float) * n, cudaMemcpyDeviceToDevice);

	int iterations = ilog2(padded_size);

	int number_of_threads = padded_size;
	for (int d = 0; d < iterations; d++) {
		number_of_threads /= 2;
		dim3 fullBlocksPerGridUpSweep((number_of_threads + blockSize - 1) / blockSize);
		upSweepOptimized << <fullBlocksPerGridUpSweep, blockSize >> > (padded_size, d, padded_idata);
	}

	cudaMemcpy(sum, padded_idata + padded_size - 1, sizeof(float), cudaMemcpyDeviceToDevice);

	cudaFree(padded_idata);
}

__global__ void normalize(int n, float* input, float* sum) {
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= n) {
		return;
	}
	/*if(index == 1)
		printf("Sum: %f \n", *sum);*/
	input[index] /= (*sum);
}

__global__ void cross_entropy_loss(int n, float* true_label, float* predicted, float* temp) {
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= n) {
		return;
	}
	temp[index] = -1 * (true_label[index] * std::log(predicted[index]));
}

__global__ void add(float* a, float* b) {
	*a = (*a) + (*b);
}

__global__ void matrix_normalization(int n, float* A, float alpha, float beta) {
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= n) {
		return;
	}
	A[index] = (A[index] * alpha) - beta;
}

void softmax(int n, float* input, float* softmax_output) {
	dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

	exponential<< <fullBlocksPerGrid, blockSize >> >(n, input, softmax_output);

	float* sum;
	cudaMalloc((void**)&sum, sizeof(float));
	checkCUDAErrorWithLine("cudaMalloc sum failed!");

	getArraySum(n, softmax_output, sum);

	normalize << <fullBlocksPerGrid, blockSize >> > (n, softmax_output, sum);
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
		cudaMemcpy(current_input, input + (instance_number * number_of_features), sizeof(float) * number_of_features, cudaMemcpyDeviceToDevice);

		/*float* output_to_print = (float *)malloc(number_of_features * sizeof(float));
		cudaMemcpy(output_to_print, current_input, sizeof(float) * number_of_features, cudaMemcpyDeviceToHost);
		printf("Current Input: \n");
		print_matrix(output_to_print, number_of_features, 1);*/

		gpu_blas_mmul(handle, current_input, weight_input_hidden, hidden, 1, number_of_features, hidden_layer_size);

		/*float* output_to_print1 = (float *)malloc(hidden_layer_size * sizeof(float));
		cudaMemcpy(output_to_print1, hidden, sizeof(float) * hidden_layer_size, cudaMemcpyDeviceToHost);
		printf("Hidden layer: \n");
		print_matrix(output_to_print1, hidden_layer_size, 1);*/

		//Compute sigmoid if hidden layer
		dim3 fullBlocksPerGrid((hidden_layer_size + blockSize - 1) / blockSize);
		sigmoid << <fullBlocksPerGrid, blockSize >> > (hidden_layer_size, hidden, hidden_non_linear);

		/*float* output_to_print2 = (float *)malloc(hidden_layer_size * sizeof(float));
		cudaMemcpy(output_to_print2, hidden_non_linear, sizeof(float) * hidden_layer_size, cudaMemcpyDeviceToHost);
		printf("Hidden layer after sigmoid: \n");
		print_matrix(output_to_print2, hidden_layer_size, 1);*/

		//Compute output layer
		gpu_blas_mmul(handle, hidden_non_linear, weight_hidden_output, output, 1, hidden_layer_size, number_of_classes);

		/*float* output_to_print3 = (float *)malloc(number_of_classes * sizeof(float));
		cudaMemcpy(output_to_print3, output, sizeof(float) * number_of_classes, cudaMemcpyDeviceToHost);
		printf("Output layer: \n");
		print_matrix(output_to_print3, number_of_classes, 1);*/

		//Compute softmax of output layer
		softmax(number_of_classes, output, output_non_linear);

		/*float* output_to_print4 = (float *)malloc(number_of_classes * sizeof(float));
		cudaMemcpy(output_to_print4, output_non_linear, sizeof(float) * number_of_classes, cudaMemcpyDeviceToHost);
		printf("After Softmax: \n");
		print_matrix(output_to_print4, number_of_classes, 1);*/
	}

	//Returns the loss computed for the given iteration
	void compute_loss(float* true_output, float* loss) {
		float* temp;
		cudaMalloc((void**)&temp, number_of_classes * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc temp failed!");

		dim3 fullBlocksPerGridUpSweep((number_of_classes + blockSize - 1) / blockSize);
		cross_entropy_loss << <fullBlocksPerGridUpSweep, blockSize >> > (number_of_classes, true_output, output_non_linear, temp);

		getArraySum(number_of_classes, temp, loss);

		cudaFree(temp);
	}

	//Computes the gradient for the current pass. Updates - weight_input_hidden_gradient and weight_hidden_output_gradient
	void compute_gradients(cublasHandle_t &handle, float* true_output, float* input, int instance_number) {
		float* current_input;
		cudaMalloc((void**)&current_input, number_of_features * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc current_input failed!");
		cudaMemcpy(current_input, input + (instance_number * number_of_features), sizeof(float) * number_of_features, cudaMemcpyDeviceToDevice);
		
		float* current_output;
		cudaMalloc((void**)&current_output, number_of_classes * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc current_output failed!");
		cudaMemcpy(current_output, true_output + (instance_number * number_of_classes), sizeof(float) * number_of_classes, cudaMemcpyDeviceToDevice);

		//Compute gradient w.r.t weights between hidden and output layer
		float* temp;
		cudaMalloc((void**)&temp, number_of_classes * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc temp failed!");

		dim3 fullBlocksPerGridUpSweep((number_of_classes + blockSize - 1) / blockSize);
		matrix_subtraction << <fullBlocksPerGridUpSweep, blockSize >> > (number_of_classes, output_non_linear, current_output, temp);

		gpu_blas_mmul(handle, hidden_non_linear, temp, weight_hidden_output_gradient, hidden_layer_size, 1, number_of_classes);

		//Compute gradient w.r.t. weights between input and hidden layer
		float* temp1;
		cudaMalloc((void**)&temp1, hidden_layer_size * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc temp1 failed!");
		
		gpu_blas_mmul(handle, weight_hidden_output, temp, temp1, hidden_layer_size, number_of_classes, 1);

		dim3 fullBlocksPerGrid((hidden_layer_size + blockSize - 1) / blockSize);
		intermediate_calculation << <fullBlocksPerGrid, blockSize >> > (hidden_layer_size, temp1, hidden_non_linear);

		gpu_blas_mmul(handle, current_input, temp1, weight_input_hidden_gradient, number_of_features, 1, hidden_layer_size);

		//Compute loss
		cudaMalloc((void**)&temp_loss, sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc loss failed!");
		compute_loss(current_output, temp_loss);

		cudaFree(temp);
		cudaFree(temp1);
		cudaFree(current_input);
		cudaFree(current_output);
	}

	//Updates the weights according to the learning rate. Updates - weight_input_hidden and weight_hidden_output
	void update_weights() {
		dim3 fullBlocksPerGridUpSweep(((hidden_layer_size * number_of_classes) + blockSize - 1) / blockSize);
		multiply_by_constant << <fullBlocksPerGridUpSweep, blockSize >> > (hidden_layer_size * number_of_classes, weight_hidden_output_gradient, learning_rate);
		matrix_subtraction << <fullBlocksPerGridUpSweep, blockSize >> > (hidden_layer_size * number_of_classes, weight_hidden_output, weight_hidden_output_gradient, weight_hidden_output);
		
		dim3 fullBlocksPerGrid(((number_of_features * hidden_layer_size) + blockSize - 1) / blockSize);
		multiply_by_constant << <fullBlocksPerGrid, blockSize >> > (number_of_features * hidden_layer_size, weight_input_hidden_gradient, learning_rate);	
		matrix_subtraction << <fullBlocksPerGrid, blockSize >> > (number_of_features * hidden_layer_size, weight_input_hidden, weight_input_hidden_gradient, weight_input_hidden);
	}

	//To initialize network parameters like size of hidden and output layers and initialize weight matrices.
	void initialize_network(int instances, int features, int classes, int hidden_size, float lr) {
		number_of_instances = instances;
		number_of_features = features;
		number_of_classes = classes;
		hidden_layer_size = hidden_size;

		printf("%d %d %d \n", number_of_classes, hidden_layer_size, learning_rate);

		//Allocate memory for weight matrices on device
		cudaMalloc((void**)&weight_input_hidden, number_of_features * hidden_layer_size * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc weight_input_hidden failed!");

		cudaMalloc((void**)&weight_hidden_output, hidden_layer_size * number_of_classes * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc weight_hidden_output failed!");

		//Randomnly initialize weights
		curandGenerator_t prng;
		curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

		// Set the seed for the random number generator using the system clock
		//curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());
		curandSetPseudoRandomGeneratorSeed(prng, 7);

		// Fill the array with random numbers on the device
		curandGenerateUniform(prng, weight_input_hidden, number_of_features * hidden_layer_size);
		curandGenerateUniform(prng, weight_hidden_output, hidden_layer_size * number_of_classes);

		dim3 fullBlocksPerGridUpSweep((number_of_features * hidden_layer_size + blockSize - 1) / blockSize);
		matrix_normalization << <fullBlocksPerGridUpSweep, blockSize >> > (number_of_features * hidden_layer_size, weight_input_hidden, 2, 1);

		dim3 fullBlocksPerGrid((hidden_layer_size * number_of_classes + blockSize - 1) / blockSize);
		matrix_normalization << <fullBlocksPerGrid, blockSize >> > (hidden_layer_size * number_of_classes, weight_hidden_output, 2, 1);
		//GPU_fill_rand(weight_input_hidden, number_of_features, hidden_layer_size);
		//GPU_fill_rand(weight_hidden_output, hidden_layer_size, number_of_classes);

		//float* weight1 = (float *)malloc(number_of_features * hidden_layer_size * sizeof(float));
		//float* weight2 = (float *)malloc(hidden_layer_size * number_of_classes * sizeof(float));

		//cudaMemcpy(weight1, weight_input_hidden, sizeof(float) * number_of_features * hidden_layer_size, cudaMemcpyDeviceToHost);
		//print_matrix(weight1, number_of_features, hidden_layer_size);
		//cudaMemcpy(weight2, weight_hidden_output , sizeof(float) * hidden_layer_size * number_of_classes, cudaMemcpyDeviceToHost);
		//print_matrix(weight2, hidden_layer_size, number_of_classes);

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

		cudaMalloc((void**)&learning_rate, sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc sum failed!");
		thrust::device_ptr<float> lr_ptr(learning_rate);
		thrust::fill(lr_ptr, lr_ptr+1, lr);
		//cudaMemset(learning_rate, lr, sizeof(float));
	}

	//Returns training accuracy
	void train(float* input, float* true_labels, int number_of_epochs) {
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

		cudaMalloc((void**)&loss_per_epoch, sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc loss_per_epoch failed!");

		cudaMalloc((void**)&all_losses, number_of_epochs * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc loss_per_epoch failed!");
		

		/*forward_pass(handle, dev_input, 2 * number_of_features);

		compute_gradients(handle, dev_true_labels, 2 * number_of_features);

		update_weights();*/
		for (int i = 0; i < number_of_epochs; i++) {
			thrust::device_ptr<float> loss_ptr(loss_per_epoch);
			thrust::fill(loss_ptr, loss_ptr + 1, 0);
			for (int j = 0; j < number_of_instances; j++) {
				/*float* output_to_print1 = (float *)malloc(number_of_features * sizeof(float));
				cudaMemcpy(output_to_print1, dev_input + (j * number_of_features), sizeof(float) * number_of_features, cudaMemcpyDeviceToHost);
				printf("Input:  ");
				print_matrix(output_to_print1, 1, number_of_features);*/

				/*float* output_to_print2 = (float *)malloc(number_of_classes * sizeof(float));
				cudaMemcpy(output_to_print2, dev_true_labels + (j * number_of_classes), sizeof(float) * number_of_classes, cudaMemcpyDeviceToHost);
				printf("Output:  ");
				print_matrix(output_to_print2, 1, number_of_classes);*/

				//1. Forward Pass through network
				forward_pass(handle, dev_input, j);
				/*float* output_to_print3 = (float *)malloc(number_of_classes * sizeof(float));
				cudaMemcpy(output_to_print3, output_non_linear, sizeof(float) * number_of_classes, cudaMemcpyDeviceToHost);
				printf("Initial output probabilities:  ");
				print_matrix(output_to_print3, 1, number_of_classes);*/
				//2. Compute Loss
				//loss = compute_loss(true_labels, output);

				//3. Compute Gradients for all weight matrices
				compute_gradients(handle, dev_true_labels, dev_input, j);

				add << <1,1 >> > (loss_per_epoch, temp_loss);

				/*float* output_to_print9 = (float *)malloc(number_of_features * hidden_layer_size * sizeof(float));
				cudaMemcpy(output_to_print9, weight_input_hidden_gradient, sizeof(float) * number_of_features * hidden_layer_size, cudaMemcpyDeviceToHost);
				printf("Updated gradients [Input - Hidden]: \n");
				print_matrix(output_to_print9, number_of_features, hidden_layer_size);

				float* output_to_print10 = (float *)malloc(hidden_layer_size * number_of_classes * sizeof(float));
				cudaMemcpy(output_to_print10, weight_hidden_output_gradient, sizeof(float) * hidden_layer_size * number_of_classes, cudaMemcpyDeviceToHost);
				printf("Updated Gradients [Hidden - Output]: \n");
				print_matrix(output_to_print10,hidden_layer_size, number_of_classes);

				float* output_to_print6 = (float *)malloc(number_of_features * hidden_layer_size * sizeof(float));
				cudaMemcpy(output_to_print6, weight_input_hidden, sizeof(float) * number_of_features * hidden_layer_size, cudaMemcpyDeviceToHost);
				printf("Current weights [Input - Hidden]: \n");
				print_matrix(output_to_print6, number_of_features, hidden_layer_size);

				float* output_to_print7 = (float *)malloc(hidden_layer_size * number_of_classes * sizeof(float));
				cudaMemcpy(output_to_print7, weight_hidden_output, sizeof(float) * hidden_layer_size * number_of_classes, cudaMemcpyDeviceToHost);
				printf("Current weights [Hidden - Output]: \n");
				print_matrix(output_to_print7, hidden_layer_size, number_of_classes);*/

				//4. Update weights
				update_weights();

				/*float* output_to_print4 = (float *)malloc(number_of_features * hidden_layer_size * sizeof(float));
				cudaMemcpy(output_to_print4, weight_input_hidden, sizeof(float) * number_of_features * hidden_layer_size, cudaMemcpyDeviceToHost);
				printf("Updated weights [Input - Hidden]: \n");
				print_matrix(output_to_print4, number_of_features, hidden_layer_size);

				float* output_to_print5 = (float *)malloc(hidden_layer_size * number_of_classes * sizeof(float));
				cudaMemcpy(output_to_print5, weight_hidden_output, sizeof(float) * hidden_layer_size * number_of_classes, cudaMemcpyDeviceToHost);
				printf("Updated weights [Hidden - Output]: \n");
				print_matrix(output_to_print5, hidden_layer_size, number_of_classes);*/

				forward_pass(handle, dev_input, j);

				/*float* output_to_print8 = (float *)malloc(number_of_classes * sizeof(float));
				cudaMemcpy(output_to_print8, output_non_linear, sizeof(float) * number_of_classes, cudaMemcpyDeviceToHost);
				printf("Output Probabilities after update:  ");
				print_matrix(output_to_print8, 1, number_of_classes);*/

				
			}
			//Print loss after each epoch
			float* loss_print = (float *)malloc(sizeof(float));
			cudaMemcpy(loss_print, loss_per_epoch, sizeof(float), cudaMemcpyDeviceToHost);
			printf("EPOCH %d LOSS: %f \n", i, *loss_print/52);

			cudaMemcpy(all_losses + i, loss_per_epoch, sizeof(float), cudaMemcpyDeviceToDevice);
		}

		float* all_losses_print = (float *)malloc(number_of_epochs * sizeof(float));
		cudaMemcpy(all_losses_print, all_losses, sizeof(float) * number_of_epochs, cudaMemcpyDeviceToHost);
		printf("All losses  ");
		print_matrix(all_losses_print, number_of_epochs, 1);

		// Destroy the handle
		cublasDestroy(handle);
	
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
