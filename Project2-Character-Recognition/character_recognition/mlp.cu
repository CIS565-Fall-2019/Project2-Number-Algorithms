#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>
#include "common.h"
#include "mlp.h"
#include "device_launch_parameters.h"
#include <thrust/device_vector.h>
#include <thrust/fill.h>

#define blockSize 128

int INPUT_LAYER_SIZE;
int HIDDEN_LAYER_SIZE;
int OUTPUT_LAYER_SIZE;
float *weights_IH, *weights_HO, *g_weights_IH, *g_weights_HO, *hidden, *h_sigmoid, *output, *o_softmax;
cublasHandle_t cublas_handle;
curandGenerator_t prng;

void print_matrix(const float *devA, int nr_rows_A, int nr_cols_A) {
	float *A = new float[nr_rows_A*nr_cols_A];
	cudaMemcpy(A, devA, nr_rows_A*nr_cols_A * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < nr_rows_A; ++i) {
		for (int j = 0; j < nr_cols_A; ++j) {
			printf("%f \t", A[j * nr_rows_A + i]);
		}
		printf("\n");
	}
	printf("\n");
}

namespace StreamCompaction {
	__global__ void kernelUpSweepStepEfficient(int n, int d, float* cdata) {
		int k = (blockIdx.x * blockDim.x) + threadIdx.x;
		if (k >= n)
			return;
		int prev_step_size = 1 << d;
		int cur_step_size = 2 * prev_step_size;
		int new_offset = k * cur_step_size;
		cdata[new_offset + cur_step_size - 1] += cdata[new_offset + prev_step_size - 1];
	}
	/**
	 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
	 */
	void sumArray(int n, float* sum, const float *idata) {
		// Memory Allocation and Copying
		int power_size = pow(2, ilog2ceil(n));
		float *sumArray;
		cudaMalloc((void**)&sumArray, power_size * sizeof(float));
		checkCUDAErrorFn("cudaMalloc sumArray failed!");
		cudaMemset(sumArray, 0, power_size * sizeof(float));
		cudaMemcpy(sumArray, idata, n * sizeof(float), cudaMemcpyDeviceToDevice);

		int numThreads;
		//Up Sweep
		for (int d = 0; d <= ilog2ceil(power_size) - 1; d++) {
			numThreads = pow(2, (ilog2ceil(power_size) - 1 - d));
			dim3 fullBlocks((numThreads + blockSize - 1) / blockSize);
			kernelUpSweepStepEfficient << <fullBlocks, blockSize >> > (numThreads, d, sumArray);
		}
		// Copy Back and Free Memory
		cudaMemcpy(sum, sumArray + power_size - 1, sizeof(float), cudaMemcpyDeviceToDevice);
		cudaFree(sumArray);
	}
}

namespace CharacterRecognition {

    using Common::PerformanceTimer;
    PerformanceTimer& timer()
    {
        static PerformanceTimer timer;
        return timer;
    }
        
	// Reference: https://solarianprogrammer.com/2012/05/31/matrix-multiplication-cuda-cublas-curand-thrust/
	// Matrix Multiplication
	// nr_rows_A, nr_cols_A, nr_cols_B
	void gpu_blas_mmul(const float *A, const float *B, float *C, const int nr_rows_A, const int nr_cols_A, const int nr_cols_B) {
		int lda = nr_rows_A, ldb = nr_cols_A, ldc = nr_rows_A;
		const float alf = 1;
		const float bet = 0;
		const float *alpha = &alf;
		const float *beta = &bet;
	    cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, nr_rows_A, nr_cols_B, nr_cols_A, alpha, A, lda, B, ldb, beta, C, ldc);
	}

	/* Forward Pass for one instance
	   1. Multiply input with input and hidden layer weights
	   2. Apply Sigmoid 
	   3. Multiply hidden layer activation with hidden and output layer weights
	   4. Apply Softmax and calculate ouput
	*/
	// TODO: Can Incorporate Bias
	void forwardPass(float* idata) {
		// Matrix Multiply Input Layer and Weights 1
		gpu_blas_mmul(idata, weights_IH, hidden, 1, INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE);
		//printf("Hidden After: ");
		//print_matrix(hidden, 1, HIDDEN_LAYER_SIZE);
		
		// Apply Sigmoid
		dim3 hiddenLayerBlocks((HIDDEN_LAYER_SIZE + blockSize - 1) / blockSize);
		Functions::reluActivation<<<hiddenLayerBlocks, blockSize>>>(hidden, h_sigmoid, 1, HIDDEN_LAYER_SIZE);
		//printf("Hidden Sigmoid After: ");
		//print_matrix(h_sigmoid, 1, HIDDEN_LAYER_SIZE);
		
		// Matrix Multiply Hidden layer and Weights 2
		gpu_blas_mmul(h_sigmoid, weights_HO, output, 1, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE);
		//printf("Output After: ");
		//print_matrix(output, 1, OUTPUT_LAYER_SIZE);
		
		// Apply Softmax
		dim3 outputLayerBlocks((OUTPUT_LAYER_SIZE + blockSize - 1) / blockSize);
		Functions::ExponentialActivation <<<outputLayerBlocks, blockSize >>> (output, o_softmax, 1, OUTPUT_LAYER_SIZE);
		float *sum;
		cudaMalloc((void**)&sum, sizeof(float));
		StreamCompaction::sumArray(OUTPUT_LAYER_SIZE, sum, o_softmax);
		Functions::Divide << <outputLayerBlocks, blockSize >> > (o_softmax, sum, 1, OUTPUT_LAYER_SIZE);
		printf("Output Probabilities: ");
		print_matrix(o_softmax, 1, OUTPUT_LAYER_SIZE);
	}

	void backwardPropagation(float* dev_input, float* dev_output, float* learning_rate) {
		// Memory Allocation
		float *temp_hidden, *temp_output;
		cudaMalloc((void**)&temp_output, OUTPUT_LAYER_SIZE * sizeof(float));
		cudaMalloc((void**)&temp_hidden, HIDDEN_LAYER_SIZE * sizeof(float));

		// Gradient of Loss w.r.t Weight2
		dim3 outputLayerBlocks((OUTPUT_LAYER_SIZE + blockSize - 1) / blockSize);
		Functions::ElementwiseSubtraction << <outputLayerBlocks, blockSize >> > (o_softmax, dev_output, temp_output, 1, OUTPUT_LAYER_SIZE);
		gpu_blas_mmul(h_sigmoid, temp_output, g_weights_HO, HIDDEN_LAYER_SIZE, 1, OUTPUT_LAYER_SIZE);

		// Gradient of Loss w.r.t Weight1
		gpu_blas_mmul(weights_HO, temp_output, temp_hidden, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE, 1);
		dim3 hiddenLayerBlocks((HIDDEN_LAYER_SIZE + blockSize - 1) / blockSize);
		Functions::KernelElementwiseMultiplyRelu << <outputLayerBlocks, blockSize >> > (temp_hidden, h_sigmoid, 1, HIDDEN_LAYER_SIZE);
		gpu_blas_mmul(dev_input, temp_hidden, g_weights_IH, INPUT_LAYER_SIZE, 1, HIDDEN_LAYER_SIZE);

		// Gradient Updates
		dim3 IHBlocks(((INPUT_LAYER_SIZE * HIDDEN_LAYER_SIZE) + blockSize - 1) / blockSize);
		Functions::Multiply << <IHBlocks, blockSize >> > (g_weights_IH, learning_rate, INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE);
		Functions::ElementwiseSubtraction << <IHBlocks, blockSize >> > (weights_IH, g_weights_IH, weights_IH, INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE);
		dim3 HOBlocks(((HIDDEN_LAYER_SIZE * OUTPUT_LAYER_SIZE) + blockSize - 1) / blockSize);
		Functions::Multiply << <HOBlocks, blockSize >> > (g_weights_HO, learning_rate, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE);
		Functions::ElementwiseSubtraction << <HOBlocks, blockSize >> > (weights_HO, g_weights_HO, weights_HO, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE);

		// Free Memory
		cudaFree(temp_hidden);
		cudaFree(temp_output);
	}

	float calculateLoss(int* label, int* prediction) {
		return -1;
	}

	void train(float* idata, float* ilabel, int num_instances, int epochs, float learning_rate) {
		// Create Device Buffers for Input and Output
		float *dev_input, *dev_output, *dev_lr;
		cudaMalloc((void**)&dev_input, num_instances * INPUT_LAYER_SIZE * sizeof(float));
		checkCUDAErrorFn("cudaMalloc dev_input failed!");
		cudaMemcpy(dev_input, idata, num_instances * INPUT_LAYER_SIZE * sizeof(float), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_output, num_instances * OUTPUT_LAYER_SIZE * sizeof(float));
		checkCUDAErrorFn("cudaMalloc dev_output failed!");
		cudaMemcpy(dev_output, ilabel, num_instances * OUTPUT_LAYER_SIZE * sizeof(float), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_lr, sizeof(float));
		thrust::device_ptr<float> dev_ptr(dev_lr);
		thrust::fill(dev_ptr, dev_ptr + 1, learning_rate);

		// Train
		for (int e = 0; e < epochs; e++) {
			for (int i = 0; i < num_instances; i++) {
				printf("Input: ");
				print_matrix(dev_input + (i * INPUT_LAYER_SIZE), 1, INPUT_LAYER_SIZE);
				// Forward Pass
				forwardPass(dev_input + (i * INPUT_LAYER_SIZE));
				// Back Propagation
				backwardPropagation(dev_input + (i * INPUT_LAYER_SIZE), dev_output + (i * OUTPUT_LAYER_SIZE), dev_lr);
			}
		}
	}

	void test(int* idata, int* ilabel, int* olabel) {

	}

	void init(int input_size, int hidden_size, int output_size) {
		printf("Init\n");
		// Initialize Layer Sizes
		INPUT_LAYER_SIZE = input_size;
		HIDDEN_LAYER_SIZE = hidden_size;
		OUTPUT_LAYER_SIZE = output_size;

		// Memory Allocation for Weight Matrices, Gradient Matrics and Hidden Layers
		cudaMalloc((void**)&weights_IH, INPUT_LAYER_SIZE * HIDDEN_LAYER_SIZE * sizeof(float));
		checkCUDAErrorFn("cudaMalloc weights_IH failed!");
		cudaMalloc((void**)&weights_HO, HIDDEN_LAYER_SIZE * OUTPUT_LAYER_SIZE * sizeof(float));
		checkCUDAErrorFn("cudaMalloc weights_HO failed!");
		cudaMalloc((void**)&g_weights_IH, INPUT_LAYER_SIZE * HIDDEN_LAYER_SIZE * sizeof(float));
		checkCUDAErrorFn("cudaMalloc g_weights_IH failed!");
		cudaMalloc((void**)&g_weights_HO, HIDDEN_LAYER_SIZE * OUTPUT_LAYER_SIZE * sizeof(float));
		checkCUDAErrorFn("cudaMalloc g_weights_HO failed!");
		cudaMalloc((void**)&hidden, HIDDEN_LAYER_SIZE * sizeof(float));
		checkCUDAErrorFn("cudaMalloc hidden failed!");
		cudaMalloc((void**)&h_sigmoid, HIDDEN_LAYER_SIZE * sizeof(float));
		checkCUDAErrorFn("cudaMalloc h_sigmoid failed!");
		cudaMalloc((void**)&output, OUTPUT_LAYER_SIZE * sizeof(float));
		checkCUDAErrorFn("cudaMalloc output failed!");
		cudaMalloc((void**)&o_softmax, OUTPUT_LAYER_SIZE * sizeof(float));
		checkCUDAErrorFn("cudaMalloc o_softmax failed!");

		// Create a handle for CUBLAS
		cublasCreate(&cublas_handle);
		// Curand Random Number Generator
		curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
		// Seed for Random Number Generator
		curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

	    // Initialize weight matrices with random numbers
		curandGenerateUniform(prng, weights_IH, INPUT_LAYER_SIZE * HIDDEN_LAYER_SIZE);
		curandGenerateUniform(prng, weights_HO, HIDDEN_LAYER_SIZE * OUTPUT_LAYER_SIZE);

		// Debug/Print
		print_matrix(weights_IH, INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE);
		print_matrix(weights_HO, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE);
	}

	void free() {
		cudaFree(weights_IH);
		cudaFree(weights_HO);
		cublasDestroy(cublas_handle);
	}
}
