#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "mlp.h"

#define DIM 101
#define LABELS 52

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

#define blockSize 128

namespace CharacterRecognition {
    using Common::PerformanceTimer;
    PerformanceTimer& timer()
    {
        static PerformanceTimer timer;
        return timer;
    }

	float *dev_input;
	float *dev_hidden;
	float *dev_output;
	float *dev_w_kj;
	float *dev_w_ki;

	int inputDims = DIM * DIM;
	int hiddenDims = inputDims;
	int outputDims = LABELS;

	__global__ void backprop() {
		int index = threadIdx.x + (blockIdx.x * blockDim.x);
		if (index >= n_output) {
			return;
		}
	}
        
	void train() {


		-PRODUCT(C7 - R9, (1 / (1 + EXP(-R9))), (1 - 1 / (1 + EXP(-R9))), O5)
		-PRODUCT(J5, 1 / (1 + EXP(-O5)), 1 - 1 / (1 + EXP(-O5)), C7 - R9, 1 / (1 + EXP(-R9)), 1 - 1 / (1 + EXP(-R9)), P8)
	}

	__global__ void kernComputeLayer(float *inputLr, float *outputLr, int n_input, int n_output, float *weights) {
		int index = threadIdx.x + (blockIdx.x * blockDim.x);
		if (index >= n_output) {
			return;
		}

		// Weighted sum of previous layer inputs
		for (int i = 0; i < n_input; i++) {
			int weightIndex = i + index * n_input;
			outputLr[index] += inputLr[i] * weights[weightIndex];
		}

		// Activation function
		outputLr[index] = 1 / (1 + expf(-outputLr[index]));
	}

	void printArray(const float *array, int n) {
		printf("[");
		for (int i = 0; i < n; i++) {
			printf("%f, ", array[i]);
		}
		printf("]\n");
	}

	void evaluate(float *input) {
		float *results = new float[outputDims];

		dim3 fullBlocksPerGridInToHid((inputDims * hiddenDims + blockSize - 1) / blockSize);
		dim3 fullBlocksPerGridHidToOut((hiddenDims * outputDims + blockSize - 1) / blockSize);

		cudaMemcpy(dev_input, input, sizeof(float) * inputDims, cudaMemcpyHostToDevice);

		kernComputeLayer << <fullBlocksPerGridInToHid, blockSize >> > (dev_input, dev_hidden, inputDims, hiddenDims, dev_w_kj);
		kernComputeLayer << <fullBlocksPerGridHidToOut, blockSize >> > (dev_hidden, dev_output, hiddenDims, outputDims, dev_w_ki);

		cudaMemcpy(results, dev_output, sizeof(float) * outputDims, cudaMemcpyDeviceToHost);

		printArray(results, outputDims);
		delete[] results;
	}

	void init() {
		dim3 fullBlocksPerGridInToHid((inputDims * hiddenDims + blockSize - 1) / blockSize);
		dim3 fullBlocksPerGridHidToOut((hiddenDims * outputDims + blockSize - 1) / blockSize);

		cudaMalloc((void**)&dev_input, inputDims * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_input failed!");

		cudaMalloc((void**)&dev_hidden, hiddenDims * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_hidden failed!");

		cudaMalloc((void**)&dev_output, outputDims * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_output failed!");

		cudaMalloc((void**)&dev_w_kj, inputDims * hiddenDims * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_w_kj failed!");

		cudaMalloc((void**)&dev_w_ki, hiddenDims * outputDims * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_w_ki failed!");

		float *weights1 = new float[inputDims * hiddenDims];
		for (int i = 0; i < inputDims * hiddenDims; i++) {
			float r = ((double)rand() / (RAND_MAX));
			weights1[i] = r * 2.0f - 1.0f;
		}
		float *weights2 = new float[hiddenDims * outputDims];
		for (int i = 0; i < hiddenDims * outputDims; i++) {
			float r = ((double)rand() / (RAND_MAX));
			weights2[i] = r * 2.0f - 1.0f;
		}
		cudaMemcpy(dev_w_kj, weights1, sizeof(float) * inputDims * hiddenDims, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_w_ki, weights2, sizeof(float) * hiddenDims * outputDims, cudaMemcpyHostToDevice);

		delete[] weights1;
		delete[] weights2;
	}

	void end() {
		cudaFree(dev_input);
		cudaFree(dev_hidden);
		cudaFree(dev_output);
		cudaFree(dev_w_kj);
		cudaFree(dev_w_ki);
	}
}
