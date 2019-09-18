#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "mlp.h"
#include <string>

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

	__global__ void backprop(float *inputLr, float *hiddenLr, float *outputLr, int n_input, int n_hidden, int n_output, float *weightsIH, float *weightsHO, float *d_weightsIH, float *d_weightsHO, float label) {
		int index = threadIdx.x + (blockIdx.x * blockDim.x);
		if (index >= n_input) {
			return;
		}

		for (int i = 0; i < n_hidden; i++) {
			float totalError = 0.0f;
			float gradientProd = 1.0f;
			float weightProd = 1.0f;
			for (int j = 0; j < n_output; j++) {
				float expected = label == j ? 1 : 0;
				float error = expected - outputLr[j];

				int weight2Index = i + j * n_output;
				d_weightsHO[weight2Index] += hiddenLr[i] * outputLr[j] * (1 - outputLr[j]) * -error;
				totalError += error;
				gradientProd *= outputLr[j] * (1 - outputLr[j]);
				weightProd *= weightsHO[weight2Index];
			}
			int weight1Index = index + i * n_hidden;
			d_weightsIH[weight1Index] += inputLr[index] * hiddenLr[i] * (1 - hiddenLr[i]) * -totalError * gradientProd * weightProd;
		}
	}

	__global__ void zeroBuffer(float *buffer, int n) {
		int index = threadIdx.x + (blockIdx.x * blockDim.x);
		if (index >= n) {
			return;
		}

		buffer[index] = 0;
	}

	__global__ void addTwoBuffers(float *addTo, float *addFrom, float lambda, int n) {
		int index = threadIdx.x + (blockIdx.x * blockDim.x);
		if (index >= n) {
			return;
		}

		addTo[index] += addFrom[index] * lambda;
	}
        
	void train(float lambda) {
		dim3 fullBlocksPerGrid((inputDims + blockSize - 1) / blockSize);
		dim3 fullBlocksPerGridInToHid((inputDims * hiddenDims + blockSize - 1) / blockSize);
		dim3 fullBlocksPerGridHidToOut((hiddenDims * outputDims + blockSize - 1) / blockSize);

		float totalError = 0;

		float *dev_d_w_kj;
		float *dev_d_w_ki;
		cudaMalloc((void**)&dev_d_w_kj, inputDims * hiddenDims * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_d_w_kj failed!");
		cudaMalloc((void**)&dev_d_w_ki, hiddenDims * outputDims * sizeof(float));
		checkCUDAErrorWithLine("cudaMalloc dev_d_w_ki failed!");

		for (int i = 0; i < LABELS; i++) {

			std::string filename = "../data-set/";
			std::string number = std::to_string(i + 1);
			if (number.length() == 1) {
				number = std::string("0").append(number);
			}
			filename.append(number);
			filename.append("info.txt");
			FILE * image = std::fopen(filename.c_str(), "r");
			int label;
			int dimensions;
			fscanf(image, "%d", &label);
			fscanf(image, "%d", &dimensions);
			float *colors = new float[dimensions];
			for (int j = 0; j < dimensions; j++) {
				int color;
				fscanf(image, "%d", &color);
				colors[j] = color;
			}

			float *output = new float[outputDims];
			evaluate(colors, output);
			backprop << <fullBlocksPerGrid, blockSize >> > (dev_input, dev_hidden, dev_output, inputDims, hiddenDims, outputDims, dev_w_kj, dev_w_ki, dev_d_w_kj, dev_d_w_ki, i);
			for (int j = 0; j < outputDims; j++) {
				float expected = i == j ? 1 : 0;
				float error = expected - output[j];
				totalError += error * error;
			}
			delete[] colors;
			delete[] output;
		}
		totalError /= 2.f;

		addTwoBuffers << <fullBlocksPerGridInToHid, blockSize >> > (dev_w_kj, dev_d_w_kj, -lambda * totalError, inputDims * hiddenDims);
		addTwoBuffers << <fullBlocksPerGridHidToOut, blockSize >> > (dev_w_ki, dev_d_w_ki, -lambda * totalError, hiddenDims * outputDims);

		cudaFree(dev_d_w_kj);
		cudaFree(dev_d_w_ki);

		printf("Total error is %f\n", totalError);
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

	void evaluate(float *input, float *output) {
		dim3 fullBlocksPerGridInToHid((inputDims * hiddenDims + blockSize - 1) / blockSize);
		dim3 fullBlocksPerGridHidToOut((hiddenDims * outputDims + blockSize - 1) / blockSize);
		dim3 fullBlocksPerGrid((inputDims + blockSize - 1) / blockSize);
		
		zeroBuffer << <fullBlocksPerGrid, blockSize >> > (dev_input, inputDims);
		zeroBuffer << <fullBlocksPerGrid, blockSize >> > (dev_hidden, hiddenDims);
		zeroBuffer << <fullBlocksPerGrid, blockSize >> > (dev_output, outputDims);

		cudaMemcpy(dev_input, input, sizeof(float) * inputDims, cudaMemcpyHostToDevice);

		kernComputeLayer << <fullBlocksPerGridInToHid, blockSize >> > (dev_input, dev_hidden, inputDims, hiddenDims, dev_w_kj);
		kernComputeLayer << <fullBlocksPerGridHidToOut, blockSize >> > (dev_hidden, dev_output, hiddenDims, outputDims, dev_w_ki);

		cudaMemcpy(output, dev_output, sizeof(float) * outputDims, cudaMemcpyDeviceToHost);
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
