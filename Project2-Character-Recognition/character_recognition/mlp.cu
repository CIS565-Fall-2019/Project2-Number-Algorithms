#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "mlp.h"
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/random.h>

namespace CharacterRecognition {
    using Common::PerformanceTimer;
    PerformanceTimer& timer()
    {
        static PerformanceTimer timer;
        return timer;
    }

	int blockSize = 128;
	dim3 threadsPerBlock(blockSize);

	__host__ __device__ unsigned int hash(unsigned int a) {
		a = (a + 0x7ed55d16) + (a << 12);
		a = (a ^ 0xc761c23c) ^ (a >> 19);
		a = (a + 0x165667b1) + (a << 5);
		a = (a + 0xd3a2646c) ^ (a << 9);
		a = (a + 0xfd7046c5) + (a << 3);
		a = (a ^ 0xb55a4f09) ^ (a >> 16);
		return a;
	}

	__global__ void kernFillRandom(int N, float *weights, float time) {
		int index = (blockIdx.x * blockDim.x) + threadIdx.x;
		if (index >= N) {
			return;
		}

		
		thrust::default_random_engine rng(hash((int)(index * time)));
		thrust::uniform_real_distribution<float> unitDistrib(-12, 12);


		weights[index] = (float)unitDistrib(rng);
	}


	void fillRandomWeights(int n, float *data, float seed) {
		float *dev_weightsArray;

		cudaMalloc((void**)&dev_weightsArray, n * sizeof(float));
		checkCUDAError("cudaMalloc dev_weightsArray failed!");

		int numThreads = n;
		dim3 blocksPerGrid((numThreads + blockSize - 1) / blockSize);

		kernFillRandom<<<blocksPerGrid, threadsPerBlock>>>(numThreads, dev_weightsArray, seed);
		checkCUDAError("kernFillRandom failed!");

		cudaMemcpy(data, dev_weightsArray, n * sizeof(float), cudaMemcpyDeviceToHost);

		cudaFree(dev_weightsArray);
	}


	__global__ void kernLayer1Mult(int numHidden, float *hiddenLayers, int inputSize, const float* input, const float *weights) {
		int index = (blockIdx.x * blockDim.x) + threadIdx.x;
		if (index >= numHidden) {
			return;
		}
		float sum = 0;
		for (int i = 0; i < inputSize; ++i) {
			sum += input[i] * weights[index + numHidden * i];
		}

		hiddenLayers[index] = 1 / (1 + exp(-sum));
	}

	__global__ void kernLayer2Mult(int n, int numHiddenlayers, float *output, const float *hiddenLayers, const float *weights) {
		int index = (blockIdx.x * blockDim.x) + threadIdx.x;
		if (index >= n) {
			return;
		}
		float sum = 0;
		for (int i = 0; i < numHiddenlayers; ++i) {
			sum += hiddenLayers[i] * weights[i];
		}
		output[index] = 1 / (1 + exp(-sum));
	}

	__global__ void kernPartialErrorDeriv1(int n,
		float expectedValue, float output, int inputSize, int numHidden,
		const float *input, const float *hidden, const float *weights2, float *partials1) {
		
		int index = (blockIdx.x * blockDim.x) + threadIdx.x;
		if (index >= n) {
			return;
		}

		//float originalWeight = adjustedWeights[index]; // Do the memory acces first and let the following math hide latency

		int inputIndex = floorf(index / (numHidden));
		int hiddenIndex = index % numHidden;

		float inputValue = input[inputIndex];
		float hiddenValue = hidden[hiddenIndex];
		float hiddenWeight = weights2[hiddenIndex];

		float partialErrorDeriv = -inputValue * (1 / (1 + exp(-hiddenValue))) *
			(1 - (1 / (1 + exp(-hiddenValue)))) * (expectedValue - output) *
			(1 / (1 + exp(-output))) * (1 - (1 / (1 + exp(-output)))) *
			hiddenWeight;

		//float deltaWeight = (error / 10.0) * partialErrorDeriv;

		partials1[index] = partialErrorDeriv;
	}

	__global__ void kernPartialErrorDeriv2(int n,
		float expectedValue, float output,
		const float *hidden, float *partials2) {

		int index = (blockIdx.x * blockDim.x) + threadIdx.x;
		if (index >= n) {
			return;
		}

		//float originalWeight = adjustedWeights[index];

		float partialErrorDeriv = (-(expectedValue - output)) * (1 / (1 + exp(-output))) * (1 - (1 / (1 + exp(-output)))) * hidden[index];

		//float deltaWeight = (error / 10.0) * partialErrorDeriv;

		partials2[index] =  partialErrorDeriv;



	}


	float mlp(int inputSize, int numHiddenLayers, float expectedValue, 
		const float *weights1, const float *weights2, 
		const float *idata, 
		float *partialDerivatives1, float *partialDerivatives2) {
		// size of input is 2 for xor and 512 by 512 for characters
		// hidden layer somewhere between 1 and size of input
		// first number of weights is size of hidden layer * size of input
		// second number of weights is size of hidden layer * size of output(1)

		int numWeights1 = inputSize * numHiddenLayers;
		int numWeights2 = numHiddenLayers;


		// Initialize buffers
		float *dev_inputData;
		float *dev_hidden;
		float *dev_weights1;
		float *dev_weights2;
		float *dev_output;

		float *dev_partials1;
		float *dev_partials2;

		float *host_output;

		float *host_hidden;

		// Malloc for buffers
		cudaMalloc((void**)&dev_inputData, inputSize * sizeof(float));
		checkCUDAError("cudaMalloc dev_inputData failed!");
		
		cudaMalloc((void**)&dev_hidden, numHiddenLayers * sizeof(float));
		checkCUDAError("cudaMalloc dev_hidden failed!");

		cudaMallocHost((void**)&host_hidden, numHiddenLayers * sizeof(float));
		checkCUDAError("cudaMallocHost host_hidden failed!");
		
		cudaMalloc((void**)&dev_weights1, numWeights1 * sizeof(float));
		checkCUDAError("cudaMalloc dev_weights1 failed!");
		
		cudaMalloc((void**)&dev_weights2, numWeights2 * sizeof(float));
		checkCUDAError("cudaMalloc dev_weights2 failed!");

		cudaMalloc((void**)&dev_partials1, numWeights1 * sizeof(float));
		checkCUDAError("cudaMalloc dev_partials1 failed!");

		cudaMalloc((void**)&dev_partials2, numWeights2 * sizeof(float));
		checkCUDAError("cudaMalloc dev_partials2 failed!");

		cudaMalloc((void**)&dev_output, sizeof(float));
		checkCUDAError("cudaMalloc dev_output failed!");

		cudaMallocHost((void**)&host_output, sizeof(float));
		checkCUDAError("cudaMallocHost host_output failed!");

		// Fille input and weights data
		cudaMemcpy(dev_inputData, idata, inputSize * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_weights1, weights1, numWeights1 * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_weights2, weights2, numWeights2 * sizeof(float), cudaMemcpyHostToDevice);
		
		// Perform the multiplications for layer 1 to get the hidden layers
		int numThreads = numHiddenLayers;
		dim3 blocksPerGrid((numThreads + blockSize - 1) / blockSize);
		
		kernLayer1Mult<<<blocksPerGrid, threadsPerBlock>>>(numHiddenLayers, dev_hidden, inputSize, dev_inputData, dev_weights1);
	
		// perform the multiplications for layer 2 to get the output value
		int layer2_numThreads = 1;
		dim3 layer2_blocksPerGrid((layer2_numThreads + blockSize - 1) / blockSize);

		kernLayer2Mult<<<layer2_blocksPerGrid, threadsPerBlock>>>(1, numHiddenLayers, dev_output, dev_hidden, dev_weights2);

		// Copy the output onto the host
		cudaMemcpy(host_output, dev_output, sizeof(float), cudaMemcpyDeviceToHost);
		float output = host_output[0];

		// Find the error from the output
		//float error = (output - expectedValue) * (output - expectedValue);
		//std::cout << "error " << error << std::endl;

		// Adjust the weights of the layer 1 weights
		int weight1Adjust_numThreads = numWeights1;
		dim3 weight1Adjust_blocksPerGrid((weight1Adjust_numThreads + blockSize - 1) / blockSize);

		kernPartialErrorDeriv1<<<weight1Adjust_blocksPerGrid, threadsPerBlock>>>(numWeights1, expectedValue, 
			output, inputSize,
			numHiddenLayers, dev_inputData, dev_hidden, 
			dev_weights2, dev_partials1);

		// Copy the weights into the input array
		cudaMemcpy(partialDerivatives1, dev_partials1, numWeights1 * sizeof(float), cudaMemcpyDeviceToHost);


		// Adjust the weights of the layer 2 weights
		int weight2Adjust_numThreads = numWeights2;
		dim3 weight2Adjust_blocksPerGrid((weight2Adjust_numThreads + blockSize - 1) / blockSize);

		kernPartialErrorDeriv2<<<weight2Adjust_blocksPerGrid, threadsPerBlock>>>(numWeights2, 
			expectedValue, output, dev_hidden, dev_partials2);

		cudaMemcpy(partialDerivatives2, dev_partials2, numWeights2 * sizeof(float), cudaMemcpyDeviceToHost);

		//for (int i = 0; i < numWeights1; ++i) {
		//	//std::cout << "adjusted weight: " << adjustedWeights1[i] << std::endl;
		//}


		// Free buffer memory
		cudaFree(dev_inputData);
		cudaFree(dev_hidden);
		cudaFree(dev_weights1);
		cudaFree(dev_weights2);
		cudaFree(dev_partials1);
		cudaFree(dev_partials2);
		cudaFree(dev_output);
		cudaFreeHost(host_output);
		cudaFreeHost(host_hidden);


		return output;

	}


	__global__ void kernAddDelta(int n, float accumulatedError, const float *partials,
		float *weights) {
		int index = (blockIdx.x * blockDim.x) + threadIdx.x;
		if (index >= n) {
			return;
		}

		float delta = -(accumulatedError / 5.0) * partials[index];
		weights[index] += delta;
	}

	void updateWeights(int numWeights, float accumulatedError, const float *partials, float *weights) {
		float *dev_partials;
		float *dev_weights;

		cudaMalloc((void**)&dev_partials, numWeights * sizeof(float));
		checkCUDAError("cudaMalloc dev_partials failed!");
		
		cudaMalloc((void**)&dev_weights, numWeights * sizeof(float));
		checkCUDAError("cudaMalloc dev_weights failed!");

		cudaMemcpy(dev_partials, partials, numWeights * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_weights, weights, numWeights * sizeof(float), cudaMemcpyHostToDevice);


		int numThreads = numWeights;
		dim3 blocksPerGrid((numThreads + blockSize - 1) / blockSize);

		kernAddDelta<<<blocksPerGrid, threadsPerBlock>>>(numWeights, accumulatedError, dev_partials, dev_weights);

		cudaMemcpy(weights, dev_weights, numWeights * sizeof(float), cudaMemcpyDeviceToHost);

		cudaFree(dev_weights);
		cudaFree(dev_partials);


	}

}
