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

	int blockSize = 128;
	dim3 threadsPerBlock(blockSize);

	/*__host__ __device__ glm::vec3 generateRandomVec3(float time, int index) {
		thrust::default_random_engine rng(hash((int)(index * time)));
		thrust::uniform_real_distribution<float> unitDistrib(-1, 1);

		return glm::vec3((float)unitDistrib(rng), (float)unitDistrib(rng), (float)unitDistrib(rng));
	}*/

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
		thrust::uniform_real_distribution<float> unitDistrib(-50, 50);


		weights[index] = (float)unitDistrib(rng);
	}


	void fillRandomWeights(int n, float *data) {
		float *dev_weightsArray;

		cudaMalloc((void**)&dev_weightsArray, n * sizeof(float));
		checkCUDAError("cudaMalloc dev_weightsArray failed!");

		int numThreads = n;
		dim3 blocksPerGrid((numThreads + blockSize - 1) / blockSize);

		kernFillRandom<<<blocksPerGrid, threadsPerBlock>>>(numThreads, dev_weightsArray, 2);
		checkCUDAError("kernFillRandom failed!");

		cudaMemcpy(data, dev_weightsArray, n * sizeof(float), cudaMemcpyDeviceToHost);

		cudaFree(dev_weightsArray);
	}

	void updateWeights(int n, int *input, float *weights, const float *patialErrorDeriv, float error) {

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


	float mlp(int inputSize, int numHiddenLayers, float expectedValue, 
		const float *weights1, const float *weights2, 
		const float *idata, 
		float *adjustedWeights1, float *adjustedWeights2) {
		// size of input is 2 for xor and 512 by 512 for characters
		// hidden layer somewhere between 1 and size of input
		// first number of weights is size of hidden layer * size of input
		// second number of weights is size of hidden layer * size of output(1)

		int numWeights1 = inputSize * numHiddenLayers;
		int numWeights2 = numHiddenLayers;

		float *dev_inputData;
		float *dev_hidden;
		float *dev_weights1;
		float *dev_weights2;
		float *dev_output;

		float *host_output;

		float *host_hidden;


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

		cudaMalloc((void**)&dev_output, sizeof(float));
		checkCUDAError("cudaMalloc dev_output failed!");

		cudaMallocHost((void**)&host_output, sizeof(float));
		checkCUDAError("cudaMallocHost host_output failed!");

		cudaMemcpy(dev_inputData, idata, inputSize * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_weights1, weights1, numWeights1 * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_weights2, weights2, numWeights2 * sizeof(float), cudaMemcpyHostToDevice);



		int numThreads = numHiddenLayers;
		dim3 blocksPerGrid((numThreads + blockSize - 1) / blockSize);
		
		kernLayer1Mult<<<blocksPerGrid, threadsPerBlock>>>(numHiddenLayers, dev_hidden, inputSize, dev_inputData, dev_weights1);
	

		int layer2_numThreads = 1;
		dim3 layer2_blocksPerGrid((layer2_numThreads + blockSize - 1) / blockSize);


		kernLayer2Mult<<<blocksPerGrid, threadsPerBlock>>>(1, numHiddenLayers, dev_output, dev_hidden, dev_weights2);		

		cudaMemcpy(host_output, dev_output, sizeof(float), cudaMemcpyDeviceToHost);
		float output = host_output[0];

		float error = (output - expectedValue) * (output - expectedValue);
		std::cout << "error " << error << std::endl;



		cudaFree(dev_inputData);
		cudaFree(dev_hidden);
		cudaFree(dev_weights1);
		cudaFree(dev_weights2);
		cudaFree(dev_output);
		cudaFreeHost(host_output);
		cudaFreeHost(host_hidden);


		return output;

	}

	// TODO: implement required elements for MLP sections 1 and 2 here


}
