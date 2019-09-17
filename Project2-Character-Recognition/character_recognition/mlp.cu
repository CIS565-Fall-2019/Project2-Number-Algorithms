#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "mlp.h"
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <thrust/random.h>

#define EPSILON 0.0005
#define MAX_ITER 1 << 10

namespace CharacterRecognition {
    using Common::PerformanceTimer;
    PerformanceTimer& timer()
    {
        static PerformanceTimer timer;
        return timer;
    }

	inline int ilog2(int x) {
		int lg = 0;
		while (x >>= 1) {
			++lg;
		}
		return lg;
	}

	inline int ilog2ceil(int x) {
		return x == 1 ? 0 : ilog2(x - 1) + 1;
	}

	__host__ __device__ unsigned int hash(unsigned int a) {
		a = (a + 0x7ed55d16) + (a << 12);
		a = (a ^ 0xc761c23c) ^ (a >> 19);
		a = (a + 0x165667b1) + (a << 5);
		a = (a + 0xd3a2646c) ^ (a << 9);
		a = (a + 0xfd7046c5) + (a << 3);
		a = (a ^ 0xb55a4f09) ^ (a >> 16);
		return a;
	}

	__host__ __device__ float generateRandom(float time, int index) {
		thrust::default_random_engine rng(hash((int)(index * time)));
		thrust::uniform_real_distribution<float> unitDistrib(0, 1);

		return (double)unitDistrib(rng);
	}

	// compute random float value between 0 and 1
	__global__ void kernRandom(int n, int time, double* out) {
		int index = (blockIdx.x * blockDim.x) + threadIdx.x;
		if (index >= n) {
			return;
		}

		out[index] = generateRandom(time, index);
	}

	__global__ void kernComputeLayerSum2(int n, int inCount, int outCount, float *in, float *out, float *weights) {
		int index = (blockIdx.x * blockDim.x) + threadIdx.x;
		if (index >= n) {
			return;
		}

		//int inIndex = floor(index / outCount);
		int inIndex = index / outCount;
		int outIndex = index - (inIndex * outCount);

		out[outIndex] += in[inIndex] * weights[index];
	}

	// per OUTPUT
	// numHidden, dev_hiddenLayer, numInput, dev_input, dev_weights, 0
	// 1,         dev_output,      numHidden, dev_hiddenLayer, dev_weights, numWeights1s
	__global__ void kernComputeLayerSum(int n, float *out, int inCount, float *in, double *weights, int weightsOffset) {
		int index = (blockIdx.x * blockDim.x) + threadIdx.x;
		if (index >= n) {
			return;
		}

		float sum = 0.0;
		for (int i = 0; i < inCount; i++) {
			sum += in[i] * weights[index + (i * n) + weightsOffset];
		}

		out[index] = sum;
	}
        
	__host__ __device__ float activationFxn(float x) {
		// activation function: f(x) = 1 / (1 + e^-x)
		return (1.0 / (1.0 + exp(-x)));
	}

	__global__ void kernComputeActivationFxn(int n, float *in) {
		int index = (blockIdx.x * blockDim.x) + threadIdx.x;
		if (index >= n) {
			return;
		}

		in[index] = activationFxn(in[index]);
	}

	__global__ void kernComputePartialDerivativeLayer1(int n, double *outDerivatives, double *weights,
		int numWeights1, float *input, int numHidden, float *hiddenOutput, float output, float expected) {
		int index = (blockIdx.x * blockDim.x) + threadIdx.x;
		if (index >= n) {
			return;
		}

		int inputIndex = index / numHidden;
		int hiddenIndex = index % numHidden;
		float hiddenActivation = activationFxn(hiddenOutput[hiddenIndex]);
		float outputActivation = activationFxn(output);

		float partialDerivative = -input[inputIndex] * hiddenActivation * (1.0 - hiddenActivation) *
			(expected - output) * outputActivation * (1.0 - outputActivation) * weights[numWeights1 + hiddenIndex];

		outDerivatives[index] = partialDerivative;
	}

	__global__ void kernComputePartialDerivativeLayer2(int n, double *outDerivatives, double *weights,
		int numWeights1, float *input, int numHidden, float *hiddenOutput, float output, float expected) {
		int index = (blockIdx.x * blockDim.x) + threadIdx.x;
		if (index >= n) {
			return;
		}

		float outputActivation = activationFxn(output);
		float partialDerivative = -(expected - output) * outputActivation * (1.0 - outputActivation) * hiddenOutput[index];

		outDerivatives[numWeights1 + index] = partialDerivative;
	}

	void getWeights(int numWeights, double *weights, bool training) {
		if (training) {
			double *dev_weights;
			cudaMalloc((void**)&dev_weights, numWeights * sizeof(double));
			checkCUDAError("cudaMalloc dev_weights failed!");

			// fill weights with random numbers between 0 and 1
			dim3 fullBlocksPerGrid((numWeights + blockSize - 1) / blockSize);
			kernRandom<<<fullBlocksPerGrid, blockSize>>>(numWeights, 1, dev_weights);
			checkCUDAError("kernRandom failed!");

			// copy weights back to host
			cudaMemcpy(weights, dev_weights, numWeights * sizeof(double), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy weights dev_weights failed!");

			// TODO: REMOVE LATER
			
			weights[0] = 10.1;
			weights[1] = 0.9;
			weights[2] = 20.0;
			weights[3] = 0.87;
			weights[4] = 41.0;
			weights[5] = -54.0;

			cudaFree(dev_weights);
			checkCUDAError("cudaFree dev_weights failed!");
		}
		else {
			// pull weights from txt file
			// TODO
		}
	}

	__global__ void kernScale(int n, double *buffer, float scale) {
		int index = (blockIdx.x * blockDim.x) + threadIdx.x;
		if (index >= n) {
			return;
		}

		buffer[index] = scale * buffer[index];
	}

	__global__ void kernScanUpsweep(int n, int iteration, double *buffer) {
		int index = (blockIdx.x * blockDim.x) + threadIdx.x;
		if (index >= n) {
			return;
		}

		int power1 = ceil(pow(2.0, double(iteration + 1)));
		if (index % power1 == 0) {
			int power2 = ceil(pow(2.0, double(iteration)));
			buffer[index + power1 - 1] += buffer[index + power2 - 1];
		}
	}

	// finds the next power of 2 greater than or equal to n
	int nextPowerOfTwo(int n) {
		if (n && !(n & (n - 1)))
			return n;

		int count = 0;
		while (n != 0) {
			n >>= 1;
			count++;
		}

		return 1 << count;
	}

	__global__ void kernModifyWeights(int n, double *weights, double *allWeightsErrors, int offset) {
		int index = (blockIdx.x * blockDim.x) + threadIdx.x;
		if (index >= n) {
			return;
		}
		weights[index] += allWeightsErrors[offset * index + (offset - 1)];
	}

	void adjustWeights(int numTotalInput, int numWeights, double *weights, std::vector<double*> allWeightsDerivatives, float error) {
		int numTotalPowerOf2 = nextPowerOfTwo(numTotalInput);

		// reorganize all weights derivatives to be in one buffer
		// buffer with zero's so it can be used in scan upsweep
		double *allWeightsDerivativesShuffled = new double[numTotalPowerOf2 * numWeights];
		for (int i = 0; i < numTotalInput; i++) {
			for (int j = 0; j < numWeights; j++) {
				allWeightsDerivativesShuffled[j * numTotalPowerOf2 + i] = allWeightsDerivatives[i][j];
			}
		}
		for (int i = numTotalInput; i < numTotalPowerOf2; i++) {
			for (int j = 0; j < numWeights; j++) {
				allWeightsDerivativesShuffled[j * numTotalPowerOf2 + i] = allWeightsDerivatives[i][j];
			}
		}


		double *dev_allWeightsDerivatives;
		cudaMalloc((void**)&dev_allWeightsDerivatives, (numTotalPowerOf2 * numWeights) * sizeof(double));
		checkCUDAError("cudaMalloc dev_allWeightsDerivatives failed!");

		cudaMemcpy(dev_allWeightsDerivatives, allWeightsDerivativesShuffled, (numTotalPowerOf2 * numWeights) * sizeof(double), cudaMemcpyHostToDevice);
		checkCUDAError("cudaMemcpy dev_allWeightsDerivatives allWeightsDerivativesShuffled failed!");

		double *dev_weights;
		cudaMalloc((void**)&dev_weights, numWeights * sizeof(double));
		checkCUDAError("cudaMalloc dev_weights failed!");

		cudaMemcpy(dev_weights, weights, numWeights * sizeof(double), cudaMemcpyHostToDevice);
		checkCUDAError("cudaMemcpy dev_weights weights failed!");

		// for each weight derivative, compute delta weight
		float lambda = -error / 5.0;
		dim3 gridSize = dim3(((numTotalPowerOf2 * numWeights) + blockSize - 1) / blockSize, 1, 1);
		kernScale<<<gridSize, blockSize>>>(numTotalPowerOf2 * numWeights, dev_allWeightsDerivatives, lambda);
		checkCUDAError("kernScale failed!");

		cudaMemcpy(allWeightsDerivativesShuffled, dev_allWeightsDerivatives, (numTotalPowerOf2 * numWeights) * sizeof(double), cudaMemcpyDeviceToHost);
		checkCUDAError("cudaMemcpy dev_allWeightsDerivatives allWeightsDerivativesShuffled failed!");

		// sum delta weights over all test input for a given weight
		gridSize = dim3(((numTotalPowerOf2 * numWeights) + blockSize - 1) / blockSize, 1, 1);
		for (int d = 0; d < ilog2ceil(numTotalPowerOf2); d++) {
			kernScanUpsweep<<<gridSize, blockSize >>>(numTotalPowerOf2 * numWeights, d, dev_allWeightsDerivatives);
			checkCUDAError("kernScanUpsweep failed!");
		}

		cudaMemcpy(allWeightsDerivativesShuffled, dev_allWeightsDerivatives, (numTotalPowerOf2 * numWeights) * sizeof(double), cudaMemcpyDeviceToHost);
		checkCUDAError("cudaMemcpy dev_allWeightsDerivatives allWeightsDerivativesShuffled failed!");

		// add delta weights to weights
		gridSize = dim3((numWeights + blockSize - 1) / blockSize, 1, 1);
		kernModifyWeights<<<gridSize, blockSize>>>(numWeights, dev_weights, dev_allWeightsDerivatives, numTotalPowerOf2);
		checkCUDAError("kernModifyWeights failed!");

		// copy weights back to host
		cudaMemcpy(weights, dev_weights, numWeights * sizeof(double), cudaMemcpyDeviceToHost);
		checkCUDAError("cudaMemcpy weights dev_weights failed!");

		cudaFree(dev_allWeightsDerivatives);
		cudaFree(dev_weights);
		checkCUDAError("cudaFree failed!");

	}

    /**
        * Example of use case (follow how you did it in stream compaction)
        */
    /*void scan(int n, int *odata, const int *idata) {
        timer().startGpuTimer();
		// TODO
		timer().endGpuTimer();
	}
	*/
float runOneInput(int numInput, int numHidden, const float *input, const double *weights, double *weightDerivatives, float expected, bool training) {
	int numWeights1 = numInput * numHidden;
	int numWeights2 = numHidden;

	float *dev_input;
	float *dev_hiddenLayer;
	float *dev_output;
	double *dev_weights;
	double *dev_weightDerivatives;

	// malloc device buffers
	cudaMalloc((void**)&dev_input, numInput * sizeof(float));
	checkCUDAError("cudaMalloc dev_input failed!");

	cudaMalloc((void**)&dev_hiddenLayer, numHidden * sizeof(float));
	checkCUDAError("cudaMalloc dev_hiddenLayer failed!");

	cudaMalloc((void**)&dev_output, numHidden * sizeof(float));
	checkCUDAError("cudaMalloc dev_output failed!");

	cudaMalloc((void**)&dev_weights, (numWeights1 + numWeights2) * sizeof(double));
	checkCUDAError("cudaMalloc dev_weights failed!");

	cudaMalloc((void**)&dev_weightDerivatives, (numWeights1 + numWeights2) * sizeof(double));
	checkCUDAError("cudaMalloc dev_weightsError failed!");

	// copy weights from host to device
	cudaMemcpy(dev_weights, weights, (numWeights1 + numWeights2) * sizeof(double), cudaMemcpyHostToDevice);
	checkCUDAError("cudaMemcpy dev_weights weights failed!");

	// copy input from host to device
	cudaMemcpy(dev_input, input, numInput * sizeof(float), cudaMemcpyHostToDevice);
	checkCUDAError("cudaMemcpy dev_input input failed!");

	// compute first layer summation
	dim3 gridSize = dim3((numHidden + blockSize - 1) / blockSize, 1, 1);
	kernComputeLayerSum<<<gridSize, blockSize>>>(numHidden, dev_hiddenLayer, numInput, dev_input, dev_weights, 0);
	checkCUDAError("kernComputeLayerSum failed!");

	// DEBUG
	float *test = new float[6];
	cudaMemcpy(test, dev_hiddenLayer, numHidden * sizeof(float), cudaMemcpyDeviceToHost);

	// compute result of hidden layer (activation function)
	gridSize = dim3((numHidden + blockSize - 1) / blockSize, 1, 1);
	kernComputeActivationFxn<<<gridSize, blockSize>>>(numHidden, dev_hiddenLayer);
	checkCUDAError("kernComputeActivationFxn failed!");

	// DEBUG
	cudaMemcpy(test, dev_hiddenLayer, numHidden * sizeof(float), cudaMemcpyDeviceToHost);

	// compute second layer summation
	// TODO this is one thread
	gridSize = dim3((1 + blockSize - 1) / blockSize, 1, 1);
	kernComputeLayerSum<<<gridSize, blockSize>>>(1, dev_output, numHidden, dev_hiddenLayer, dev_weights, numWeights1);
	checkCUDAError("kernComputeLayerSum failed!");

	cudaMemcpy(test, dev_output, 1 * sizeof(float), cudaMemcpyDeviceToHost);

	// compute activation function of output layer node
	float output;
	cudaMemcpy(&output, dev_output, 1 * sizeof(float), cudaMemcpyDeviceToHost);
	output = activationFxn(output);

	// if training, compute partial derivatives for error/weight
	if (training) {
		// first layer weights
		gridSize = dim3((numWeights1 + blockSize - 1) / blockSize, 1, 1);
		kernComputePartialDerivativeLayer1<<<gridSize, blockSize>>>(numWeights1, dev_weightDerivatives, dev_weights, numWeights1, 
			dev_input, numHidden, dev_hiddenLayer, output, expected);
		checkCUDAError("kernComputePartialDerivativeLayer1 failed!");

		// second layer weights
		gridSize = dim3((numWeights2 + blockSize - 1) / blockSize, 1, 1);
		kernComputePartialDerivativeLayer2<<<gridSize, blockSize>>>(numWeights2, dev_weightDerivatives, dev_weights, numWeights1,
			dev_input, numHidden, dev_hiddenLayer, output, expected);
		checkCUDAError("kernComputePartialDerivativeLayer2 failed!");

		// copy derivatives to host
		cudaMemcpy(weightDerivatives, dev_weightDerivatives, (numWeights1 + numWeights2) * sizeof(double), cudaMemcpyDeviceToHost);
		checkCUDAError("cudaMemcpy weightDerivatives dev_weightDerivatives failed!");
	}

	cudaFree(dev_input);
	cudaFree(dev_hiddenLayer);
	cudaFree(dev_output);
	cudaFree(dev_weights);
	cudaFree(dev_weightDerivatives);
	checkCUDAError("cudaFree failed!");

	// return square of difference
	return (output - expected) * (output - expected);
}

void train(int numInput, std::vector<float*> inputs, std::vector<int> expected) {
	int numTotalInput = inputs.size();

	// determine how many nodes in hidden layer (average of # of nodes in input and # of nodes in output)
	int numHidden = ceil((numInput + 1.0) / 2.0);

	// determine number of weights in all layers
	int numWeights = (numInput * numHidden) + numHidden;

	// get weights
	double *weights = new double[numWeights];
	getWeights(numWeights, weights, true);

	// create weight errors 
	std::vector<double*> allWeightErrors;
	for (int i = 0; i < numTotalInput; i++) {
		double *weightErrors = new double[numWeights];
		allWeightErrors.push_back(weightErrors);
	}

	float totalError;
	int numIter = 0;

	while (true) {
		totalError = 0.f;
		// train on each input
		for (int i = 0; i < numTotalInput; i++) {
			float *in = inputs[i];
			totalError += runOneInput(numInput, numHidden, in, weights, allWeightErrors[i], expected[i], true);
		}
		totalError /= 2.f;

		if (totalError < EPSILON || numIter > MAX_ITER) {
			// finished training
			// save weights to txt file
			std::ofstream weightsFile;
			weightsFile.open("../weights.txt");
			for (int i = 0; i < numWeights; i++) {
				float weight = weights[i];
				weightsFile << weight;
				weightsFile << "\n";
			}
			weightsFile.close();
			break; // break while loop
		}
		else {
			adjustWeights(numTotalInput, numWeights, weights, allWeightErrors, totalError);
		}
		numIter++;
	}

	delete[] weights;
		
	for (int i = 0; i < numTotalInput; i++) {
		delete[] allWeightErrors[i];
	}
}

	void run() {
		// get input
		std::vector<float*> inputs;
		std::vector<float> expected;
		int numInput;

		// determine how many nodes in hidden layer (average of # of nodes in input and # of nodes in output)
		int numHidden = (numInput + 1.0) / 2.0;

		// determine number of weights in all layers
		int numWeights = (numInput * numHidden) + numHidden;

		// get weights
		double* weights = new double[numWeights];
		getWeights(numWeights, weights, false);

		float totalError = 0.f;
		for (int i = 0; i < inputs.size(); i++) {
			float *in = inputs[i];
			double* weightDerivatives = new double[0];
			totalError += runOneInput(numInput, numHidden, in, weights, weightDerivatives, expected[i], false);
		}

		delete(weights);
	}

	// TODO: implement required elements for MLP sections 1 and 2 here
}
