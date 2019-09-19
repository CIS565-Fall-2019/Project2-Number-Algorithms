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
#include <cstdio>
#include <thrust/random.h>

#define EPSILON 0.0005
#define MAX_ITER 1 << 5

#define XOR_HARD_CODED_WEIGHTS 1

namespace CharacterRecognition {
    using Common::PerformanceTimer;
    PerformanceTimer& timer()
    {
        static PerformanceTimer timer;
        return timer;
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


	__host__ __device__ float generateRandom(float time, int index, float range) {
		thrust::default_random_engine rng(hash((int)(index * time)));
		thrust::uniform_real_distribution<float> unitDistrib(-1, 1);

		return range * unitDistrib(rng);
	}


	// compute random float value between 0 and 1
	__global__ void kernRandom(int n, int time, float* out) {
		int index = (blockIdx.x * blockDim.x) + threadIdx.x;
		if (index >= n) {
			return;
		}

		out[index] = generateRandom(time, index, 10.0);
	}


	__global__ void kernComputeLayerSum(int n, float *out, int inCount, float *in, float *weights, int weightsOffset) {
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


	__global__ void kernComputePartialDerivativeLayer1(int n, float *outDerivatives, float *weights,
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


	__global__ void kernComputePartialDerivativeLayer2(int n, float *outDerivatives, float *weights,
		int numWeights1, float *input, int numHidden, float *hiddenOutput, float output, float expected) {
		int index = (blockIdx.x * blockDim.x) + threadIdx.x;
		if (index >= n) {
			return;
		}

		float outputActivation = activationFxn(output);
		float partialDerivative = -(expected - output) * outputActivation * (1.0 - outputActivation) * hiddenOutput[index];

		outDerivatives[numWeights1 + index] = partialDerivative;
	}


	void createWeights(int numWeights, float *weights) {
		float *dev_weights;
		cudaMalloc((void**)&dev_weights, numWeights * sizeof(float));
		checkCUDAError("cudaMalloc dev_weights failed!");

		// fill weights with random numbers between 0 and 1
		dim3 gridSize = dim3((numWeights + blockSize - 1) / blockSize, 1, 1);
		kernRandom<<<gridSize, blockSize>>>(numWeights, 1, dev_weights);
		checkCUDAError("kernRandom failed!");

		// copy weights back to host
		cudaMemcpy(weights, dev_weights, numWeights * sizeof(float), cudaMemcpyDeviceToHost);
		checkCUDAError("cudaMemcpy weights dev_weights failed!");
			
#if XOR_HARD_CODED_WEIGHTS
		weights[0] = 10.1;
		weights[1] = 0.9;
		weights[2] = 20.0;
		weights[3] = 0.87;
		weights[4] = 41.0;
		weights[5] = -54.0;
#endif // #if XOR_HARD_CODED_WEIGHTS
			
		cudaFree(dev_weights);
		checkCUDAError("cudaFree dev_weights failed!");
	}


	void getWeightsFromFile(int numWeights, float *weights, std::string filename) {
		std::string prefix = "../weights_";
		std::string suffix = ".txt";
		std::stringstream buffer;
		buffer << prefix << filename << suffix;

		int index = 0;
		std::ifstream inputFile(buffer.str());
		if (inputFile.is_open()) {
			std::string line;
			while (std::getline(inputFile, line)) {
				weights[index] = stof(line);
				index++;
			}
		}
	}


	__global__ void kernScale(int n, float *buffer, float scale) {
		int index = (blockIdx.x * blockDim.x) + threadIdx.x;
		if (index >= n) {
			return;
		}

		buffer[index] = scale * buffer[index];
	}


	__global__ void kernScanUpsweep(int n, int iteration, float *buffer) {
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


	__global__ void kernModifyWeights(int n, float *weights, float *allWeightsErrors, int numTotalInput) {
		int index = (blockIdx.x * blockDim.x) + threadIdx.x;
		if (index >= n) {
			return;
		}

		for (int i = 0; i < numTotalInput; i++) {
			weights[index] += allWeightsErrors[i * n + index];
		}
	}

	__global__ void kernModifyWeights2(int n, float *weights, float *weightsErrors, int numTotalInput, int currWeight) {
		int index = (blockIdx.x * blockDim.x) + threadIdx.x;
		if (index >= n) {
			return;
		}

		for (int i = 0; i < numTotalInput; i++) {
			weights[currWeight] += weightsErrors[i];
		}
	}

	__global__ void kernModifyWeights(int n, float *weights, float *weightsErrors) {
		int index = (blockIdx.x * blockDim.x) + threadIdx.x;
		if (index >= n) {
			return;
		}

		weights[index] += weightsErrors[index];
	}


	void adjustWeights(int numTotalInput, int numWeights, float *weights, std::vector<float*> allWeightsDerivatives, float error) {
		float lambda = -error / 5.0;

		float *dev_weights;
		cudaMalloc((void**)&dev_weights, numWeights * sizeof(float));
		checkCUDAError("cudaMalloc dev_weights failed!");

		cudaMemcpy(dev_weights, weights, numWeights * sizeof(float), cudaMemcpyHostToDevice);
		checkCUDAError("cudaMemcpy dev_weights weights failed!");

		float *dev_weightErrors;
		cudaMalloc((void**)&dev_weightErrors, numWeights * sizeof(float));
		checkCUDAError("cudaMalloc dev_weightErrors failed!");

		for (int i = 0; i < allWeightsDerivatives.size(); i++) {
			cudaMemcpy(dev_weightErrors, allWeightsDerivatives[i], numWeights * sizeof(float), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy dev_weightErrors allWeightsDerivatives[i] failed!");

			// for each weight derivative, compute delta weight
			dim3 gridSize = dim3((numWeights + blockSize - 1) / blockSize, 1, 1);
			kernScale<<<gridSize, blockSize >>>(numWeights, dev_weightErrors, lambda);
			checkCUDAError("kernScale failed!");

			kernModifyWeights<<<gridSize, blockSize>>>(numWeights, dev_weights, dev_weightErrors);
			checkCUDAError("kernModifyWeights failed!");
		}

		// copy weights back to host
		cudaMemcpy(weights, dev_weights, numWeights * sizeof(float), cudaMemcpyDeviceToHost);
		checkCUDAError("cudaMemcpy weights dev_weights failed!");

		cudaFree(dev_weightErrors);
		cudaFree(dev_weights);
		checkCUDAError("cudaFree failed!");
	}
      

	float runOneInput(int numInput, int numHidden, const float *input, const float *weights, float *weightErrors, float expected, bool training) {
		int numWeights1 = numInput * numHidden;
		int numWeights2 = numHidden;

		float *dev_input;
		float *dev_hiddenLayer;
		float *dev_output;
		float *dev_weights;
		float *dev_weightErrors;

		// malloc device buffers
		cudaMalloc((void**)&dev_input, numInput * sizeof(float));
		checkCUDAError("cudaMalloc dev_input failed!");

		cudaMalloc((void**)&dev_hiddenLayer, numHidden * sizeof(float));
		checkCUDAError("cudaMalloc dev_hiddenLayer failed!");

		cudaMalloc((void**)&dev_output, numHidden * sizeof(float));
		checkCUDAError("cudaMalloc dev_output failed!");

		cudaMalloc((void**)&dev_weights, (numWeights1 + numWeights2) * sizeof(float));
		checkCUDAError("cudaMalloc dev_weights failed!");

		cudaMalloc((void**)&dev_weightErrors, (numWeights1 + numWeights2) * sizeof(float));
		checkCUDAError("cudaMalloc dev_weightsError failed!");

		// copy weights from host to device
		cudaMemcpy(dev_weights, weights, (numWeights1 + numWeights2) * sizeof(float), cudaMemcpyHostToDevice);
		checkCUDAError("cudaMemcpy dev_weights weights failed!");

		// copy input from host to device
		cudaMemcpy(dev_input, input, numInput * sizeof(float), cudaMemcpyHostToDevice);
		checkCUDAError("cudaMemcpy dev_input input failed!");

		// compute first layer summation
		dim3 gridSize = dim3((numHidden + blockSize - 1) / blockSize, 1, 1);
		kernComputeLayerSum<<<gridSize, blockSize>>>(numHidden, dev_hiddenLayer, numInput, dev_input, dev_weights, 0);
		checkCUDAError("kernComputeLayerSum failed!");

		// compute result of hidden layer (activation function)
		gridSize = dim3((numHidden + blockSize - 1) / blockSize, 1, 1);
		kernComputeActivationFxn<<<gridSize, blockSize>>>(numHidden, dev_hiddenLayer);
		checkCUDAError("kernComputeActivationFxn failed!");

		// compute second layer summation (this is one thread) TODO hm
		gridSize = dim3((1 + blockSize - 1) / blockSize, 1, 1);
		kernComputeLayerSum<<<gridSize, blockSize>>>(1, dev_output, numHidden, dev_hiddenLayer, dev_weights, numWeights1);
		checkCUDAError("kernComputeLayerSum failed!");

		// compute activation function of output layer node
		float output;
		cudaMemcpy(&output, dev_output, 1 * sizeof(float), cudaMemcpyDeviceToHost);
		output = activationFxn(output);

		if (!training) {
			std::cout << "Expected output: " << std::to_string(expected) << std::endl;
			std::cout << "Actual output:   " << std::to_string(output) << std::endl;
			std::cout << std::endl;
		}

		// if training, compute partial derivatives for error/weight
		if (training) {
			// first layer weights
			gridSize = dim3((numWeights1 + blockSize - 1) / blockSize, 1, 1);
			kernComputePartialDerivativeLayer1<<<gridSize, blockSize>>>(numWeights1, dev_weightErrors, dev_weights, numWeights1,
				dev_input, numHidden, dev_hiddenLayer, output, expected);
			checkCUDAError("kernComputePartialDerivativeLayer1 failed!");

			// second layer weights
			gridSize = dim3((numWeights2 + blockSize - 1) / blockSize, 1, 1);
			kernComputePartialDerivativeLayer2<<<gridSize, blockSize>>>(numWeights2, dev_weightErrors, dev_weights, numWeights1,
				dev_input, numHidden, dev_hiddenLayer, output, expected);
			checkCUDAError("kernComputePartialDerivativeLayer2 failed!");

			// copy derivatives to host
			cudaMemcpy(weightErrors, dev_weightErrors, (numWeights1 + numWeights2) * sizeof(float), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy weightErrors dev_weightErrors failed!");
		}

		cudaFree(dev_input);
		cudaFree(dev_hiddenLayer);
		cudaFree(dev_output);
		cudaFree(dev_weights);
		cudaFree(dev_weightErrors);
		checkCUDAError("cudaFree failed!");

		// return square of difference
		return (output - expected) * (output - expected);
	}

	void train(int numInput, std::vector<float*> inputs, std::vector<float> expected, std::string filename) {
		int numTotalInput = inputs.size();

		// determine how many nodes in hidden layer (average of # of nodes in input and # of nodes in output)
		int numHidden = ceil((numInput + 1.0) / 2.0);

		// determine number of weights in all layers
		int numWeights = (numInput * numHidden) + numHidden;

		// get weights
		float *weights = new float[numWeights];
		createWeights(numWeights, weights);

		// create weight errors 
		std::vector<float*> allWeightErrors;
		for (int i = 0; i < numTotalInput; i++) {
			allWeightErrors.push_back(new float[numWeights]);
		}

		float totalError;
		int numIter = 0;

		while (true) {
			totalError = 0.0;
			// train on each input
			for (int i = 0; i < numTotalInput; i++) {
				float *in = inputs[i];
				totalError += runOneInput(numInput, numHidden, in, weights, allWeightErrors[i], expected[i], true);
			}
			totalError /= 2.0;

			if (totalError < EPSILON || numIter > MAX_ITER) {
				// finished training, save weights to txt file
				std::ofstream weightsFile;
				weightsFile.open("../weights_" + filename + ".txt");
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

	void run(int numInput, std::vector<float*> inputs, std::vector<float> expected, std::string filename) {

		// determine how many nodes in hidden layer (average of # of nodes in input and # of nodes in output)
		int numHidden = ceil((numInput + 1.0) / 2.0);

		// determine number of weights in all layers
		int numWeights = (numInput * numHidden) + numHidden;

		// get weights
		float* weights = new float[numWeights];
		getWeightsFromFile(numWeights, weights, filename);

		float totalError = 0.f;
		for (int i = 0; i < inputs.size(); i++) {
			float *in = inputs[i];
			totalError += runOneInput(numInput, numHidden, in, weights, nullptr, expected[i], false);
		}

		std::cout << "Total error:   " << std::to_string(totalError / 2.0) << std::endl;

		delete[] weights;
	}


}
