#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "mlp.h"


#ifndef imax
#define imax(a,b) (((a)>(b))?(a):(b))
#endif

#define blockSize 128

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

	// TODO: implement required elements for MLP sections 1 and 2 here


	__global__ void kernCrossEntropyLoss(int n, float *predicted, float *label, float *lossForEachLabel) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;

		if (index < n) {
			lossForEachLabel[index] = -1 * (label[index] * logf(predicted[index]));
		}
	}

	float MultiLayerPerceptron::loss(float *label, float *predicted) {
		/*float *devLabel;
		cudaMalloc((void**)&devLabel, batchDim*layer2->getOutputDim() * sizeof(float));
		checkCUDAError("cudaMalloc");
		cudaMemcpy(devLabel, label, batchDim * layer2->getOutputDim() * sizeof(float), cudaMemcpyHostToDevice);
		checkCUDAError("cudaMemcpy");

		float *devPredicted;
		cudaMalloc((void**)&devPredicted, batchDim*layer2->getOutputDim() * sizeof(float));
		checkCUDAError("cudaMalloc");
		cudaMemcpy(devPredicted, predicted, batchDim*layer2->getOutputDim() * sizeof(float), cudaMemcpyHostToDevice);
		checkCUDAError("cudaMemcpy");


		float *lossForEachLabel = new float[batchDim * layer2->getOutputDim()];
		float *devLossForEachLabel;
		cudaMalloc((void**)&devLossForEachLabel, batchDim * layer2->getOutputDim() * sizeof(float));
		checkCUDAError("cudaMalloc");

		int gridRows = ((batchDim * layer2->getOutputDim()) + blockSize - 1) / blockSize;
		kernCrossEntropyLoss << <gridRows, blockSize >> > (batchDim * layer2->getOutputDim(), devPredicted, devLabel, devLossForEachLabel);
		checkCUDAError("kernCrossEntropyLoss");

		cudaMemcpy(lossForEachLabel, devLossForEachLabel, batchDim * layer2->getOutputDim() * sizeof(float), cudaMemcpyDeviceToHost);
		checkCUDAError("cudaMemcpy");

		float loss = 0;

		for (int i = 0; i < batchDim * layer2->getOutputDim(); i++) {
			loss += lossForEachLabel[i];
		}
		return loss / batchDim;*/

		float *devLabel;
		cudaMalloc((void**)&devLabel, batchDim*layers[layers.size() - 1]->getOutputDim() * sizeof(float));
		checkCUDAError("cudaMalloc");
		cudaMemcpy(devLabel, label, batchDim * layers[layers.size() - 1]->getOutputDim() * sizeof(float), cudaMemcpyHostToDevice);
		checkCUDAError("cudaMemcpy");

		float *devPredicted;
		cudaMalloc((void**)&devPredicted, batchDim*layers[layers.size() - 1]->getOutputDim() * sizeof(float));
		checkCUDAError("cudaMalloc");
		cudaMemcpy(devPredicted, predicted, batchDim*layers[layers.size() - 1]->getOutputDim() * sizeof(float), cudaMemcpyHostToDevice);
		checkCUDAError("cudaMemcpy");
		

		float *lossForEachLabel = new float[batchDim * layers[layers.size() - 1]->getOutputDim()];
		float *devLossForEachLabel;
		cudaMalloc((void**)&devLossForEachLabel, batchDim * layers[layers.size() - 1]->getOutputDim() * sizeof(float));
		checkCUDAError("cudaMalloc");
		

		int gridRows = ((batchDim * layers[layers.size() - 1]->getOutputDim()) + blockSize - 1) / blockSize;
		kernCrossEntropyLoss << <gridRows, blockSize >> > (batchDim * layers[layers.size() - 1]->getOutputDim(), devPredicted, devLabel, devLossForEachLabel);
		checkCUDAError("kernCrossEntropyLoss");

		cudaMemcpy(lossForEachLabel, devLossForEachLabel, batchDim * layers[layers.size() - 1]->getOutputDim() * sizeof(float), cudaMemcpyDeviceToHost);
		checkCUDAError("cudaMemcpy");

		float loss = 0;

		for (int i = 0; i < batchDim * layers[layers.size() - 1]->getOutputDim(); i++) {
			loss += lossForEachLabel[i];
		}
		return loss / batchDim;
		
	}


	__global__ void kernSubtractMatrices(float *input1, float *input2, float *output, int m, int n) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		int row = index / n;
		int col = index % n;

		if (col < n && row < m) {
			int pos = row * n + col;
			output[pos] = input1[pos] - input2[pos];
		}

	}

	__global__ void kernMultiplyMatrices(float *input, float *weight, float *output, int m, int n, int k) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		int row = index/k;
		int col = index%k;
		float sum = 0;

		if (col < k && row < m) {
			for (int i = 0; i < n; i++) {
				sum += input[row * n + i] * weight[i*k + col];
			}
			output[row*k + col] = sum;
		}
	}

	__global__ void kernMultMatricesHammard(float *input1, float *input2, float *output, int m, int n) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		int row = index / n;
		int col = index % n;

		if (col < n && row < m) {
			output[row*n + col] = input1[row*n + col] * input2[row*n + col];
		}
	}

	__global__ void kernMultMatricesWithScalar(float *input, float *output, int m, int n, float scalar) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		int row = index / n;
		int col = index % n;

		if (col < n && row < m) {
			int pos = row * n + col;
			output[pos] = scalar * input[pos];
		}
	}


	__global__ void kernTransposeMatrices(float *input, float *output, int m, int n) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		int row = index / n;
		int col = index % n;

		if (col < n && row < m) {
			int pos = row * n + col;
			int newPos = col * m + row;
			output[newPos] = input[pos];
		}
	}

	__global__ void kernActivateReLU(float *input, int n) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index < n) {
			input[index] = imax(input[index], 0);
		}
	}

	__global__ void kernActivateReLUDerivative(float *input, float *output, int n) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index < n) {
			output[index] = (input[index] > 0) ? 1 : 0;
		}
	}

	__global__ void kernActivateSoftmax(float *input, int n, int outputDim, float *softmaxDenominator) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		int batchIndex = index / outputDim;
		if (index < n) {
			input[index] = expf(input[index]) / softmaxDenominator[batchIndex];
		}
	}


	void genArray(int n, float *a) {
		//11, 15
		srand(11);
		//srand(time(NULL));

		for (int i = 0; i < n; i++) {
			a[i] = ((2 *((rand() * 1.0 )/ RAND_MAX)) - 1) * 0.02;
		}
	}

	FullyConnectedLayer::FullyConnectedLayer(int inputDim, int outputDim, int batchDim, bool lastLayer) {
		this->inputDim = inputDim;
		this->outputDim = outputDim;
		this->batchDim = batchDim;
		this->lastLayer = lastLayer;
		cudaMalloc((void **)&weight, inputDim * outputDim * sizeof(float));
		float *weightRand = new float[inputDim * outputDim];
		genArray(inputDim * outputDim, weightRand);
		cudaMemcpy(weight, weightRand, inputDim * outputDim * sizeof(float), cudaMemcpyHostToDevice);
		cudaMalloc((void**)&inputs, inputDim * batchDim * sizeof(float));
	}

	int FullyConnectedLayer::getInputDim() {
		return inputDim;
	}

	int FullyConnectedLayer::getOutputDim() {
		return outputDim;
	}

	void FullyConnectedLayer::forward(float *inputArg, float *outputArg) {
		cudaMemcpy(inputs, inputArg, batchDim * inputDim * sizeof(float), cudaMemcpyDeviceToDevice);
		int gridRows = (batchDim*outputDim + blockSize - 1) / blockSize;

		kernMultiplyMatrices<<<gridRows, blockSize >>>(inputArg, weight, outputArg, batchDim, inputDim, outputDim);
		checkCUDAError("kernMultiplyMatricesFailed");

		dim3 fullBlocksPerGrid((outputDim*batchDim + blockSize - 1) / blockSize);
		if (!lastLayer) {
			kernActivateReLU << <fullBlocksPerGrid, blockSize >> > (outputArg, outputDim*batchDim);
		}
		else {
			float *output = new float[outputDim * batchDim];
			cudaMemcpy(output, outputArg, batchDim * outputDim * sizeof(float), cudaMemcpyDeviceToHost);
			float *softmaxDenominator = new float[batchDim];
			memset(softmaxDenominator, 0, batchDim * sizeof(float));
			for (int j = 0; j < batchDim; j++) {
				for (int i = 0; i < outputDim; i++) {
					softmaxDenominator[j] += exp(output[j * outputDim + i]);
				}
			}

			float *devSoftmaxDenominator;
			cudaMalloc((void **)&devSoftmaxDenominator, batchDim * sizeof(float));
			cudaMemcpy(devSoftmaxDenominator, softmaxDenominator, batchDim * sizeof(float), cudaMemcpyHostToDevice);
			kernActivateSoftmax << <fullBlocksPerGrid, blockSize >> > (outputArg, batchDim * outputDim, outputDim, devSoftmaxDenominator);
			checkCUDAError("kernActivateSoftmax");

			delete(output);
		}
	}

	void FullyConnectedLayer::backward(float learningRate, float *incomingGradient, float *outgoingGradient) {


		float *delLByDelX;
		cudaMalloc((void**)&delLByDelX, inputDim * sizeof(float));
		checkCUDAError("cudaMalloc");

		float *weightTranspose;
		cudaMalloc((void**)&weightTranspose, inputDim * outputDim * sizeof(float));
		checkCUDAError("cudaMalloc");

		int gridRows = (inputDim*outputDim + blockSize - 1) / blockSize;
		kernTransposeMatrices << <gridRows, blockSize >> > (weight, weightTranspose, inputDim, outputDim);
		checkCUDAError("kernTransposeMatrices");

		float *outgoingGradientLocal;
		cudaMalloc((void**)&outgoingGradientLocal, inputDim*batchDim * sizeof(float));
		checkCUDAError("cudaMalloc");

	
		gridRows = (inputDim*batchDim + blockSize - 1) / blockSize;
		kernMultiplyMatrices << <gridRows, blockSize >> > (incomingGradient, weightTranspose, outgoingGradientLocal, batchDim, outputDim, inputDim);
		checkCUDAError("kernMultiplyMatrices");

		float *inputDerivatived;
		cudaMalloc((void**)&inputDerivatived, batchDim * inputDim * sizeof(float));
		dim3 fullBlocksPerGrid((inputDim * batchDim + blockSize - 1) / blockSize);
		kernActivateReLUDerivative << <fullBlocksPerGrid, blockSize >> > (inputs, inputDerivatived, inputDim * batchDim);
		checkCUDAError("kernActivateReLUDerivative");


		gridRows = (inputDim*batchDim + blockSize - 1) / blockSize;
		kernMultMatricesHammard << <gridRows, blockSize >> > (outgoingGradientLocal, inputDerivatived, outgoingGradient, batchDim, inputDim);
		checkCUDAError("kernMultMatricesHammard");


		float *inputTranspose;
		cudaMalloc((void**)&inputTranspose, inputDim * batchDim * sizeof(float));

		gridRows = (inputDim*batchDim + blockSize - 1) / blockSize;
		kernTransposeMatrices << <gridRows, blockSize >> > (inputs, inputTranspose, batchDim, inputDim);
		checkCUDAError("kernTransposeMatrices");

		float *gradient;
		cudaMalloc((void**)&gradient, inputDim * outputDim * sizeof(float));
		gridRows = (inputDim*outputDim + blockSize - 1) / blockSize;
		kernMultiplyMatrices << <gridRows, blockSize >> > (inputTranspose, incomingGradient, gradient, inputDim, batchDim, outputDim);
		checkCUDAError("kernMultiplyMatrices");


		kernMultMatricesWithScalar << <gridRows, blockSize >> > (gradient, gradient, inputDim, outputDim, learningRate);
		checkCUDAError("kernMultMatricesWithScalar");


		kernSubtractMatrices << <gridRows, blockSize >> > (weight, gradient, weight, inputDim, outputDim);
		checkCUDAError("kernSubtractMatrices");
	}



	/*MultiLayerPerceptron::MultiLayerPerceptron(int inputDim, int hiddenDim, int outputDim, int batchDim) {
		this->batchDim = batchDim;

		this->layer1 = new FullyConnectedLayer(inputDim, hiddenDim, batchDim, false);
		this->layer2 = new FullyConnectedLayer(hiddenDim, outputDim, batchDim, true);

	}*/


	MultiLayerPerceptron::MultiLayerPerceptron(int inputDim, int numHiddenLayers, int *hiddenDim, int outputDim, int batchDim) {
		this->batchDim = batchDim;

		FullyConnectedLayer *tempLayer = new FullyConnectedLayer(inputDim, hiddenDim[0], batchDim, false);
		layers.push_back(tempLayer);
		for (int i = 1; i < numHiddenLayers - 1; i++) {
			FullyConnectedLayer *tempLayer = new FullyConnectedLayer(hiddenDim[i - 1], hiddenDim[i], batchDim, false);
			layers.push_back(tempLayer);
		}
		tempLayer = new FullyConnectedLayer(hiddenDim[numHiddenLayers - 1], outputDim, batchDim, true);
		layers.push_back(tempLayer);

	}

	void MultiLayerPerceptron::forward(float *input, float *output) {
		/*float *devInput;
		cudaMalloc((void**)&devInput, batchDim * layer1->getInputDim() * sizeof(float));
		checkCUDAError("cudaMalloc");
		cudaMemset(devInput, 1, batchDim * layer1->getInputDim() * sizeof(float));
		checkCUDAError("cudaMemset");
		cudaMemcpy(devInput, input, batchDim * layer1->getInputDim() * sizeof(float), cudaMemcpyHostToDevice);
		checkCUDAError("cudaMemcpy");


		float *hiddenOutput;
		cudaMalloc((void**)&hiddenOutput, batchDim * layer1->getOutputDim() * sizeof(float));
		checkCUDAError("cudaMalloc");
		layer1->forward(devInput, hiddenOutput);

		cudaFree(devInput);
		cudaMalloc((void**)&devInput, batchDim * layer2->getInputDim() * sizeof(float));
		checkCUDAError("cudaMalloc");
		cudaMemcpy(devInput, hiddenOutput, batchDim * layer2->getInputDim() * sizeof(float), cudaMemcpyDeviceToDevice);
		checkCUDAError("cudaMemcpy");
		cudaFree(hiddenOutput);

		float *devOutput;
		cudaMalloc((void**)&devOutput, batchDim * layer2->getOutputDim() * sizeof(float));
		checkCUDAError("cudaMalloc");
		layer2->forward(devInput, devOutput);
		cudaMemcpy(output, devOutput, batchDim * layer2->getOutputDim() * sizeof(float), cudaMemcpyDeviceToHost);
		checkCUDAError("cudaMemcpy");

		cudaFree(devInput);
		cudaFree(devOutput);*/



		float *devOutput;
		cudaMalloc((void**)&devOutput, batchDim * layers[0]->getInputDim() * sizeof(float));
		checkCUDAError("cudaMalloc");/*
		cudaMemset(devOutput, 1, batchDim * layers[0]->getInputDim() * sizeof(float));
		checkCUDAError("cudaMemset");*/
		cudaMemcpy(devOutput, input, batchDim * layers[0]->getInputDim() * sizeof(float), cudaMemcpyHostToDevice);
		checkCUDAError("cudaMemcpy");
		float *hiddenOutput;
		for (int i = 0; i < layers.size(); i++) {
			cudaMalloc((void**)&hiddenOutput, batchDim * layers[i]->getOutputDim() * sizeof(float));
			checkCUDAError("cudaMalloc");
			layers[i]->forward(devOutput, hiddenOutput);
			cudaFree(devOutput);
			cudaMalloc((void**)&devOutput, batchDim * layers[i]->getOutputDim() * sizeof(float));
			checkCUDAError("cudaMalloc");
			cudaMemcpy(devOutput, hiddenOutput, batchDim * layers[i]->getOutputDim() * sizeof(float), cudaMemcpyDeviceToDevice);
			checkCUDAError("cudaMemcpy");
			cudaFree(hiddenOutput);
		}
		cudaMemcpy(output, devOutput, batchDim * layers[layers.size() - 1]->getOutputDim() * sizeof(float), cudaMemcpyDeviceToHost);
		checkCUDAError("cudaMemcpy");

		cudaFree(devOutput);
	}

	void MultiLayerPerceptron::backward(float *label, float *predicted, float learningRate) {


		/*float *devLabel;
		cudaMalloc((void**)&devLabel, batchDim*layer2->getOutputDim() * sizeof(float));
		checkCUDAError("cudaMalloc");
		cudaMemcpy(devLabel, label, batchDim * layer2->getOutputDim() * sizeof(float), cudaMemcpyHostToDevice);
		checkCUDAError("cudaMemcpy");

		float *devPredicted;
		cudaMalloc((void**)&devPredicted, batchDim*layer2->getOutputDim() * sizeof(float));
		checkCUDAError("cudaMalloc");
		cudaMemcpy(devPredicted, predicted, batchDim*layer2->getOutputDim() * sizeof(float), cudaMemcpyHostToDevice);
		checkCUDAError("cudaMemcpy");

		float *incomingGradient;
		cudaMalloc((void**)&incomingGradient, batchDim*layer2->getOutputDim() * sizeof(float));
		checkCUDAError("cudaMalloc");

		int gridRows = ((batchDim * layer2->getOutputDim()) + blockSize - 1) / blockSize;
		kernSubtractMatrices << <gridRows, blockSize >> > (devPredicted, devLabel, incomingGradient, batchDim, layer2->getOutputDim());
		checkCUDAError("kernSubtractMatrices");

		float *outgoingGradient;
		cudaMalloc((void**)&outgoingGradient, batchDim*layer2->getInputDim() * sizeof(float));
		checkCUDAError("cudaMalloc");
		layer2->backward(learningRate, incomingGradient, outgoingGradient);
		cudaFree(incomingGradient);
		cudaMalloc((void**)&incomingGradient, batchDim*layer2->getInputDim() * sizeof(float));
		checkCUDAError("cudaMalloc");
		cudaMemcpy(incomingGradient, outgoingGradient, batchDim*layer2->getInputDim() * sizeof(float), cudaMemcpyDeviceToDevice);
		checkCUDAError("cudaMemcpy");
		cudaFree(outgoingGradient);
		cudaMalloc((void**)&outgoingGradient, batchDim*layer1->getInputDim() * sizeof(float));
		checkCUDAError("cudaMalloc");
		layer1->backward(learningRate, incomingGradient, outgoingGradient);*/



		float *devLabel;
		cudaMalloc((void**)&devLabel, batchDim*layers[layers.size() - 1]->getOutputDim() * sizeof(float));
		checkCUDAError("cudaMalloc");
		cudaMemcpy(devLabel, label, batchDim * layers[layers.size() - 1]->getOutputDim() * sizeof(float), cudaMemcpyHostToDevice);
		checkCUDAError("cudaMemcpy");

		float *devPredicted;
		cudaMalloc((void**)&devPredicted, batchDim*layers[layers.size() - 1]->getOutputDim() * sizeof(float));
		checkCUDAError("cudaMalloc");
		cudaMemcpy(devPredicted, predicted, batchDim*layers[layers.size() - 1]->getOutputDim() * sizeof(float), cudaMemcpyHostToDevice);
		checkCUDAError("cudaMemcpy");

		float *incomingGradient;
		cudaMalloc((void**)&incomingGradient, batchDim*layers[layers.size() - 1]->getOutputDim() * sizeof(float));
		checkCUDAError("cudaMalloc");

		int gridRows = ((batchDim * layers[layers.size() - 1]->getOutputDim()) + blockSize - 1) / blockSize;
		kernSubtractMatrices << <gridRows, blockSize >> > (devPredicted, devLabel, incomingGradient, batchDim, layers[layers.size() - 1]->getOutputDim());
		checkCUDAError("kernSubtractMatrices");


		checkCUDAError("cudaMemcpy");
		float *outgoingGradient;
		for (int i = layers.size() - 1; i >= 0; i--) {
			cudaMalloc((void**)&outgoingGradient, batchDim*layers[i]->getInputDim() * sizeof(float));
			checkCUDAError("cudaMalloc");
			layers[i]->backward(learningRate, incomingGradient, outgoingGradient); 
			cudaFree(incomingGradient);
			cudaMalloc((void**)&incomingGradient, batchDim*layers[i]->getInputDim() * sizeof(float));
			checkCUDAError("cudaMalloc");
			cudaMemcpy(incomingGradient, outgoingGradient, batchDim*layers[i]->getInputDim() * sizeof(float), cudaMemcpyDeviceToDevice);
			checkCUDAError("cudaMemcpy");
			cudaFree(outgoingGradient);
		}
		cudaFree(incomingGradient);


	}


}
