#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "mlp.h"
#include <algorithm>
#include <thrust/device_vector.h>
#include <curand.h>
#include <curand_kernel.h>

constexpr int blockSize = 64;
constexpr int maxUnit = 100;
constexpr int TILE_WIDTH = 8;

namespace CharacterRecognition {
    using Common::PerformanceTimer;
    PerformanceTimer& timer()
    {
        static PerformanceTimer timer;
        return timer;
    }

	__device__ void matrixMul(int size1, int size2, int size3, float* matrix1, float* matrix2, float *output) {
		__shared__ float sM[TILE_WIDTH][TILE_WIDTH];
		__shared__ float sN[TILE_WIDTH][TILE_WIDTH];

		int bx = blockIdx.x; 		int by = blockIdx.y;
		int tx = threadIdx.x;		int ty = threadIdx.y;

		int col = bx * TILE_WIDTH + tx;
		int row = by * TILE_WIDTH + ty;

		// Initialize accumulator to 0
		float pValue = 0;

		// Multiply and add
		for (int m = 0; m < (size2 + TILE_WIDTH - 1) / TILE_WIDTH; m++) {
			int tmpCR = m * TILE_WIDTH + tx;
			sM[ty][tx] = tmpCR < size2 ? matrix1[row * size2 + tmpCR] : 0;
			tmpCR = m * TILE_WIDTH + ty;
			sN[ty][tx] = tmpCR < size2 ? matrix2[col + tmpCR * size3] : 0;
			__syncthreads();

			for (int k = 0; k < TILE_WIDTH; ++k)
				pValue += sM[ty][k] * sN[k][tx];
			__syncthreads();
		}
		if (col >= size3 || row >= size1)
			return;
		output[0] = pValue;
	}

	__global__ void kernSubtract(int n, float* output, float* error) {
		int index = blockDim.x * blockIdx.x + threadIdx.x;
		if (index < n) {
			error[index] = output[index] - error[index];
		}
	}

	__global__ void kernSquare(int n, float *input, float *output) {
		int index = blockDim.x * blockIdx.x + threadIdx.x;
		if (index < n) {
			float tmp = input[index];
			output[index] = tmp * tmp;
		}
	}

	__global__ void kernDotMul(int n, float *m1, float *m2, float *output) {
		int index = blockDim.x * blockIdx.x + threadIdx.x;
		if (index < n) {
			output[index] = m1[index] * m2[index];
		}
	}

	__global__ void kernSigGrad(int n, float *m, float *output) {
		int index = blockDim.x * blockIdx.x + threadIdx.x;
		if (index < n) {
			float tmp = m[index];
			output[index] = tmp *(1.0f - tmp);
		}
	}

	__global__ void kernCalGrad(int curLSize, int batchSize, int preLSize, float *delta, float *X, float *grad) {
		//grad = delta * X' / batchSize;
		__shared__ float sM[TILE_WIDTH][TILE_WIDTH];
		__shared__ float sN[TILE_WIDTH][TILE_WIDTH];

		int bx = blockIdx.x; 		int by = blockIdx.y;
		int tx = threadIdx.x;		int ty = threadIdx.y;

		int col = bx * TILE_WIDTH + tx;
		int row = by * TILE_WIDTH + ty;

		// Initialize accumulator to 0
		float pValue = 0;

		// Multiply and add
		for (int m = 0; m < (batchSize + TILE_WIDTH - 1) / TILE_WIDTH; m++) {
			int tmpCR = m * TILE_WIDTH + tx;
			sM[ty][tx] = tmpCR < batchSize ? delta[row * batchSize + tmpCR] : 0;
			tmpCR = m * TILE_WIDTH + ty;
			sN[ty][tx] = tmpCR < batchSize ? X[col * batchSize + tmpCR] : 0;
			__syncthreads();

			for (int k = 0; k < TILE_WIDTH; ++k)
				pValue += sM[ty][k] * sN[k][tx];
			__syncthreads();
		}
		if (col >= preLSize || row >= curLSize)
			return;
		grad[row * preLSize + col] = pValue / batchSize;
	}

	__global__ void kernBpDelta(int preLSize, int curLSize, int batchSzie, float *weight, float *delta, float *X, float *newDelta) {
		__shared__ float sM[TILE_WIDTH][TILE_WIDTH];
		__shared__ float sN[TILE_WIDTH][TILE_WIDTH];

		int bx = blockIdx.x; 		int by = blockIdx.y;
		int tx = threadIdx.x;		int ty = threadIdx.y;

		int col = bx * TILE_WIDTH + tx;
		int row = by * TILE_WIDTH + ty;

		// Initialize accumulator to 0
		float pValue = 0;

		// Multiply and add
		for (int m = 0; m < (curLSize + TILE_WIDTH - 1) / TILE_WIDTH; m++) {
			int tmpCR = m * TILE_WIDTH + tx;
			sM[ty][tx] = tmpCR < curLSize ? weight[row + tmpCR * preLSize] : 0;
			tmpCR = m * TILE_WIDTH + ty;
			sN[ty][tx] = tmpCR < curLSize ? delta[col + tmpCR * batchSzie] : 0;
			__syncthreads();

			for (int k = 0; k < TILE_WIDTH; ++k)
				pValue += sM[ty][k] * sN[k][tx];
			__syncthreads();
		}
		if (col >= batchSzie || row >= preLSize)
			return;
		float tmp = X[row * batchSzie + col];
		newDelta[row * batchSzie + col] = pValue * tmp * (1.0f - tmp);
	}

	__global__ void kernForwardOne(int curLSize, int preLSize, int batchSize, float *weights, float *input, float *output) {
		//input : the input matrix, preLSize * batchSize
		//weights : the weight matrix, curLSize * preLSize, followed by curLSize constant term 
		//output : the output matrix, curLSize * batchSize
		
		__shared__ float sM[TILE_WIDTH][TILE_WIDTH];
		__shared__ float sN[TILE_WIDTH][TILE_WIDTH];

		int bx = blockIdx.x; 		int by = blockIdx.y;
		int tx = threadIdx.x;		int ty = threadIdx.y;

		int col = bx * TILE_WIDTH + tx;
		int row = by * TILE_WIDTH + ty;

		// Initialize accumulator to 0
		float pValue = 0;

		// Multiply and add
		for (int m = 0; m < 1/*(preLSize + TILE_WIDTH - 1) / TILE_WIDTH*/; m++) {
			int tmpCR = m * TILE_WIDTH + tx;
			sM[ty][tx] = tmpCR < preLSize ? weights[row * preLSize + tmpCR] : 0;
			tmpCR = m * TILE_WIDTH + ty;
			sN[ty][tx] = tmpCR < preLSize ? input[col + tmpCR * batchSize] : 0;
			__syncthreads();

			for (int k = 0; k < TILE_WIDTH; ++k)
				pValue += sM[ty][k] * sN[k][tx];
			__syncthreads();
		}
		if (col >= batchSize || row >= curLSize)
			return;
		output[row * batchSize + col] = 1.0f / (1.0f + expf(-pValue - weights[row + curLSize * preLSize]));
	}

	__global__ void kernUpdateW(int n, float *weight, float *weightgrad, float alpha) {
		int index = blockDim.x * blockIdx.x + threadIdx.x;
		if (index >= n)
			return;
		weight[index] -= weightgrad[index] * alpha;
	}

	__global__ void initStates(curandState *state, unsigned long seed) {
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		curand_init(seed, idx, 0, &state[idx]);
	}

	__global__ void kernIniW(int n, int n2, float r, float *weights, curandState *state) {
		int index = blockDim.x * blockIdx.x + threadIdx.x;
		if (index >= n2) {
			return;
		}
		if (index >= n) {
			weights[index] = 0;
			return;
		}
		float rand = curand_uniform(&state[index]);
		weights[index] = (rand * 2.0f - 1.0f) * r;
	}

	void initializeW(float *weights, int *layerSizes, int layerNum) {
		//Xavier Uniform Initialization
		int curIdx = 0;
		for (int i = 1; i < layerNum; i++) {
			int n = layerSizes[i] * layerSizes[i - 1];
			/*
			curandState* dev_states;
			cudaMalloc(&dev_states, n * sizeof(curandState));
			checkCUDAError("cudaMalloc dev_states failed!");

			srand(time(0));
			unsigned long seed = rand();
			dim3 GridSize2((n + blockSize - 1) / blockSize);
			initStates<<<GridSize2, blockSize>>>(dev_states, seed);

			float* dev_weights;
			cudaMalloc((void**)&dev_weights, sizeof(float) * (n + layerSizes[i]));
			checkCUDAError("cudaMalloc dev_weight failed!");
			*/
			float r = std::sqrtf(6.0f / (layerSizes[i] + layerSizes[i - 1]));
			/*
			dim3 GridSize((n + layerSizes[i] + blockSize - 1) / blockSize);
			kernIniW<<<GridSize, blockSize>>>(n, n + layerSizes[i], r, dev_weights, dev_states);
			checkCUDAError("weight initialization failed!");

			printf("!!\n");
			cudaMemcpy(weights + curIdx, dev_weights, sizeof(float) * (n + layerSizes[i]), cudaMemcpyDeviceToHost);
			checkCUDAError("copy to weights failed!");
			curIdx += n + layerSizes[i];
			printf("!\n");

			cudaFree(dev_states);
			cudaFree(dev_weights);
			*/
			srand(time(0));
			for (int j = curIdx; j < n + curIdx; j++) {
				weights[j] = (2 * ((double)rand() / (RAND_MAX)) - 1) * r;
			}
			curIdx += n;
			for (int j = curIdx; j < layerSizes[i] + curIdx; j++) {
				weights[j] = 0;
			}
			curIdx += layerSizes[i];
		}
	}

	// TODO: implement required elements for MLP sections 1 and 2 here
	float computeCostGrad(int *layerSizes, int layerNum, int batchSize, float *weights, float *grad, float *data, float *label) {
		float *dev_weight;
		cudaMalloc((void**)&dev_weight, sizeof(float) * 
			maxUnit * (std::max(layerSizes[0], maxUnit) + 1));
		checkCUDAError("cudaMalloc dev_weight failed!");

		int totalUnit = 0;
		for (int i = 0; i < layerNum; i++) {
			totalUnit += layerSizes[i];
		}

		float *dev_buffer;
		cudaMalloc((void**)&dev_buffer, sizeof(float) *
			batchSize * totalUnit);
		checkCUDAError("cudaMalloc dev_buffer failed!");

		int *indices = new int[layerNum];
		indices[0] = 0;
		int *wIndices = new int[layerNum];
		wIndices[0] = 0;
		
		int curLSize, preLSize;
		curLSize = layerSizes[0];
		cudaMemcpy(dev_buffer, data, curLSize * batchSize * sizeof(float), cudaMemcpyHostToDevice);

		//forward propagation
		dim3 TileSize(TILE_WIDTH, TILE_WIDTH, 1);
		for (int i = 1; i < layerNum; i++) {
			preLSize = curLSize;
			curLSize = layerSizes[i];
			indices[i] = indices[i - 1] + preLSize * batchSize;
			wIndices[i] = (preLSize + 1) * curLSize;
			cudaMemcpy(dev_weight, weights + wIndices[i - 1], sizeof(float) * wIndices[i], cudaMemcpyHostToDevice);
			checkCUDAError("copy to dev_weight failed!");
			wIndices[i] += wIndices[i - 1];
			dim3 GridSize((batchSize + TILE_WIDTH - 1) / TILE_WIDTH, (curLSize + TILE_WIDTH - 1) / TILE_WIDTH, 1);
			kernForwardOne<<<GridSize, TileSize>>>(curLSize, preLSize, batchSize, dev_weight, dev_buffer + indices[i - 1], dev_buffer + indices[i]);
		}

		float *dev_error;
		int numOutput = batchSize * layerSizes[layerNum - 1];
		cudaMalloc((void**)&dev_error, sizeof(float) * numOutput);
		checkCUDAError("cudaMalloc dev_error failed!");
		cudaMemcpy(dev_error, label, sizeof(float) * numOutput, cudaMemcpyHostToDevice);
		checkCUDAError("set dev_error(label) failed!");
		
		dim3 GridSize2((numOutput + blockSize - 1) / blockSize);
		kernSubtract<<<GridSize2, blockSize>>>(numOutput, dev_buffer + indices[layerNum - 1], dev_error);

		float *dev_sqrErr;
		cudaMalloc((void**)&dev_sqrErr, sizeof(float) * numOutput);
		checkCUDAError("cudaMalloc dev_sqrErr failed!");
		kernSquare<<<GridSize2, blockSize>>>(numOutput, dev_error, dev_sqrErr);

		thrust::device_ptr<float> thrust_dev_sqrErr(dev_sqrErr);
		float cost = thrust::reduce(thrust_dev_sqrErr, thrust_dev_sqrErr + numOutput, (float) 0, thrust::plus<float>());
		cost *= 0.5 / batchSize;

		cudaFree(dev_sqrErr);

		///////Calculate Grad///////

		float *dev_grad;
		cudaMalloc((void**)&dev_grad, sizeof(float) *
			maxUnit * (std::max(layerSizes[0], maxUnit) + 1));
		checkCUDAError("cudaMalloc dev_grad failed!");

		float *dev_delta;
		cudaMalloc((void**)&dev_delta, sizeof(float) *
			maxUnit * batchSize);
		checkCUDAError("cudaMalloc dev_delta failed!");

		float *dev_deltaBuffer;
		cudaMalloc((void**)&dev_deltaBuffer, sizeof(float) *
			maxUnit * batchSize);
		checkCUDAError("cudaMalloc dev_deltaBuffer failed!");

		kernSigGrad<<<GridSize2, blockSize>>>(numOutput, dev_buffer + indices[layerNum - 1], dev_delta);
		checkCUDAError("kernDotMul failed!");

		kernDotMul<<<GridSize2, blockSize>>>(numOutput, dev_error, dev_delta, dev_delta);
		checkCUDAError("kernDotMul failed!");

		for(int i = layerNum - 1; i >= 1; i--) {
			curLSize = layerSizes[i];
			preLSize = layerSizes[i - 1];
			//compute Wij grad
			dim3 GridSize3((preLSize + TILE_WIDTH - 1) / TILE_WIDTH, (curLSize + TILE_WIDTH - 1) / TILE_WIDTH, 1);
			kernCalGrad<<<GridSize3, TileSize>>>(curLSize, batchSize, preLSize, dev_delta, dev_buffer + indices[i - 1], dev_grad);
			checkCUDAError("kernCalGrad failed!");

			cudaMemcpy(grad + wIndices[i - 1], dev_grad, sizeof(float) * (wIndices[i] - wIndices[i - 1] - curLSize), cudaMemcpyDeviceToHost);
			checkCUDAError("copy dev_grad to grad failed!");

			//compute constant term grad
			float *start = dev_delta;
			for (int j = 0; j < curLSize; j++) {
				thrust::device_ptr<float> thrust_dev_delta(start);
				grad[wIndices[i] - curLSize + j] = thrust::reduce(thrust_dev_delta, thrust_dev_delta + batchSize, (float)0, thrust::plus<float>()) / batchSize;
				start += batchSize;
			}

			cudaMemcpy(dev_weight, weights + wIndices[i - 1], sizeof(float) * (wIndices[i] - wIndices[i - 1] - curLSize), cudaMemcpyHostToDevice);
			checkCUDAError("copy to dev_weight failed!");
			dim3 GridSize4((batchSize + TILE_WIDTH - 1) / TILE_WIDTH, (preLSize + TILE_WIDTH - 1) / TILE_WIDTH, 1);
			kernBpDelta<<<GridSize4, TileSize>>>(preLSize, curLSize, batchSize, dev_weight, dev_delta, dev_buffer + indices[i - 1], dev_deltaBuffer);
			checkCUDAError("delta bp failed!");
			float *tmp = dev_delta;
			dev_delta = dev_deltaBuffer;
			dev_deltaBuffer = tmp;
		}

		cudaFree(dev_delta);
		cudaFree(dev_error);
		cudaFree(dev_deltaBuffer);
		cudaFree(dev_grad);
		cudaFree(dev_weight);
		cudaFree(dev_buffer);

		delete[] indices;
		delete[] wIndices;
		
		return cost;
	}

	void updateWeights(int n, float *weight, float *weightgrad, float alpha) {
		float *dev_weights, float *dev_weightgrad;
		cudaMalloc((void**)&dev_weights, n * sizeof(float));
		checkCUDAError("malloc dev_weight failed!");
		cudaMalloc((void**)&dev_weightgrad, n * sizeof(float));
		checkCUDAError("malloc dev_weightgrad failed!");

		cudaMemcpy(dev_weights, weight, sizeof(float) * n, cudaMemcpyHostToDevice);
		checkCUDAError("copy to dev_weights failed!");
		cudaMemcpy(dev_weightgrad, weightgrad, sizeof(float) * n, cudaMemcpyHostToDevice);
		checkCUDAError("copy to dev_weightgrad failed!");

		dim3 GridSize((n + blockSize - 1) / blockSize);
		kernUpdateW<<<GridSize, blockSize>>>(n, dev_weights, dev_weightgrad, alpha);

		cudaMemcpy(weight, dev_weights, sizeof(float) * n, cudaMemcpyDeviceToHost);
		checkCUDAError("copy to weight failed!");

		cudaFree(dev_weightgrad);
		cudaFree(dev_weights);
	}
}
