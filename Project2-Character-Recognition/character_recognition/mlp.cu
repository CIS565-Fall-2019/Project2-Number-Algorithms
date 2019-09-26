#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "mlp.h"
#include <thrust/random.h>
#include <iostream>
#include <cublas_v2.h>


#define blockSize 128

void gpu_blas_mmul(cublasHandle_t &handle, const float *A, const float *B, float *C, const int aRow, const int bRow, const int cRow) {

	const float alf = 1;
	const float bet = 0;

	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, aRow, cRow, bRow, &alf, A, aRow, B, bRow, &bet, C, aRow);
}

void gpu_blas_mtrans(cublasHandle_t &handle, const float *A, float *B, const int row, const int col) {
	float alf = 1;
	float bet = 0;

	cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, row, col, &alf, A, col, &bet, A, col, B, row);
}

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

	__global__ void kernRandomNumber(int n, int time, float* array) {
		int index = threadIdx.x + (blockIdx.x * blockDim.x);
		if (index >= n) {
			return;
		}
		thrust::default_random_engine rng(hash((int)(index * time)));
		thrust::uniform_real_distribution<float> unitDistrib(-1, 1);
		array[index] = (float)unitDistrib(rng);
	}

	__global__ void kernSigmoid(int n, float *out, float *in) {
		int index = threadIdx.x + (blockIdx.x * blockDim.x);
		if (index >= n) {
			return;
		}
		out[index] = 1 / (1 + exp(-in[index]));
	}

	__global__ void kernSmDerivative(int n, float *out, float *in) {
		int index = threadIdx.x + (blockIdx.x * blockDim.x);
		if (index >= n) {
			return;
		}
		float sigma = 1 / (1 + exp(-in[index]));
		out[index] = (1 - sigma) * sigma;
	}

	__global__ void kernLoss(int n, float *out, float *a, float *b) {
		int index = threadIdx.x + (blockIdx.x * blockDim.x);
		if (index >= n) {
			return;
		}
		float diff = a[index] - b[index];
		out[index] = powf(diff, 2);
	}

	__global__ void kernUpdate(int n, float *weights, float *gradients, float lambda) {
		int index = threadIdx.x + (blockIdx.x * blockDim.x);
		if (index >= n) {
			return;
		}
		weights[index] -= gradients[index] * lambda;
	}

	__global__ void kernSubtract(int n, float *out, float *a, float *b) {
		int index = threadIdx.x + (blockIdx.x * blockDim.x);
		if (index >= n) {
			return;
		}
		out[index] = a[index] - b[index];
	}

	__global__ void kernDotProd(int n, float *out, float *a, float *b) {
		int index = threadIdx.x + (blockIdx.x * blockDim.x);
		if (index >= n) {
			return;
		}
		out[index] = a[index] * b[index];
	}


	mlp::mlp(int input, int hidden, int output) {
		input_size = input;
		hidden_size = hidden;
		output_size = output;

		cudaMalloc((void**)&wkj, input_size * hidden_size * sizeof(float));
		cudaMalloc((void**)&wji, hidden_size * output_size * sizeof(float));
		cudaMalloc((void**)&gwkj, input_size * hidden_size * sizeof(float));
		cudaMalloc((void**)&gwji, hidden_size * output_size * sizeof(float));

		cublasCreate(&handle);
	}

	mlp::~mlp() {
		cudaFree(wkj);
		cudaFree(wji);
		cudaFree(gwkj);
		cudaFree(gwji);

		cublasDestroy(handle);
	}

	void mlp::initRandom() {
		// TODO: initialize random weights
		int n1 = input_size * hidden_size;
		int fullBlockPerGrid1 = (n1 + blockSize - 1) / blockSize;
		kernRandomNumber << <fullBlockPerGrid1, blockSize >> > (n1, 1, wji);

		int n2 = hidden_size * output_size;
		int fullBlockPerGrid2 = (n2 + blockSize - 1) / blockSize;
		kernRandomNumber << <fullBlockPerGrid1, blockSize >> > (n2, 1, wkj);
	}

	void mlp::initWeights(float *xwkj, float *xwji) {
		cudaMemcpy(wkj, xwkj, input_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(wji, xwji, hidden_size * output_size * sizeof(float), cudaMemcpyHostToDevice);
	}


	void mlp::train(float *x_train, float *y_train, int n, int epoch) {
		num = n;
		int n1 = n * input_size;
		int n2 = n * hidden_size;
		int n3 = n * output_size;

		cudaMalloc((void**)&dev_x, n1 * sizeof(float));
		cudaMalloc((void**)&dev_target, n3 * sizeof(float));
		cudaMemcpy(dev_x, x_train, n1 * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_target, y_train, n3 * sizeof(float), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_hidden, n2 * sizeof(float));
		cudaMalloc((void**)&dev_hidden_sm, n2 * sizeof(float));
		cudaMalloc((void**)&dev_y, n3 * sizeof(float));
		cudaMalloc((void**)&dev_y_sm, n3 * sizeof(float));

		for (int i = 0; i < epoch; i++) {
			forward();
			backProp();
			loss();
			update();
		}

		cudaFree(dev_x);
		cudaFree(dev_target);
		cudaFree(dev_hidden);
		cudaFree(dev_hidden_sm);
		cudaFree(dev_y);
		cudaFree(dev_y_sm);
	}

	void mlp::predict(float *x, float *y, int n) {
		num = n;
		cudaMalloc((void**)&dev_x, n * input_size * sizeof(float));
		cudaMalloc((void**)&dev_hidden, n * hidden_size * sizeof(float));
		cudaMalloc((void**)&dev_hidden_sm, n * hidden_size * sizeof(float));
		cudaMalloc((void**)&dev_y, n * output_size * sizeof(float));
		cudaMalloc((void**)&dev_y_sm, n * output_size * sizeof(float));
		cudaMemcpy(dev_x, x, n * input_size * sizeof(float), cudaMemcpyHostToDevice);

		forward();
		cudaMemcpy(y, dev_y_sm, n * output_size * sizeof(float), cudaMemcpyDeviceToHost);

		cudaFree(dev_x);
		cudaFree(dev_hidden);
		cudaFree(dev_hidden_sm);
		cudaFree(dev_y);
		cudaFree(dev_y_sm);
	}

	void mlp::forward() {
		gpu_blas_mmul(handle, dev_x, wkj, dev_hidden, num, input_size, hidden_size);
		dim3 fullBlockPerGrid1((hidden_size * num + blockSize - 1) / blockSize);
		kernSigmoid << <fullBlockPerGrid1, blockSize >> > (hidden_size * num, dev_hidden_sm, dev_hidden);

		gpu_blas_mmul(handle, dev_hidden_sm, wji, dev_y, num, hidden_size, output_size);
		dim3 fullBlockPerGrid2((output_size * num + blockSize - 1) / blockSize);
		kernSigmoid << <fullBlockPerGrid2, blockSize >> > (output_size * num, dev_y_sm, dev_y);
	}

	void mlp::backProp() {
		float *wji_T, *y_smDer, *temp1, *psi_right, *hidden_sm_T, *hidden_smDer, *psi_left, *temp2, *dev_x_T;
		
		cudaMalloc((void**)&dev_x_T, num * input_size * sizeof(float));
		cudaMalloc((void**)&y_smDer, num * output_size * sizeof(float));
		cudaMalloc((void**)&temp1, num * output_size * sizeof(float));
		cudaMalloc((void**)&psi_right, num * output_size * sizeof(float));
		cudaMalloc((void**)&hidden_sm_T, num * hidden_size * sizeof(float));
		cudaMalloc((void**)&hidden_smDer, num * hidden_size * sizeof(float));
		cudaMalloc((void**)&psi_left, num * hidden_size * sizeof(float));
		cudaMalloc((void**)&wji_T, output_size * hidden_size * sizeof(float));
		cudaMalloc((void**)&temp2, num * hidden_size * sizeof(float));

		int n1 = output_size * num;
		dim3 fullBlockPerGrid1((n1 + blockSize - 1) / blockSize);
		kernSmDerivative << <fullBlockPerGrid1, blockSize >> > (n1, y_smDer, dev_y_sm);
		kernSubtract << <fullBlockPerGrid1, blockSize >> > (num * output_size, temp1, dev_y_sm, dev_target);
		kernDotProd << <fullBlockPerGrid1, blockSize >> > (n1, psi_right, y_smDer, temp1);
		gpu_blas_mtrans(handle, dev_hidden_sm, hidden_sm_T, hidden_size, num);
		gpu_blas_mmul(handle, hidden_sm_T, psi_right, gwji, hidden_size, num, output_size);

		int n2 = hidden_size * num;
		dim3 fullBlockPerGrid2((n2 + blockSize - 1) / blockSize);
		kernSmDerivative << <fullBlockPerGrid2, blockSize >> > (n2, hidden_smDer, dev_hidden_sm);
		gpu_blas_mtrans(handle, wji, wji_T, output_size, hidden_size);
		gpu_blas_mmul(handle, psi_right, wji_T, psi_left, num, output_size, hidden_size);
		kernDotProd << <fullBlockPerGrid2, blockSize >> > (n2, temp2, psi_left, hidden_smDer);

		gpu_blas_mtrans(handle, dev_x, dev_x_T, input_size, num);
		gpu_blas_mmul(handle, dev_x_T, temp2, gwkj, input_size, num, hidden_size);

		cudaFree(y_smDer);
		cudaFree(temp1);
		cudaFree(psi_right);
		cudaFree(hidden_sm_T);
		cudaFree(hidden_smDer);
		cudaFree(psi_left);
		cudaFree(temp2);
		cudaFree(dev_x_T);
		cudaFree(wji_T);
	}

	void mlp::loss() {

		int n1 = num * output_size;
		float *dev_loss;
		float *h_loss = new float[n1]();

		cudaMalloc((void**)&dev_loss, sizeof(float) * n1);
		dim3 fullBlockPerGrid = dim3((n1 + blockSize - 1) / blockSize);
		kernLoss << <fullBlockPerGrid, blockSize >> > (n1, dev_loss, dev_target, dev_y_sm);		
		cudaMemcpy(h_loss, dev_loss, n1 * sizeof(float), cudaMemcpyDeviceToHost);

		error = 0.0;
		for (int i = 0; i < n1; i++) {
			error += h_loss[i];
		}
		error /= (2.0 * output_size);
		cudaFree(dev_loss);
		delete[] h_loss;
	}

	void mlp::update() {
		float lambda = 0.01;

		dim3 fullBlockPerGrid1 = dim3((hidden_size * output_size + blockSize) / blockSize);
		kernUpdate << <fullBlockPerGrid1, blockSize >> > (hidden_size * output_size, wji, gwji, lambda);

		dim3 fullBlockPerGrid2 = dim3((input_size * hidden_size + blockSize) / blockSize);
		kernUpdate << <fullBlockPerGrid2, blockSize >> > (input_size * hidden_size, wkj, gwkj, lambda);
	}
}