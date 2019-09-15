#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "mlp.h"
#include <cublas_v2.h>
#include <curand.h>
#include<memory.h>
#include<iostream>

#define blockSize 128

namespace CharacterRecognition {
    using Common::PerformanceTimer;
    PerformanceTimer& timer()
    {
        static PerformanceTimer timer;
        return timer;
    }

	__global__ void kernAddVectors(int n, float* g, float* bias, float* result) {
		int index = (blockIdx.x*blockDim.x) + threadIdx.x;
		if (index >= n)
			return;
		result[index] = g[index] + bias[index];
	}

	__global__ void kernSubVectors(int n, float* y, float* yhat, float* result) {
		int index = (blockIdx.x*blockDim.x) + threadIdx.x;
		if (index >= n)
			return;
		result[index] = y[index] - yhat[index];
	}

	__global__ void kernInitBiasVectors(int n, float* b, float value) {
		int index = (blockIdx.x*blockDim.x) + threadIdx.x;
		if (index >= n)
			return;
		b[index] = value;
	}

	__global__ void kernUpSweep(int n, int d, float *itemp) {
		int k = (blockIdx.x*blockDim.x) + threadIdx.x;
		if (k > (n - 1)) {
			return;
		}
		int power = 1 << (d + 1);
		int power_2 = 1 << d;
		if (k % power == 0 && k + power - 1 < n && k + power_2 - 1 < n)
			itemp[k + power - 1] += itemp[k + power_2 - 1];
	}

	__global__ void kernSoftmaxActivation(int n, float *g, float *output, float exp_sum) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;

		if (index >= n)
			return;
		output[index] = expf(g[index]) / exp_sum;
	}

	__global__ void kernSoftmaxDerivative(int n, float *input, float *grad) {
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y;
		if (i >= n || j >= n) {
			return;
		}
		if (i == j) {
			grad[i*n + j] = input[i] * (1 - input[i]);
		}
		else {
			grad[i*n + j] = -input[i] * input[j];
		}
	}

	__global__ void kernReluActivationForward(int n, float* g, float* a) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;

		if (index >= n)
			return; 
		a[index] = fmaxf(g[index], 0);
	}

	__global__ void kernReluDerivative(int n, float* g, float* a) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;

		if (index >= n)
			return;
		a[index*n+index] = (g[index] > 0) ? 1 : 0;
	}

	__global__ void kernCopyVectors(int n, float *g, float *output) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= n)
			return;
		output[index] = expf(g[index]);
	}

	void random_init(float * A, int rows, int cols) {
		curandGenerator_t prng;
		curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
		curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());
		curandGenerateNormal(prng, A, rows * cols, 0, 2.0/rows);
	}

	//C(m,n) = A(m,k)*B(k,n)
	void NeuralNet::mmul(const float* A, const float* B, float* C, const int m, const int k, const int n) {
		int lda = m, ldb = k, ldc = m;
		const float alf = 1;
		const float bet = 0;
		const float *alpha = &alf;
		const float *beta = &bet;		
		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	}
	void printCuda(float *a1, int n, std::string name) {
		float *print_a = new float[n];
		std::cout << name.c_str() << std::endl;
		std::cout << "{" << std::endl;
		cudaMemcpy(print_a, a1, n * sizeof(float), cudaMemcpyDeviceToHost);
		for (int i = 0; i < n; i++) {
			std::cout << "\t" << print_a[i] << std::endl;
		}
		std::cout << "}" << std::endl;
		delete[]print_a;
	}

	NeuralNet::NeuralNet() {
		cudaMalloc((void**)&inp, 1 * 196 * sizeof(float));
		checkCUDAError("Cuda Malloc failed");
		cudaMalloc((void**)&theta1, 196 * 98 * sizeof(float));
		checkCUDAError("Cuda Malloc failed");
		cudaMalloc((void**)&bias1, 1 * 98 * sizeof(float));
		checkCUDAError("Cuda Malloc failed");
		cudaMalloc((void**)&theta2, 98 * 65 * sizeof(float));
		checkCUDAError("Cuda Malloc failed");
		cudaMalloc((void**)&bias2, 1 * 65 * sizeof(float));
		checkCUDAError("Cuda Malloc failed");
		cudaMalloc((void**)&theta3, 65 * 52 * sizeof(float));
		checkCUDAError("Cuda Malloc failed");
		cudaMalloc((void**)&bias3, 1 * 52 * sizeof(float));
		checkCUDAError("Cuda Malloc failed");
		cudaMalloc((void**)&g1, 1 * 98 * sizeof(float));
		checkCUDAError("Cuda Malloc failed");
		cudaMalloc((void**)&g2, 1 * 65 * sizeof(float));
		checkCUDAError("Cuda Malloc failed");
		cudaMalloc((void**)&g3, 1 * 52 * sizeof(float));
		checkCUDAError("Cuda Malloc failed");
		cudaMalloc((void**)&a1, 1 * 98 * sizeof(float));
		checkCUDAError("Cuda Malloc failed");
		cudaMalloc((void**)&a2, 1 * 65 * sizeof(float));
		checkCUDAError("Cuda Malloc failed");
		cudaMalloc((void**)&output, 1 * 52 * sizeof(float));
		checkCUDAError("Cuda Malloc failed");
		cudaMalloc((void**)&dtheta1, 196 * 98 * sizeof(float));
		checkCUDAError("Cuda Malloc failed");
		cudaMalloc((void**)&dbias1, 1 * 98 * sizeof(float));
		checkCUDAError("Cuda Malloc failed");
		cudaMalloc((void**)&dtheta2, 98 * 65 * sizeof(float));
		checkCUDAError("Cuda Malloc failed");
		cudaMalloc((void**)&dbias2, 1 * 65 * sizeof(float));
		checkCUDAError("Cuda Malloc failed");
		cudaMalloc((void**)&dtheta3, 65 * 52 * sizeof(float));
		checkCUDAError("Cuda Malloc failed");
		cudaMalloc((void**)&dbias3, 1 * 52 * sizeof(float));
		checkCUDAError("Cuda Malloc failed");
		cudaMalloc((void**)&dyhatg3, 52 * 52 * sizeof(float));
		checkCUDAError("Cuda Malloc failed");
		cudaMalloc((void**)&da2g2, 65 * 65 * sizeof(float));
		checkCUDAError("Cuda Malloc failed");
		cudaMalloc((void**)&da1g1, 98 * 98 * sizeof(float));
		checkCUDAError("Cuda Malloc failed");
		//cudaMemset(theta1, 0, 196 * 98 * sizeof(float));
		//cudaMemset(theta2, 0, 98 * 65 * sizeof(float));
		//cudaMemset(theta3, 0, 65 * 52 * sizeof(float));
		random_init(theta1, 196, 98);
		random_init(theta2, 98, 65);
		random_init(theta3, 65, 52);
		dim3 fullBlocksPerGrid((196*98 + blockSize - 1) / blockSize);
		//kernInitBiasVectors << <fullBlocksPerGrid, blockSize >> > (196*98, theta1, 0.00001);
		//fullBlocksPerGrid = ((98*65 + blockSize - 1) / blockSize);
		//kernInitBiasVectors << <fullBlocksPerGrid, blockSize >> > (98*65, theta2, 0.000001);
		//fullBlocksPerGrid = ((65*52 + blockSize - 1) / blockSize);
		//kernInitBiasVectors << <fullBlocksPerGrid, blockSize >> > (65*52, theta3, 0.00001);
		fullBlocksPerGrid = ((98 + blockSize - 1) / blockSize);
		kernInitBiasVectors << <fullBlocksPerGrid, blockSize >> > (98, bias1, 0.1);
		fullBlocksPerGrid = ((65 + blockSize - 1) / blockSize);
		kernInitBiasVectors << <fullBlocksPerGrid, blockSize >> > (65, bias2, 0.1);
		fullBlocksPerGrid = ((52 + blockSize - 1) / blockSize);
		kernInitBiasVectors << <fullBlocksPerGrid, blockSize >> > (52, bias3, 0.1);

		cublasCreate(&handle);
	}

	
	float* NeuralNet::forward(float *input) {
		cudaMemset(g1, 0, 1 * 98 * sizeof(float));
		checkCUDAError("Cuda Memset failed");
		cudaMemset(g2, 0, 1 * 65 * sizeof(float));
		checkCUDAError("Cuda Memset failed");
		cudaMemset(g3, 0, 1 * 52 * sizeof(float));
		checkCUDAError("Cuda Memset failed");
		cudaMemset(a1, 0, 1 * 98 * sizeof(float));
		checkCUDAError("Cuda Memset failed");
		cudaMemset(a2, 0, 1 * 65 * sizeof(float));
		checkCUDAError("Cuda Memset failed");
		cudaMemset(output, 0, 1 * 52 * sizeof(float));
		checkCUDAError("Cuda Memset failed");
		dim3 fullBlocksPerGrid((98 + blockSize - 1) / blockSize);
		cudaMemcpy(inp, input, 196*sizeof(float),cudaMemcpyHostToDevice);
		mmul(inp, theta1, g1, 1, 196, 98);
		kernAddVectors << <fullBlocksPerGrid, blockSize >> > (98, g1, bias1, g1);
		kernReluActivationForward << <fullBlocksPerGrid, blockSize >> > (98, g1, a1);
		fullBlocksPerGrid = ((65 + blockSize - 1) / blockSize);
		mmul(a1, theta2, g2, 1, 98, 65);
		kernAddVectors << <fullBlocksPerGrid, blockSize >> > (65, g2, bias2, g2);
		kernReluActivationForward << <fullBlocksPerGrid, blockSize >> > (65, g2, a2);
		fullBlocksPerGrid = ((52 + blockSize - 1) / blockSize);
		mmul(a2, theta3, g3, 1, 65, 52);
		kernAddVectors << <fullBlocksPerGrid, blockSize >> > (52, g3, bias3, g3);
		int D = ilog2ceil(52);
		int tot_size = (1 << D);
		kernCopyVectors << <fullBlocksPerGrid, blockSize >> > (52, g3, output);
		float *out = new float[52];
		cudaMemcpy(out, output, 52 * sizeof(float), cudaMemcpyDeviceToHost);
		float exp_sum = 0;
		for (int i = 0; i < 52; i++) {
			exp_sum += out[i];
		}
		fullBlocksPerGrid = ((52 + blockSize - 1) / blockSize);
		
		kernSoftmaxActivation << <fullBlocksPerGrid, blockSize >> > (52, g3, output, exp_sum);
		cudaMemcpy(out, output, 52 * sizeof(float), cudaMemcpyDeviceToHost);
		return out;
		
		

	}
	NeuralNet::~NeuralNet() {
		cudaFree(theta1);
		cudaFree(theta2);
		cudaFree(theta3);
		cudaFree(bias1);
		cudaFree(bias2);
		cudaFree(bias3);
		cudaFree(dtheta1);
		cudaFree(dtheta2);
		cudaFree(dtheta3);
		cudaFree(dbias1);
		cudaFree(dbias2);
		cudaFree(dbias3);
		cublasDestroy(handle);
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
}
