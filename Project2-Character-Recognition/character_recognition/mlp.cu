#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "mlp.h"
#include <cublas_v2.h>
#define blockSize 32
//https://solarianprogrammer.com/2012/05/31/matrix-multiplication-cuda-cublas-curand-thrust/
// m =  number of rows in A
// k = number of columns in A
// n = number of columns in B
void gpu_blas_mmul(const float *A, const float *B, float *C, const int m, const int k, const int n) {
	int lda = m, ldb = k, ldc = m;
	const float alf = 1;
    const float bet = 0;
    const float *alpha = &alf;
    const float *beta = &bet;
    // Create a handle for CUBLAS
	 cublasHandle_t handle;
     cublasCreate(&handle);
	
    // Do the actual multiplication
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	
	// Destroy the handle
    cublasDestroy(handle);
	
}

__global__ void activation_rule(int n, float const *idata, float* odata) {
	int index = (blockDim.x * blockIdx.x) + threadIdx.x;
	if (index >= n) {
		return;
	}

	odata[index] = 1.0 / (1.0 + std::exp(-idata[index]));
}

__global__ void activation_rule_d(int n, float const *idat, float* odata) {
	int index = (blockDim.x * blockIdx.x) + threadIdx.x;
	if (index >= n) {
		return;
	}
	odata[index] = idat[index] * (1 - idat[index]);
	
}
__global__  void matrixSub(int n, const float* a, const float* b, float* odata) {

	int index = (blockDim.x * blockIdx.x) + threadIdx.x;
	if (index >= n) {
		return;
	}
	odata[index] = a[index] - b[index];
}


void train(const float *A_input, const int A_m, const int A_n, float* weights_into_layer, float* weights_out_layer, float* predicted_output, 
	float* err, float* layer, const int layer_n, const float* expected_output, const int expected_output_n, float* post_activation_layer) {
	float *dev_input, *dev_weights_0, *dev_weights_1, *dev_error, *dev_layer, *dev_output, *dev_activation_layer, *dev_expected_output;
	cudaMalloc((void**)&dev_input, sizeof(float) * A_m * A_n);
	cudaMemcpy(dev_input, A_input, A_m * A_n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&dev_weights_0, A_m * layer_n * sizeof(float));
	cudaMemcpy(dev_weights_0, weights_into_layer, A_m * layer_n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&dev_weights_1, layer_n * sizeof(float));
	cudaMemcpy(dev_weights_1, weights_out_layer, layer_n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&dev_layer, sizeof(float) * layer_n);
	cudaMemcpy(dev_layer, layer, layer_n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&dev_activation_layer, sizeof(float) * layer_n);
	cudaMalloc((void**)&dev_error, expected_output_n * sizeof(float));
	cudaMalloc((void**)&dev_output, expected_output_n * sizeof(float));
	cudaMalloc((void**)&dev_expected_output, expected_output_n * sizeof(float));
	cudaMemcpy(dev_expected_output, expected_output, expected_output_n* sizeof(float), cudaMemcpyHostToDevice);
	float* dev_subtraction, *dev_derivative, *dev_output_d;
	cudaMalloc((void**)&dev_subtraction, expected_output_n * sizeof(float));
	cudaMalloc((void**)&dev_derivative, expected_output_n * sizeof(float));
	cudaMalloc((void**)&dev_output_d, expected_output_n * sizeof(float));
	for (int i = 0; i < 50; i++) {
		gpu_blas_mmul(dev_input, dev_weights_1, dev_layer, 1, A_n, layer_n);
		activation_rule<<<((layer_n + blockSize - 1) / blockSize), blockSize>>>(layer_n, dev_layer, dev_activation_layer);
		gpu_blas_mmul(dev_activation_layer, dev_weights_1, dev_output, 1, layer_n, A_n);
		//compute derivate. derivative of sigmoid is out * expected-out
		matrixSub << <((layer_n + blockSize - 1) / blockSize), blockSize >> > (dev_expected_output, dev_output, dev_subtraction);
		activation_rule_d << <((layer_n + blockSize - 1) / blockSize), blockSize >> > (layer_n, dev_output, dev_derivative);
		gpu_blas_mmul(dev_subtraction, dev_derivative, dev_output_d, 1, expected_output_n, expected_output_n);
        
	}

}
__global__ void runTrain(const float *A_input, const int A_m, const int A_n, float* weights_into_layer, float* weights_out_layer, const float* predicted_output, float* err, float* layer, const int layer_n, const float* expected_output, const int expected_output_n) {

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
}
