#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "mlp.h"

/*! Block size used for CUDA kernel launch. */
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
	__global__ void addVector(int n, float* vec1, float* vec2, float* result) {
		int index = threadIdx.x + (blockIdx.x * blockDim.x);
		if (index >= n) {
			return;
		}
		result[index] = vec1[index] + vec2[index];
	}

	__global__ void elementMultVect(int n, float *input, float *output, float *weight) {
		int index = threadIdx.x + (blockIdx.x * blockDim.x);
		if (index >= n) {
			return;
		}
		output[index] = input[index]*weight[index];
	}

	float sigmoid(float x) {
		return 1.0f / (1 + exp(-x));
	}

	float mlp(float *input, float *output, float *weight) {


		//thrust::inclusive_scan(idata, idata + n, odata);
	}

	void network() {

	}

	void partition() {

	}
	void train(int n, float *input, float* output) {
		int *dev_input;
		int *dev_output;
		int *dev_hidden;

		cudaMalloc((void**)&dev_input, n * sizeof(int));
		checkCUDAError("cudaMalloc dev_input failed!");

		cudaMalloc((void**)&dev_hidden, n * sizeof(int));
		checkCUDAError("cudaMalloc dev_output failed!");

		cudaMalloc((void**)&dev_output, n * sizeof(int));
		checkCUDAError("cudaMalloc dev_output failed!");

		cudaMemcpy(dev_input, input, sizeof(int) * n, cudaMemcpyHostToDevice);
		checkCUDAError("Memcpy input failed!");

		dim3 threadsPerBlock(blockSize);
		dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

	}
	void test() {

	}

}
