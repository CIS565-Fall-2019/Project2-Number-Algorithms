#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include "device_launch_parameters.h"
#include <math.h>

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

//#define blockSize 128

int* dev_A;
int* dev_B;

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
		__global__ void updateSum(int N, int d, int* input, int* output) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= N) {
				return;
			}
			int offset = 1 << (d - 1);
			if (index >= offset) {
				output[index] = input[index - offset] + input[index];
			}
			else {
				output[index] = input[index];
			}
		}
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata, int blockSize) {
			cudaMalloc((void**)&dev_A, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_A failed!");

			cudaMalloc((void**)&dev_B, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_B failed!");

			cudaMemcpy(dev_A, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

			timer().startGpuTimer();
			int iterations = ilog2ceil(n);
			for (int d = 1; d <= iterations; d++) {
				dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
				updateSum << <fullBlocksPerGrid, blockSize >> > (n, d, dev_A, dev_B);
				std::swap(dev_A, dev_B);
			}
            timer().endGpuTimer();
			cudaMemcpy(odata + 1, dev_A, sizeof(int) * (n - 1), cudaMemcpyDeviceToHost);
			odata[0] = 0;
        }
    }
}
