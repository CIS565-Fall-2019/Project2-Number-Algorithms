#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

#define blockSize 128

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
		__global__ void updateSum(int N, int d, int* A, int* B) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= N) {
				return;
			}
			if(index )

		}
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
			int iterations = ilog2ceil(n);

			cudaMalloc((void**)&dev_A, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_A failed!");

			cudaMalloc((void**)&dev_B, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_B failed!");




			for (int d = 0; d < iterations; d++) {
				dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

			}
            timer().endGpuTimer();
        }
    }
}
