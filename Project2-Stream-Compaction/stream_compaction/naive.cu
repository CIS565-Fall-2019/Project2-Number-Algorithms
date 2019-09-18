#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define blockSize 128

int *devIdata;
int *devOdata;

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
		__global__ void calculatePartialSum(int n, int d, int *odata, int *idata) {
			int index = threadIdx.x + (blockDim.x * blockIdx.x);
			if (index >= n) {
				return;
			}

			odata[index] = ((index >= (1 << (d - 1))) ? (idata[index - (1 << (d - 1))]) : 0) + idata[index];
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();

			cudaMalloc((void**)&devIdata, n * sizeof(int));
			checkCUDAError("cudaMalloc devIdata failed");
			cudaMemcpy(devIdata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			cudaMalloc((void**)&devOdata, n * sizeof(int));
			checkCUDAError("cudaMalloc devOdata failed");



            // TODO
			for (int d = 1; d <= ilog2ceil(n); d++) {
				dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
				calculatePartialSum << <fullBlocksPerGrid, blockSize>> > (n, d, devOdata, devIdata);
				cudaMemcpy(devIdata, devOdata, n * sizeof(int), cudaMemcpyDeviceToDevice);
			}
			cudaMemcpy(odata + 1, devOdata, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);
			odata[0] = 0;

			cudaFree(devIdata);
			cudaFree(devOdata);
            timer().endGpuTimer();
        }
    }
}
