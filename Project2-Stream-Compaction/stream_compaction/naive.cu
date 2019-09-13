#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include "device_launch_parameters.h"

#define blockSize 32
namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__

		__global__ void kernNaive(int n, int d, int *itemp, int *otemp) {
			int k = (blockIdx.x*blockDim.x) + threadIdx.x;
			if (k >= n) {
				return;
			}
			if (k >= (1 << (d - 1)))
				otemp[k] = itemp[k - (1 << (d - 1))] + itemp[k];
			else
				otemp[k] = itemp[k];
		}
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			int *itemp, *otemp;
			cudaMalloc((void**)&itemp, n * sizeof(int));
			checkCUDAError("Malloc for input temp array failed");
			cudaMalloc((void**)&otemp, n * sizeof(int));
			checkCUDAError("Malloc for output temp array failed");
			cudaMemcpy(itemp, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
			timer().startGpuTimer();
			for (int d = 1; d <= ilog2ceil(n); d++) {
				kernNaive <<< fullBlocksPerGrid, blockSize >>> (n, d, itemp, otemp);
				std::swap(itemp, otemp);
			}
			timer().endGpuTimer();
			std::swap(itemp, otemp);
			cudaMemcpy(odata + 1, otemp, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);
			odata[0] = 0;
			cudaFree(itemp);
			cudaFree(otemp);

        }
    }
}
