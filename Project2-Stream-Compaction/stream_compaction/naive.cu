#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
		int *dev_idata;
		int *dev_odata;

        // TODO: __global__

		__global__ void scan_GPU(int N, int *Dev_idata, int *Dev_odata) {

		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
			int blockSize = 32;
			cudaMalloc((void**)&dev_idata, n * sizeof(int));
			cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

			cudaMalloc((void**)&dev_odata, n * sizeof(int));
			cudaMemcpy(dev_odata, odata, sizeof(int) * n, cudaMemcpyHostToDevice);

			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

			scan_GPU << <fullBlocksPerGrid, blockSize >> > (n, dev_idata, dev_odata);
            // TODO
            timer().endGpuTimer();
        }
    }
}
