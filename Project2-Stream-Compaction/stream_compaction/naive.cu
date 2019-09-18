#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)
#define blockSize 64

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
		__global__ void kernAddSkip(int n, int offset, int *odata, int *idata)
		{
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			if (index >= n)
				return;
			odata[index] = (index < offset ? idata[index] : idata[index] + idata[index - offset]);
		}

		__global__ void kernMakeExclusive(int n, int *odata, int *idata)
		{
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			if (index >= n)
				return;
			odata[index] = (index == 0 ? 0 : idata[index - 1]);
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			dim3 GridSize((n + blockSize - 1) / blockSize);
			int *dev_data1, *dev_data2, *tmp;
			cudaMalloc((void**)&dev_data1, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_data1 failed!");
			cudaMalloc((void**)&dev_data2, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_data2 failed!");

			cudaMemcpy(dev_data1, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            timer().startGpuTimer();

			int offset = 1;
			for(int i = 0; i < ilog2ceil(n); i++) {
				kernAddSkip<<<GridSize, blockSize>>>(n, offset, dev_data2, dev_data1);
				checkCUDAErrorWithLine("kernAddSkip failed!");
				offset <<= 1;
				tmp = dev_data1;
				dev_data1 = dev_data2;
				dev_data2 = tmp;
			}
			kernMakeExclusive<<<GridSize, blockSize>>>(n, dev_data2, dev_data1);
			checkCUDAErrorWithLine("kernMakeExclusive failed!");

            timer().endGpuTimer();

			cudaMemcpy(odata, dev_data2, sizeof(int) * n, cudaMemcpyDeviceToHost);
			cudaFree(dev_data1);
			cudaFree(dev_data2);
			return;
        }
    }
}
