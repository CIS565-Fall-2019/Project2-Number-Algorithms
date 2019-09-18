#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

# define blockSize 512

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

		int *dev_arr1;
		int *dev_arr2;

		__global__ void kernScan(int n,int pos, int *arr2, const int *arr1) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n)
				return;
			if (index >= pos)
				arr2[index] = arr1[index - pos] + arr1[index];

		}

		__global__ void kernShiftRight(int n, int *odata, int *idata) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n)
				return;

			if (index == 0) {
				odata[index] = 0;
				return;
			}
			odata[index] = idata[index-1];
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {

			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
			
			cudaMalloc((void**)&dev_arr1, n * sizeof(int));
			checkCUDAErrorFn("Malloc idata into arr1 failed");

			cudaMalloc((void**)&dev_arr2, n * sizeof(int));
			checkCUDAErrorFn("Malloc odata into arr2 failed");

			cudaMemcpy(dev_arr1, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
			checkCUDAErrorFn("Copying idata to arr1 failed");

			cudaMemcpy(dev_arr2, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
			checkCUDAErrorFn("Memory copy idata to arr2 failed");

			timer().startGpuTimer();

			for (int d = 1; d <= ilog2ceil(n); d++) {
				int pos = 1 << (d - 1);

				kernScan << <fullBlocksPerGrid, blockSize >> > (n,pos, dev_arr2, dev_arr1);
				checkCUDAErrorFn("Kernel Scan failed");
				
				cudaMemcpy(dev_arr1, dev_arr2, sizeof(int) * n, cudaMemcpyDeviceToHost);
				checkCUDAErrorFn("Memory copy from arr2 to arr1 failed");
			}
			
			kernShiftRight << <fullBlocksPerGrid, blockSize >> > (n, dev_arr2, dev_arr1);
			checkCUDAErrorFn("Kernel Scan failed");

			timer().endGpuTimer();

			cudaMemcpy(odata, dev_arr2, sizeof(int) * n, cudaMemcpyDeviceToHost);
			checkCUDAErrorFn("Copying back to Host failed");

			cudaFree(dev_arr1);
			cudaFree(dev_arr2);

        }
    }
}
