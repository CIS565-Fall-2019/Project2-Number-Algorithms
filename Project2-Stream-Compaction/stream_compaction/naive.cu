#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

#define blockSize 128

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

		int *dev_odata;
		int *dev_idata;

		__global__ void scanHelper(int *odata, int *idata, int n, int d) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) {
				return;
			}
			
			if (index >= powf(2, d - 1)) {
				int dataIndex = index - powf(2, d - 1);
				odata[index] = idata[dataIndex] + idata[index];
			}
			else {
				odata[index] = idata[index];
			}
		}

		__global__ void copyBuffer(const int *source, int *dest, int n) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) {
				return;
			}

			dest[index] = source[index];
		}

		__global__ void shiftBuffer(const int *source, int *dest, int n) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index > n) {
				return;
			}

			if (index == 0) {
				dest[index] = 0;
			}
			else {
				dest[index] = source[index - 1];
			}
		}

		void printArray(const int *array, int n) {
			printf("[");
			for (int i = 0; i < n; i++) {
				printf("%d, ", array[i]);
			}
			printf("]\n");
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			int n2 = pow(2, ceil(log2(n)));

			dim3 fullBlocksPerGrid((n2 + blockSize - 1) / blockSize);

			cudaMalloc((void**)&dev_idata, n2 * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_idata failed!");
			cudaMalloc((void**)&dev_odata, n2 * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_odata failed!");
			cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
			cudaMemcpy(dev_odata, odata, sizeof(int) * n, cudaMemcpyHostToDevice);

            timer().startGpuTimer();

			for (int d = 1; d <= log2(n2); d++) {
				scanHelper << <fullBlocksPerGrid, blockSize >> > (dev_odata, dev_idata, n2, d);
				checkCUDAErrorWithLine("Scan helper failed!");
				copyBuffer << <fullBlocksPerGrid, blockSize >> > (dev_odata, dev_idata, n2);
				checkCUDAErrorWithLine("Copy buffer failed!");
			}
			shiftBuffer << <fullBlocksPerGrid, blockSize >> > (dev_idata, dev_odata, n2);
			timer().endGpuTimer();
			cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);
			cudaFree(dev_idata);
			cudaFree(dev_odata);
        }
    }
}
