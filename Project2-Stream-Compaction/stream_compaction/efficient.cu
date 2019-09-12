#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"


#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
/*! Block size used for CUDA kernel launch. */
#define blockSize 128

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
		
		__global__ void kernUpSweep(int N, int p, int *idata) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= N) {
				return;
			}
			if (index % 2 * p == 0) {
				idata[index + 2 * p - 1] += idata[index + p - 1];
			}
			
		}

		__global__ void kernDownSweep(int N, int p, int *idata) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= N) {
				return;
			}
			if (index == N - 1) {
				idata[N - 1] = 0;
			}

			if (index % 2 * p == 0) {
				int t = idata[index + p - 1];
				idata[index + p - 1] = idata[index + (2 * p) - 1];
				idata[index + (2 * p) - 1] += t;
			}
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			int *dev_idata;

			cudaMalloc((void**)&dev_idata, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_idata failed!");

			cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
			checkCUDAError("Memcpy idata failed!");

			dim3 threadsPerBlock(blockSize);
			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            timer().startGpuTimer();
            // TODO
			for (int d = 0; d < ilog2ceil(n) - 1; d++) {
				int p = 1 << d;
				//int p = pow(2, d);
				kernUpSweep << <fullBlocksPerGrid, threadsPerBlock >> > (n, p, dev_idata);
				checkCUDAError("kernel kernUpSweep failed!");
			}

			for (int d = ilog2ceil(n) - 1; d > 0; d--) {
				int p = 1 << d;
				//int p = pow(2, d);
				kernDownSweep << <fullBlocksPerGrid, threadsPerBlock >> > (n, p, dev_idata);
				checkCUDAError("kernel kernDownSweep failed!");
			}

            timer().endGpuTimer();

			cudaMemcpy(odata, dev_idata, sizeof(int) * n, cudaMemcpyDeviceToHost);
			checkCUDAError("Memcpy odata failed!");

			cudaFree(dev_idata);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            timer().endGpuTimer();
            return -1;
        }
    }
}
