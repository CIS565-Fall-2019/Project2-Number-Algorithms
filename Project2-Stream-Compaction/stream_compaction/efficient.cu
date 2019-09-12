#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"



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
		
		__global__ void kernUpSweep(int n, int p, int *idata) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) {
				return;
			}
			if (index % 2 * p == 0) {
				idata[index + 2 * p - 1] += idata[index + p - 1];
			}
			
		}

		__global__ void kernDownSweep(int n, int p, int *idata) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) {
				return;
			}
			if (index == n - 1) {
				idata[n - 1] = 0;
			}

			if (index % 2 * p == 0) {
				int t = idata[index + p - 1];
				idata[index + p - 1] = idata[index + (2 * p) - 1];
				idata[index + (2 * p) - 1] += t;
			}
		}

		void workEfficientScan(int n, int *dev_idata, dim3 &threadsPerBlock, dim3 &fullBlocksPerGrid) {
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
			workEfficientScan(n, dev_idata, threadsPerBlock, fullBlocksPerGrid);

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
			int *dev_idata;
			int *dev_indices;
			int *dev_odata;
			int *dev_bools;

			cudaMalloc((void**)&dev_idata, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_idata failed!");

			cudaMalloc((void**)&dev_odata, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_odata failed!");

			cudaMalloc((void**)&dev_indices, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_indices failed!");

			cudaMalloc((void**)&dev_bools, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_bools failed!");

			cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
			checkCUDAError("Memcpy idata failed!");

			dim3 threadsPerBlock(blockSize);
			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            timer().startGpuTimer();
            // TODO
			StreamCompaction::Common::kernMapToBoolean<< <fullBlocksPerGrid, threadsPerBlock >> > (n, dev_bools, dev_idata);

			cudaMemcpy(dev_indices, dev_bools, sizeof(int) * n, cudaMemcpyDeviceToDevice);
			checkCUDAError("Memcpy odata failed!");

			workEfficientScan(n, dev_indices, threadsPerBlock, fullBlocksPerGrid);

			StreamCompaction::Common::kernScatter << <fullBlocksPerGrid, threadsPerBlock >> > (n, dev_odata, dev_idata, dev_bools, dev_indices);

            timer().endGpuTimer();

			cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);
			checkCUDAError("Memcpy odata failed!");

			cudaFree(dev_idata);
			cudaFree(dev_indices);
			cudaFree(dev_odata);
			cudaFree(dev_bools);

            return -1;
        }
    }
}
