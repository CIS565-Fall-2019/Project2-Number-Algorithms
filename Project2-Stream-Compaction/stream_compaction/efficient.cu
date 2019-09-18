#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)
#define blockSize 512

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

		__global__ void kernUpSweep(int n, int d, int *data) {
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			if (index >= n >> (d + 1))
				return;
			int offset = 1 << d;
			int ai = offset * (2 * index + 1) - 1;
			int bi = offset * (2 * index + 2) - 1;
			data[bi] += data[ai];
		}

		__global__ void kernDownSweep(int n, int d, int *data) {
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			if (index >= 1 << d)
				return;
			int offset = n >> (d + 1);
			int ai = offset * (2 * index + 1) - 1;
			int bi = offset * (2 * index + 2) - 1;
			float t = data[ai];
			data[ai] = data[bi];
			data[bi] += t;
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			if (n < 2)
				return;
			int n_ = 1 << ilog2ceil(n);
			int *dev_data;
			cudaMalloc((void**)&dev_data, n_ * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_data failed!");

			cudaMemset(dev_data, 0, sizeof(int) * n_);
			checkCUDAErrorWithLine("Set dev_data to 0 failed!");
			cudaMemcpy(dev_data, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
			checkCUDAErrorWithLine("Copy idata to dev_data failed!");
		
            timer().startGpuTimer();

			dim3 GridSize((n_ / 2 + blockSize - 1) / blockSize);
			for (int d = 0; d < ilog2ceil(n); d++) {
				kernUpSweep<<<GridSize, blockSize>>>(n_, d, dev_data);
				checkCUDAErrorWithLine("kernUpSweep failed!");
			}

			cudaMemset(dev_data + n_ - 1, 0, sizeof(int));
			checkCUDAErrorWithLine("set zero failed!");

			for (int d = 0; d < ilog2ceil(n); d++) {
				kernDownSweep<<<GridSize, blockSize>>>(n_, d, dev_data);
				checkCUDAErrorWithLine("kernDownSweep failed!");
			}

            timer().endGpuTimer();

			cudaMemcpy(odata, dev_data, sizeof(int) * n, cudaMemcpyDeviceToHost);
			checkCUDAErrorWithLine("Copy dev_data to odata failed!");
			cudaFree(dev_data);
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
			if (n == 0)
				return 0;
			if (n == 1){
				if (idata[0] != 0) {
					odata[0] = idata[0];
					return 1;
				}
				else {
					return 0;
				}
			}

			int n_ = 1 << ilog2ceil(n);
			int *dev_idata, *dev_odata, *dev_bools, *dev_indices;

			cudaMalloc((void**)&dev_idata, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_idata failed!");
			cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

			cudaMalloc((void**)&dev_odata, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_odata failed!");
			
			cudaMalloc((void**)&dev_bools, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_bools failed!");
			
			cudaMalloc((void**)&dev_indices, n_ * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_indices failed!");

            timer().startGpuTimer();
            // TODO
			dim3 GridSize1((n + blockSize - 1) / blockSize);
			StreamCompaction::Common::kernMapToBoolean<<<GridSize1, blockSize>>>(n, dev_bools, dev_idata);
			checkCUDAErrorWithLine("kernMapToBoolean failed!");

			cudaMemset(dev_indices, 0, sizeof(int) * n_);
			checkCUDAErrorWithLine("Set dev_indices to 0 failed!");
			
			cudaMemcpy(dev_indices, dev_bools, sizeof(int) * n, cudaMemcpyDeviceToDevice);
			checkCUDAErrorWithLine("Copy dev_bools to dev_indices failed!");
			
			dim3 GridSize2((n_ / 2 + blockSize - 1) / blockSize);
			for (int d = 0; d < ilog2ceil(n); d++) {
				kernUpSweep<<<GridSize2, blockSize>>>(n_, d, dev_indices);
				checkCUDAErrorWithLine("kernUpSweep failed!");
			}

			cudaMemset(dev_indices + n_ - 1, 0, sizeof(int));
			checkCUDAErrorWithLine("set zero failed!");

			for (int d = 0; d < ilog2ceil(n); d++) {
				kernDownSweep<<<GridSize2, blockSize>>>(n_, d, dev_indices);
				checkCUDAErrorWithLine("kernDownSweep failed!");
			}

			StreamCompaction::Common::kernScatter<<<GridSize1, blockSize>>>(n, dev_odata, dev_idata, dev_bools, dev_indices);

            timer().endGpuTimer();
			
			cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);
			checkCUDAErrorWithLine("copy dev_odata to odata failed!");
			int count;
			cudaMemcpy(&count, dev_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAErrorWithLine("copy last indice failed!");
			if (idata[n - 1] != 0) {
				count++;
			}

			cudaFree(dev_idata);
			cudaFree(dev_odata);
			cudaFree(dev_bools);
			cudaFree(dev_indices);
            return count;
        }
    }
}
