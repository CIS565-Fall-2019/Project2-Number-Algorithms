#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

#define blockSize 128

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

		int *dev_data;

		__global__ void upSweep(int *data, int n, int d) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			int stride = powf(2, d + 1);
			if (index >= n || index % stride != 0) {
				return;
			}

			int index2 = index + powf(2, d) - 1;
			data[index + stride - 1] += data[index2];
		}

		__global__ void downSweep(int *data, int n, int d) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			int stride = powf(2, d + 1);
			if (index >= n || index % stride != 0) {
				return;
			}

			int index2 = index + powf(2, d) - 1;
			int index3 = index + powf(2, d + 1) - 1;
			int t = data[index2];
			data[index2] = data[index3];
			data[index3] += t;
		}

		__global__ void copyBuffer(const int *source, int *dest, int n) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) {
				return;
			}

			dest[index] = source[index];
		}

		__global__ void kern0LastElement(int *data, int n) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index != n - 1) {
				return;
			}

			data[index] = 0;
		}

		__global__ void kernReduction(int *data, int n, int d) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			int stride = powf(2, d + 1);
			if (index >= n || index % stride != 0) {
				return;
			}

			int index2 = index + powf(2, d) - 1;
			data[index + stride - 1] += data[index2];
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

			cudaMalloc((void**)&dev_data, n2 * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_data failed!");
			cudaMemcpy(dev_data, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

			timer().startGpuTimer();

			for (int d = 0; d < log2(n2); d++) {
				upSweep << <fullBlocksPerGrid, blockSize >> > (dev_data, n2, d);
				checkCUDAErrorWithLine("Up sweep failed!");
			}
			kern0LastElement << <fullBlocksPerGrid, blockSize >> > (dev_data, n2);
			for (int d = log2(n2) - 1; d >= 0; d--) {
				downSweep << <fullBlocksPerGrid, blockSize >> > (dev_data, n2, d);
				checkCUDAErrorWithLine("Down sweep failed!");
			}

			timer().endGpuTimer();
			cudaMemcpy(odata, dev_data, sizeof(int) * n, cudaMemcpyDeviceToHost);
			cudaFree(dev_data);
        }

		int *dev_bools;
		int *dev_idata;
		int *dev_odata;
		int *dev_scanned;

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
			int n2 = pow(2, ceil(log2(n)));

			dim3 fullBlocksPerGrid((n2 + blockSize - 1) / blockSize);

			cudaMalloc((void**)&dev_bools, n2 * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_bools failed!");
			cudaMalloc((void**)&dev_idata, n2 * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_idata failed!");
			cudaMalloc((void**)&dev_odata, n2 * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_odata failed!");
			cudaMalloc((void**)&dev_scanned, n2 * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_scanned failed!");
			cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            timer().startGpuTimer();
			Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> > (n2, dev_bools, dev_idata);
			cudaMemcpy(dev_scanned, dev_bools, sizeof(int) * n2, cudaMemcpyDeviceToDevice);
			for (int d = 0; d < log2(n2); d++) {
				upSweep << <fullBlocksPerGrid, blockSize >> > (dev_scanned, n2, d);
				checkCUDAErrorWithLine("Up sweep failed!");
			}
			kern0LastElement << <fullBlocksPerGrid, blockSize >> > (dev_scanned, n2);
			for (int d = log2(n2) - 1; d >= 0; d--) {
				downSweep << <fullBlocksPerGrid, blockSize >> > (dev_scanned, n2, d);
				checkCUDAErrorWithLine("Down sweep failed!");
			}
			Common::kernScatter << <fullBlocksPerGrid, blockSize >> > (n2, dev_odata, dev_idata, dev_bools, dev_scanned);
			for (int d = 0; d < log2(n2); d++) {
				kernReduction << <fullBlocksPerGrid, blockSize >> > (dev_bools, n2, d);
				checkCUDAErrorWithLine("Reduction failed!");
			}
            timer().endGpuTimer();

			int *summedBools = new int[n2];
			cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);
			cudaMemcpy(summedBools, dev_bools, sizeof(int) * n2, cudaMemcpyDeviceToHost);
			int toReturn = summedBools[n2 - 1];

			cudaFree(dev_bools);
			cudaFree(dev_idata);
			cudaFree(dev_odata);
			cudaFree(dev_scanned);
			delete[] summedBools;

            return toReturn;
        }
    }
}
