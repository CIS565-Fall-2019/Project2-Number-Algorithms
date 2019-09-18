#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define blockSize 128

int *devIdataEfficient;
int *devIdataEfficientCompact;
int *devIdataEfficientBinaryMap;
int *devIdataEfficientNewIndices;

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

		__global__ void upSweep(int n, int d, int *input) {
			int index = (threadIdx.x + (blockDim.x * blockIdx.x))*(1 << (d + 1));
			if (index + (1 << (d + 1)) - 1 >= n) {
				return;
			}
			input[index + (1 << (d + 1)) - 1] += input[index + (1 << d) - 1];
		}


		__global__ void downSweep(int n, int d, int *input) {
			int index = (threadIdx.x + (blockDim.x * blockIdx.x))*(1 << (d + 1));
			if (index + (1 << (d + 1)) - 1 >= n) {
				return;
			}
			int t = input[index + (1 << d) - 1];
			input[index + (1 << d) - 1] = input[index + (1 << (d + 1)) - 1];
			input[index + (1 << (d + 1)) - 1] += t;
		}

		/**
		 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
		 */
		void scan(int n, int *odata, const int *idata) {
			bool exception = true;
			try {
				timer().startGpuTimer();
				exception = false;
			}
			catch (const std::exception& e) {
				exception = true;
			}




			int newSize = 1 << ilog2ceil(n);
			cudaMalloc((void**)&devIdataEfficient, newSize * sizeof(int));
			checkCUDAError("cudaMalloc devIdataEfficient failed");
			cudaMemset(devIdataEfficient, 0, newSize * sizeof(int));
			cudaMemcpy(devIdataEfficient, idata, n * sizeof(int), cudaMemcpyHostToDevice);




			// TODO
			for (int d = 0; d < ilog2(newSize); d++) {
				dim3 fullBlocksPerGrid((((int)(newSize / (1 << (d + 1)))) + blockSize - 1) / blockSize);
				upSweep << <fullBlocksPerGrid, blockSize >> > (newSize, d, devIdataEfficient);
			}


			cudaMemset(devIdataEfficient + (newSize - 1), 0, 1 * sizeof(int));

			for (int d = ilog2(newSize) - 1; d >= 0; d--) {
				dim3 fullBlocksPerGrid(((1 << (ilog2(newSize) - d)) + blockSize - 1) / blockSize);
				downSweep << <fullBlocksPerGrid, blockSize >> > (newSize, d, devIdataEfficient);
			}
			cudaMemcpy(odata, devIdataEfficient, n * sizeof(int), cudaMemcpyDeviceToHost);
			cudaFree(devIdataEfficient);


			try {
				if (exception == false) {
					timer().endGpuTimer();
				}
			}
			catch (const std::exception& e) {

			}
		}

		/**
		 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
		 */
		void scanForRadix(int n, int *odata, const int *idata, int radixBlockSize) {
			int newSize = 1 << ilog2ceil(n);

			cudaMemset(odata, 0, newSize * sizeof(int));
			checkCUDAError("cudaMemset");
			cudaMemcpy(odata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy");




			// TODO
			for (int d = 0; d < ilog2(newSize); d++) {
				dim3 fullBlocksPerGrid((((int)(newSize / (1 << (d + 1)))) + radixBlockSize - 1) / radixBlockSize);
				upSweep << <fullBlocksPerGrid, radixBlockSize >> > (newSize, d, odata);
			}


			cudaMemset(odata + (newSize - 1), 0, 1 * sizeof(int));

			for (int d = ilog2(newSize) - 1; d >= 0; d--) {
				dim3 fullBlocksPerGrid(((1 << (ilog2(newSize) - d)) + radixBlockSize - 1) / radixBlockSize);
				downSweep << <fullBlocksPerGrid, radixBlockSize >> > (newSize, d, odata);
			}
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
            // TODO



			cudaMalloc((void**)&devIdataEfficientCompact, n * sizeof(int));
			checkCUDAError("cudaMalloc devIdataEfficientCompact failed");
			cudaMemcpy(devIdataEfficientCompact, idata, n * sizeof(int), cudaMemcpyHostToDevice);

			int *binaryMap = new int[n];
			cudaMalloc((void**)&devIdataEfficientBinaryMap, n * sizeof(int));
			checkCUDAError("cudaMalloc devIdataEfficientBinaryMap failed");
			cudaMemset(devIdataEfficientBinaryMap, 0, n * sizeof(int));

			int *newIndices = new int[n];
			cudaMalloc((void**)&devIdataEfficientNewIndices, n * sizeof(int));
			checkCUDAError("cudaMalloc devIdataEfficientNewIndices failed");



			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
			StreamCompaction::Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> > (n, devIdataEfficientBinaryMap, devIdataEfficientCompact);
			cudaMemcpy(binaryMap, devIdataEfficientBinaryMap, n * sizeof(int), cudaMemcpyDeviceToHost);

			bool exception = true;
			try {
				timer().startGpuTimer();
				exception = false;
			}
			catch (const std::exception& e) {
				exception = true;
			}
			scan(n, newIndices, binaryMap);
			cudaMemcpy(devIdataEfficientNewIndices, newIndices, n * sizeof(int), cudaMemcpyHostToDevice);

			int newSize = newIndices[n - 1] + binaryMap[n - 1];
			cudaMalloc((void**)&devIdataEfficient, newSize * sizeof(int));
			checkCUDAError("cudaMalloc devIdataEfficient failed");


			StreamCompaction::Common::kernScatter << <fullBlocksPerGrid, blockSize >> > (n, devIdataEfficient, devIdataEfficientCompact, devIdataEfficientBinaryMap, devIdataEfficientNewIndices);

			try {
				if (exception == false) {
					timer().endGpuTimer();
				}
			}
			catch (const std::exception& e) {

			}
			cudaMemcpy(odata, devIdataEfficient, newSize * sizeof(int), cudaMemcpyDeviceToHost);

			cudaFree(devIdataEfficient);
			cudaFree(devIdataEfficientBinaryMap);
			cudaFree(devIdataEfficientCompact);
			cudaFree(devIdataEfficientNewIndices);


            return newSize;
        }
    }
}
