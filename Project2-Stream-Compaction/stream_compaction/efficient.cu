#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

		int blockSize = 128;
		dim3 threadsPerBlock(blockSize);

		__global__ void kernUpSweep(int N, int power, int *opArray) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= N) {
				return;
			}

			int k = index * 2 * power;

			opArray[k + 2 * power - 1] += opArray[k + power - 1];
		}

		__global__ void kernDownSweep(int N, int power, int *opArray) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= N) {
				return;
			}

			int k = index * 2 * power;

			int t = opArray[k + power - 1];
			int s = opArray[k + 2 * power - 1];
			opArray[k + power - 1] = s;
			opArray[k + 2 * power - 1] = s + t;
		}

		__global__ void kernSetLastZero(int N, int offset, int *opArray) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= N) {
				return;
			}
			opArray[index + offset] = 0;
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			int bufferLength = 1 << ilog2ceil(n);
			int *dev_inputArray;
			int *host_upSweep;

			cudaMalloc((void**)&dev_inputArray, bufferLength * sizeof(int));
			checkCUDAError("cudaMalloc dev_inputArray failed!");

			cudaMallocHost((void**)&host_upSweep, bufferLength * sizeof(int));
			checkCUDAError("cudaMallocHost host_upSweep failed!");

			cudaMemset(dev_inputArray, 0, bufferLength * sizeof(int));

			cudaMemcpy(dev_inputArray, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();

			for (int d = 0; d < ilog2ceil(n); ++d) {
				int power = pow(2, d);
				int numThreads = bufferLength / (2 * power);

				dim3 blocksPerGrid((numThreads + blockSize - 1) / blockSize);
				kernUpSweep << <blocksPerGrid, threadsPerBlock >> > (numThreads, power, dev_inputArray);
				checkCUDAError("kernUpSweep failed!");
			}

			// NOTE: dev_inputArray is now in upsweep stage

			int numThreadsSetZero = 1;
			dim3 blocksPerGridSetZero((numThreadsSetZero + blockSize - 1) / blockSize);
			kernSetLastZero << <blocksPerGridSetZero, threadsPerBlock >> > (numThreadsSetZero, bufferLength - 1, dev_inputArray);

			for (int d = ilog2ceil(n) - 1; d >= 0; --d) {
				int power = pow(2, d);
				int numThreads = bufferLength / (2 * power);
				dim3 blocksPerGrid((numThreads + blockSize - 1) / blockSize);
				kernDownSweep << <blocksPerGrid, threadsPerBlock >> > (numThreads, power, dev_inputArray);
				checkCUDAError("kernDownSweep failed!");
			}
			cudaMemcpy(odata, dev_inputArray, n * sizeof(int), cudaMemcpyDeviceToHost);

            // TODO
            timer().endGpuTimer();
			cudaFree(dev_inputArray);
			cudaFreeHost(host_upSweep);

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
