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

			cudaMalloc((void**)&dev_inputArray, bufferLength * sizeof(int));
			checkCUDAError("cudaMalloc dev_inputArray failed!");

			cudaMemset(dev_inputArray, 0, bufferLength * sizeof(int));

			cudaMemcpy(dev_inputArray, idata, n * sizeof(int), cudaMemcpyHostToDevice);

			bool newTimer = true;
			if (timer().getCpuTimerStarted()) {
				newTimer = false;
			}
			if (newTimer) {
				timer().startCpuTimer();
			}
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
			if (newTimer) {
				timer().endCpuTimer();
			}
			cudaFree(dev_inputArray);
        }


		__global__ void kernComputeInOutArray(int N, int *srcArray, int *dstArray) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= N) {
				return;
			}
			dstArray[index] = (int)(srcArray[index] != 0);
		}

		__global__ void kernComputeSize(int N, int offset, int *dst, int *src1, int *src2) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= N) {
				return;
			}

			dst[index] = src1[index + offset] + src2[index + offset];
		}

		__global__ void kernScatter(int N, int *dst, int *src, int *indexFinder) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= N) {
				return;
			}

			int currVal = src[index];
			int dstIndex = -1;
			if (currVal != 0) {
				dstIndex = indexFinder[index];
			}
			if (dstIndex >= 0) {
				dst[dstIndex] = currVal;
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
			int bufferLength = 1 << ilog2ceil(n);
			int *dev_inputArray;
			int *dev_tempInOutArray;
			int *dev_scanArray;
			int *dev_resultLength;

			int *host_resultLength;

			cudaMalloc((void**)&dev_inputArray, bufferLength * sizeof(int));
			checkCUDAError("cudaMalloc dev_inputArray failed!");

			cudaMalloc((void**)&dev_tempInOutArray, bufferLength * sizeof(int));
			checkCUDAError("cudaMalloc dev_tempInOutArray failed!");

			cudaMalloc((void**)&dev_scanArray, bufferLength * sizeof(int));
			checkCUDAError("cudaMalloc dev_scanArray failed!");

			cudaMalloc((void**)&dev_resultLength, sizeof(int));
			checkCUDAError("cudaMalloc dev_resultLength failed!");

			cudaMallocHost((void**)&host_resultLength, sizeof(int));
			checkCUDAError("cudaMallocHost host_resultLength failed!");



			cudaMemset(dev_inputArray, 0, bufferLength * sizeof(int));

			cudaMemcpy(dev_inputArray, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            // TODO
			int numThreads = bufferLength;
			dim3 blocksPerGrid((numThreads + blockSize - 1) / blockSize);

			kernComputeInOutArray << <blocksPerGrid, threadsPerBlock >> > (numThreads, dev_inputArray, dev_tempInOutArray);
			checkCUDAError("kernComputeInOutArray failed!");

			scan(bufferLength, dev_scanArray, dev_tempInOutArray);
			
			// NOTE: dev_scanArray now holds bufferLength scanned values of the 0/1 array

			int numThreadsComputeSize = 1;
			dim3 blocksPerGridComputeSize((numThreadsComputeSize + blockSize - 1) / blockSize);
			kernComputeSize << <blocksPerGrid, threadsPerBlock >> > (numThreadsComputeSize, bufferLength - 1, dev_resultLength, dev_scanArray, dev_tempInOutArray);
			checkCUDAError("kernComputeSize failed!");

			cudaMemcpy(host_resultLength, dev_resultLength, sizeof(int), cudaMemcpyDeviceToHost);
			int length = host_resultLength[0];

			int *dev_final;
			cudaMallocHost((void**)&dev_final, length * sizeof(int));

			int numThreadsScatter = bufferLength;
			dim3 blocksPerGridScatter((numThreadsScatter + blockSize - 1) / blockSize);

			kernScatter << <blocksPerGrid, threadsPerBlock >> > (numThreadsScatter, dev_final, dev_inputArray, dev_scanArray);



			cudaMemcpy(odata, dev_final, length * sizeof(int), cudaMemcpyDeviceToHost);


            timer().endGpuTimer();
			cudaFree(dev_inputArray);
			cudaFree(dev_tempInOutArray);
			cudaFree(dev_scanArray);
			cudaFree(dev_resultLength);
			cudaFreeHost(host_resultLength);

            return length;
        }
    }
}
