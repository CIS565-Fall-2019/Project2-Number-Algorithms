#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
		int blockSize = 128;
		dim3 threadsPerBlock(blockSize);

		__global__ void kernSumPairs(int N, int d, int *srcArray, int *dstArray) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= N) {
				return;
			}
			int power = powf(2, d - 1);
			index += power;
			dstArray[index] = srcArray[index - power] + srcArray[index];

			/*int power = powf(2, d - 1);
			if (index >= power) {
				dstArray[index] = srcArray[index - power] + srcArray[index];
			}*/
		}

		__global__ void kernShift(int N, int *srcArray, int *dstArray) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= N) {
				return;
			}

			if (index == 0) {
				dstArray[index] = 0;
				return;
			}
			dstArray[index] = srcArray[index - 1];

		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			int bufferLength = 1 << ilog2ceil(n);
			int *dev_arrayA;
			int *dev_arrayB;

			cudaMalloc((void**)&dev_arrayA, bufferLength * sizeof(int));
			checkCUDAError("cudaMalloc dev_arrayA failed!");
			
			cudaMalloc((void**)&dev_arrayB, bufferLength * sizeof(int));
			checkCUDAError("cudaMalloc dev_arrayB failed!");

			cudaMemset(dev_arrayA, 0, bufferLength * sizeof(int));
			cudaMemset(dev_arrayB, 0, bufferLength * sizeof(int));

			cudaMemcpy(dev_arrayA, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(dev_arrayB, idata, n * sizeof(int), cudaMemcpyHostToDevice);

			timer().startGpuTimer();

            // TODO
			bool alternator = true;
			for (int d = 1; d <= ilog2ceil(n); ++d) {
				int numThreads = bufferLength - pow(2, d - 1); // TODO: can this be smaller?
				
				dim3 blocksPerGrid((numThreads + blockSize - 1) / blockSize);

				if (alternator) {
					kernSumPairs<<<blocksPerGrid, threadsPerBlock>>>(numThreads, d, dev_arrayA, dev_arrayB);
					checkCUDAError("kernSumPairs failed!");

					cudaMemcpy(dev_arrayA, dev_arrayB, bufferLength * sizeof(int), cudaMemcpyDeviceToDevice);
					alternator = false;

				}
				else {
					kernSumPairs<<<blocksPerGrid, threadsPerBlock>>>(numThreads, d, dev_arrayB, dev_arrayA);
					checkCUDAError("kernSumPairs failed!");

					cudaMemcpy(dev_arrayB, dev_arrayA, bufferLength * sizeof(int), cudaMemcpyDeviceToDevice);
					alternator = true;

				}
			}
			// Note: dev_arrayA/B are now inclusive scans.  
			// We will take B and shift it on the gpu, storing the exlusive scan in A

			int numThreads = bufferLength;
			dim3 blocksPerGrid((numThreads + blockSize - 1) / blockSize);

			kernShift<<<blocksPerGrid, threadsPerBlock>>>(bufferLength * sizeof(int), dev_arrayB, dev_arrayA);
			checkCUDAError("kernShift failed!");

			timer().endGpuTimer();

			cudaMemcpy(odata, dev_arrayA, n * sizeof(int), cudaMemcpyDeviceToHost);

			cudaFree(dev_arrayA);
			cudaFree(dev_arrayB);
        }
    }
}
