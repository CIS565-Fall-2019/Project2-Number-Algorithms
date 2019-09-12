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
			int alternator = 0;
			for (int d = 0; d < ilog2ceil(n); ++d) {
				int numThreads = pow(2, d);
				dim3 blocksPerGrid((numThreads + blockSize - 1) / blockSize);

				if (alternator % 2 == 0) {
					kernSumPairs<<<blocksPerGrid, threadsPerBlock>>>(numThreads, d, dev_arrayA, dev_arrayB);
					cudaMemcpy(dev_arrayA, dev_arrayB, bufferLength * sizeof(int), cudaMemcpyDeviceToDevice);
					alternator++;

				}
				else {
					kernSumPairs<<<blocksPerGrid, threadsPerBlock>>>(numThreads, d, dev_arrayB, dev_arrayA);
					cudaMemcpy(dev_arrayA, dev_arrayB, bufferLength * sizeof(int), cudaMemcpyDeviceToDevice);
					alternator++;

				}

			}


            timer().endGpuTimer();


			cudaFree(dev_arrayA);
			cudaFree(dev_arrayB);
        }
    }
}
