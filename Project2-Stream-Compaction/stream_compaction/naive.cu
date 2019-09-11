#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include <iostream>
#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
		__global__ void naive_parallel_scan(int n, int *odata, const int *idata, int d) {
			int index = blockDim.x * blockIdx.x + threadIdx.x;
			if (index > n)
				return;
			if (index >= powf(2,d-1))
				odata[index] = idata[index - (int)powf(2,d - 1)] + idata[index]; // todo figure out why this doesnt work with non powers of 2
			else
				odata[index] = idata[index];
		}
		__global__ void right_shift(int n, int *odata, const int *idata) {
			int index = blockDim.x * blockIdx.x + threadIdx.x;
			if (index > n)
				return;
			if (index == 0)
				odata[index] = 0;
			else
				odata[index] = idata[index - 1];
		}
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
			int blocksize = 128;
			int blocks = (n + blocksize - 1) / blocksize;
			// allocate data
			int *dev_odata, *dev_odata_2;
			cudaMalloc((void**)&dev_odata, n * sizeof(int));
			cudaMalloc((void**)&dev_odata_2, n * sizeof(int));
			// copy data over
			cudaMemcpy(dev_odata, idata, n*sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAErrorWithLine("memcpy back failed!");
			for (int d = 1; d <= ceil(log2(n)); d++) {
				naive_parallel_scan <<<blocks, blocksize >> > (n, dev_odata_2, dev_odata, d);
				checkCUDAErrorWithLine("copy fn failed!");
				std::swap(dev_odata, dev_odata_2);
			}
			right_shift << <blocks, blocksize >> > (n, dev_odata_2, dev_odata);
			cudaMemcpy(odata, dev_odata_2, n*sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAErrorWithLine("memcpy back failed!");
			cudaFree(dev_odata);
			cudaFree(dev_odata_2);
			// create buffer
            timer().endGpuTimer();
        }
    }
}
