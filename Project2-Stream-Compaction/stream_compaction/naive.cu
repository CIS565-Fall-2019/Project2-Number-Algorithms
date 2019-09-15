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
		__global__ void naive_parallel_scan(unsigned long long int n, int *odata, const int *idata, int d) {
			unsigned long long int index = blockDim.x * blockIdx.x + threadIdx.x;
			if (index >= n)
				return;
			if (index >= d)
				odata[index] = idata[index - d] + idata[index]; // todo figure out why this doesnt work with non powers of 2
			else
				odata[index] = idata[index];
		}
		__global__ void right_shift(unsigned long long int n, int *odata, const int *idata, int amount) {
			unsigned long long int index = blockDim.x * blockIdx.x + threadIdx.x;
			if (index >= n)
				return;
			if (index < amount)
				odata[index] = 0;
			else
				odata[index] = idata[index - amount];
		}
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(unsigned long long int n, int *odata, const int *idata) {
            timer().startGpuTimer();
			int blocksize = 128;
			unsigned long long int blocks = (n + blocksize - 1) / blocksize;
			// allocate data
			int *dev_odata, *dev_odata_2;
			cudaMalloc((void**)&dev_odata, n * sizeof(int));
			checkCUDAErrorWithLine("malloc failed!");
			cudaMalloc((void**)&dev_odata_2, n * sizeof(int));
			checkCUDAErrorWithLine("malloc failed!");
			// copy data over
			cudaMemcpy(dev_odata, idata, n*sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAErrorWithLine("memcpy failed!");
			unsigned long long int uppper_limit = 1 << ilog2ceil(n);
			for (unsigned long long int d = 1; d <= uppper_limit; d<<=1) {
				naive_parallel_scan <<<blocks, blocksize >> > (n, dev_odata_2, dev_odata, d);
				checkCUDAErrorWithLine("fn failed!");
				std::swap(dev_odata, dev_odata_2);
			}
			right_shift << <blocks, blocksize >> > (n, dev_odata_2, dev_odata, 1);
			checkCUDAErrorWithLine("right shift failed failed!");
			cudaMemcpy(odata, dev_odata_2, n*sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAErrorWithLine("memcpy back failed!");
			cudaFree(dev_odata);
			cudaFree(dev_odata_2);
			// create buffer
            timer().endGpuTimer();
        }
    }
}
