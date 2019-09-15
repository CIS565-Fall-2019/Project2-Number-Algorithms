#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include <iostream>
#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

namespace StreamCompaction {
    namespace Naive {
#define blocksize 128
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
		__global__ void naive_parallel_scan(unsigned long long int n, long long *odata, const long long *idata, long d) {
			unsigned long long int index = blockDim.x * blockIdx.x + threadIdx.x;
			if (index >= n)
				return;
			if (index >= d)
				odata[index] = idata[index - d] + idata[index]; // todo figure out why this doesnt work with non powers of 2
			else
				odata[index] = idata[index];
		}
		__global__ void right_shift(unsigned long long int n, long long *odata, const long long *idata, int amount) {
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
        void scan(unsigned long long int n, long long *odata, const long long *idata) {
			timer().startGpuTimer();
			unsigned long long int blocks = (n + blocksize - 1) / blocksize;
			// allocate data
			long long *dev_odata, *dev_odata_2;
			cudaMalloc((void**)&dev_odata, n * sizeof(long long));
			checkCUDAErrorWithLine("malloc failed!");
			cudaMalloc((void**)&dev_odata_2, n * sizeof(long long));
			checkCUDAErrorWithLine("malloc failed!");
			// copy data over
			cudaMemcpy(dev_odata, idata, n*sizeof(long long), cudaMemcpyHostToDevice);
			checkCUDAErrorWithLine("memcpy failed!");
			unsigned long long int uppper_limit = 1 << ilog2ceil(n);
			for (long d = 1; d <= uppper_limit; d<<=1) {
				naive_parallel_scan <<<blocks, blocksize >> > (n, dev_odata_2, dev_odata, d);
				checkCUDAErrorWithLine("fn failed!");
				std::swap(dev_odata, dev_odata_2);
			}
			right_shift <<<blocks, blocksize >>> (n, dev_odata_2, dev_odata, 1);
			checkCUDAErrorWithLine("right shift failed failed!");
			cudaMemcpy(odata, dev_odata_2, n*sizeof(long long), cudaMemcpyDeviceToHost);
			checkCUDAErrorWithLine("memcpy back failed!");
			cudaFree(dev_odata);
			cudaFree(dev_odata_2);
			// create buffer
            timer().endGpuTimer();
        }
    }
}
