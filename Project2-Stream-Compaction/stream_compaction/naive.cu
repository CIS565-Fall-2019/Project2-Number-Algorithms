#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

/*! Block size used for CUDA kernel launch*/
#define blockSize 512
int *dev_A;
int *dev_B;

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

		__global__ void kernNaiveScan(int n, int curr_level, int* devA, int* devB) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= n) {
				return;
			}
			int offset = (int)powf(2, curr_level - 1);
			if (index >= offset) {
				devB[index] = devA[index - offset] + devA[index];
			}
			else {
				devB[index] =  devA[index];
			}
		}

		int nextPowerOf2(int n) {
			int p = 1;
			if (n && !(n & (n - 1))) {
				return n;
			}
			while (p < n) {
				p <<= 1;
			}
			return p;
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			int malloc_size = nextPowerOf2(n);
			//CUDA Malloc buffers
			cudaMalloc((void**)&dev_A, malloc_size * sizeof(int));
			checkCUDAError("cudaMalloc dev_A failed!");
			cudaMalloc((void**)&dev_B, malloc_size * sizeof(int));
			checkCUDAError("cudaMalloc dev_A failed!");

			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
			int max_level = ilog2ceil(n);
			
            timer().startGpuTimer();
			//Copy idata into dev_A
			cudaMemcpy(dev_A, idata, n * sizeof(int), cudaMemcpyHostToDevice);

			//loop over each level
			for (int curr_level = 1; curr_level <= max_level; curr_level++) {
				//Launch Kernel (thereby updating dev_B)
				kernNaiveScan<<<fullBlocksPerGrid, blockSize>>>(n, curr_level, dev_A, dev_B);

				//Copy dev_B's updated data into dev_A
				cudaMemcpy(dev_A, dev_B, n * sizeof(int), cudaMemcpyDeviceToDevice);
				checkCUDAError("cudaMemcpy dev_A to dev_B failed!");
			}
			//Exclusive Scan so shift right when copying back
			cudaMemcpy(odata+1, dev_A, (n-1) * sizeof(int), cudaMemcpyDeviceToHost);
			odata[0] = 0;
			checkCUDAError("cudaMemcpy dev_A to out failed!");
            timer().endGpuTimer();

			//Free Memory
			cudaFree(dev_A);
			cudaFree(dev_B);
        }
    }
}
