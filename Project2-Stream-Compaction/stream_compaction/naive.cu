#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include "device_launch_parameters.h"

namespace StreamCompaction {
	namespace Naive {
		using StreamCompaction::Common::PerformanceTimer;
		PerformanceTimer& timer()
		{
			static PerformanceTimer timer;
			return timer;
		}
		// TODO: __global__
		__global__ void kernelUpdateStep(int n, int d, int* dest_data, int* src_data) {
			int k = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (k > n)
				return;
			int offset = 1 << (d - 1);
			if (k >= offset)
				dest_data[k] = src_data[k] + src_data[k - offset];
			else
				dest_data[k] = src_data[k];

		}
		
		/**
		 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
		 */
		void scan(int n, int *odata, const int *idata, int blockSize) {
			// Memory Allocation and Copying
			int *adata;
			int *bdata;
			cudaMalloc((void**)&adata, n * sizeof(int));
			checkCUDAErrorFn("cudaMalloc adata failed!");
			cudaMalloc((void**)&bdata, n * sizeof(int));
			checkCUDAErrorFn("cudaMalloc bdata failed!");
			cudaMemcpy(adata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

			timer().startGpuTimer();
			// TODO
			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
			for (int d = 1; d <= ilog2ceil(n); d++) {
				kernelUpdateStep <<<fullBlocksPerGrid, blockSize >>> (n, d, bdata, adata);
				std::swap(adata, bdata);
			}
			timer().endGpuTimer();

			// Memory De-allocation and copying
			odata[0] = 0;
			cudaMemcpy(odata + 1, adata, sizeof(int) * (n - 1), cudaMemcpyDeviceToHost);
			cudaFree(adata);
			cudaFree(bdata);
		}
	}
}