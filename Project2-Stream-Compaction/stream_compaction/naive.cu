#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include <algorithm>

namespace StreamCompaction {
	namespace Naive {
		using StreamCompaction::Common::PerformanceTimer;
		PerformanceTimer& timer()
		{
			static PerformanceTimer timer;
			return timer;
		}

		// TODO: __global__
		__global__
		void kernScan(unsigned int n, int offset, int* odata, int* idata) {

			int k = blockIdx.x * blockDim.x + threadIdx.x;
			if (k >= n) return;

			if (k >= offset)
				odata[k] = idata[k - offset] + idata[k];
			else
				odata[k] = idata[k];
		}

		__global__
		void kernRightShift(unsigned int n, int* odata, int* idata) {
			// Shift right by one -> get exclusive scan
			int k = blockIdx.x * blockDim.x + threadIdx.x;
			if (k >= n) return;
			
			odata[k] = k > 0 ? idata[k - 1] : 0;
		}

		/**
		 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
		 */
		void scan(int n, int* odata, const int* idata) {

			int* d_odata;
			int* d_idata;
			cudaMalloc(&d_odata, n * sizeof(int));
			cudaMalloc(&d_idata, n * sizeof(int));
			cudaMemcpy(d_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

			int threadsPerBlock = 512;
			int blockSize = (n + threadsPerBlock - 1) / threadsPerBlock;

			timer().startGpuTimer();
			for (int offset = 1; offset < (n << 1); offset *= 2) {
				kernScan << <blockSize, threadsPerBlock >> > (n, offset, d_odata, d_idata);
				cudaDeviceSynchronize();
				// swap in and out
				int* tmp = d_odata;
				d_odata = d_idata;
				d_idata = tmp;
			}
			timer().endGpuTimer();

			cudaMemcpy(d_idata, d_odata, n * sizeof(int), cudaMemcpyDeviceToDevice);
			kernRightShift << <blockSize, threadsPerBlock >> > (n, d_odata, d_idata);

			cudaMemcpy(odata, d_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
			cudaFree(d_idata);
			cudaFree(d_odata);

		}
	}
}
