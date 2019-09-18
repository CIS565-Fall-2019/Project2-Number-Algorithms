#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include "common.h"
#include "efficient.h"

#define blockSize 128
namespace StreamCompaction {
	namespace Efficient {
		using StreamCompaction::Common::PerformanceTimer;
		PerformanceTimer& timer()
		{
			static PerformanceTimer timer;
			return timer;
		}
		__global__ void kernUpSweep(int n, int d, int *itemp) {
			int power = 1 << (d + 1);
			int k = ((blockIdx.x*blockDim.x) + threadIdx.x)*power;
			if (k >= n) {
				return;
			}
			int power_2 = 1 << d;
			itemp[k + power - 1] += itemp[k + power_2 - 1];
		}
		__global__ void kernDownSweep(int n, int d, int *itemp) {
			int power = 1 << (d + 1);
			int k = ((blockIdx.x*blockDim.x) + threadIdx.x)*power;
			if (k >= n) {
				return;
			}
			
			int power_2 = 1 << d;
			int temp = itemp[k + power_2 - 1];
			itemp[k + power_2 - 1] = itemp[k + power - 1];
			itemp[k + power - 1] += temp;
		}
		/**
		 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
		 */
		void scan(int n, int *odata, const int *idata) {
			int *scan;
			int D = ilog2ceil(n);
			int tot_size = (1 << D);
			cudaMalloc((void**)&scan, tot_size * sizeof(int));
			checkCUDAError("CUDA Malloc failed");
			cudaMemcpy(scan, idata, tot_size * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("Copy Failed");
			int timer_flag = 0;
			try {
				timer().startGpuTimer();
			}
			catch (...){
				timer_flag = 1;
			}
			for (int d = 0; d < ilog2ceil(tot_size); d++) {
				dim3 fullBlocksPerGrid (((1 << (D - d - 1)) + blockSize - 1) / blockSize);
				kernUpSweep << <fullBlocksPerGrid, blockSize >> > (tot_size, d, scan);
			}
			cudaMemset(scan + tot_size - 1, 0, sizeof(int));
			for (int d = ilog2ceil(tot_size)-1; d >= 0; d--) {
				dim3 fullBlocksPerGrid(((1 << (D - d - 1)) + blockSize - 1) / blockSize);
				kernDownSweep << <fullBlocksPerGrid, blockSize >> > (tot_size, d, scan);
			}
			if(timer_flag == 0)
				timer().endGpuTimer();
			cudaMemcpy(odata, scan, n * sizeof(int), cudaMemcpyDeviceToHost);
			cudaFree(scan);
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
			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
			int *mask, *mask_scan, *itemp, *otemp;
			cudaMalloc((void**)&mask, n * sizeof(int));
			checkCUDAError("CUDA Malloc failed");
			cudaMalloc((void**)&mask_scan, n * sizeof(int));
			checkCUDAError("CUDA Malloc failed");
			cudaMalloc((void**)&itemp, n * sizeof(int));
			checkCUDAError("CUDA Malloc failed");
			cudaMalloc((void**)&otemp, n * sizeof(int));
			checkCUDAError("CUDA Malloc failed");

			cudaMemcpy(itemp, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("Copy Failed");
			timer().startGpuTimer();
			StreamCompaction::Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> > (n, mask, itemp);
			int *cpu_otemp = new int[n];
			int *cpu_itemp = new int[n];
			cudaMemcpy(cpu_itemp, mask, n * sizeof(int), cudaMemcpyDeviceToHost);
			scan(n, cpu_otemp, cpu_itemp);
			int elements = cpu_itemp[n - 1] == 0 ? cpu_otemp[n - 1] : cpu_otemp[n - 1] + 1;
			cudaMemcpy(mask_scan, cpu_otemp, n * sizeof(int), cudaMemcpyHostToDevice);
			StreamCompaction::Common::kernScatter << <fullBlocksPerGrid, blockSize >> > (n, otemp, itemp, mask, mask_scan);
			timer().endGpuTimer();

			cudaMemcpy(odata, otemp, n * sizeof(int), cudaMemcpyDeviceToHost);
			cudaFree(mask);
			cudaFree(itemp);
			cudaFree(otemp);
			cudaFree(mask_scan);
			return elements;
		}
	}
}