#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include <iostream>
#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

namespace StreamCompaction {
	namespace Efficient {
		int block_size = 128;
		using StreamCompaction::Common::PerformanceTimer;
		PerformanceTimer& timer()
		{
			static PerformanceTimer timer;
			return timer;
		}
		/*2 scan phases, see (link)[https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html] for more details*/
		__global__ void reduce_parallel(int n, int *data, int d) {
			int tmp_d = 1 << (d + 1);
			int index = (blockDim.x * blockIdx.x + threadIdx.x)*tmp_d;
			if (index >= n )
				return;
			data[index + tmp_d - 1] += data[index + (tmp_d>>1) - 1];
		}

		__global__ void downsweep_parallel(int n, int *data, int d) {
			int new_d = 1<<(d + 1);
			int index = (blockDim.x * blockIdx.x + threadIdx.x)*new_d;
			if (index >= n)
				return;
			int t = data[index + (new_d>>1) - 1];
			data[index + (new_d>>1) - 1] = data[index + new_d - 1];
			data[index + new_d - 1] += t;
		}
		/**
		 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
		 */
		void scan(int n, int *odata, const int *idata) {
			timer().startGpuTimer();
			// allocate pointers to memory and copy data over
			int *dev_odata;
			int blocks = 0;
			int closest_pow2 = 1<<ilog2ceil(n);
			cudaMalloc((void**)&dev_odata, closest_pow2 * sizeof(int));
			checkCUDAErrorWithLine("malloc failed!");
			cudaMemcpy(dev_odata, idata, closest_pow2 * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAErrorWithLine("memcpy failed!");
			// reduce phase
			// so we dont need to do the last round of computation because we zero it anyway
			for (int d = 0; d <= ilog2ceil(closest_pow2) - 2; d++) {
				// compute number of threads to spawn
				blocks = ceil((closest_pow2 / (1<<(d + 1)) + block_size - 1) / block_size);
				reduce_parallel <<<blocks, block_size >>> (closest_pow2, dev_odata, d);
				checkCUDAErrorWithLine("reduce phase failed!");
			}
			// down-sweep phase
			// zero last value
			cudaMemset(dev_odata + (closest_pow2 - 1), 0, 1 * sizeof(int));
			for (int d = ceil(log2(closest_pow2) - 1); d >= 0; d--) {
				blocks = ceil((closest_pow2 / (1 << (d + 1)) + block_size - 1) / block_size);
				downsweep_parallel <<<blocks, block_size >>> (closest_pow2, dev_odata, d);
				checkCUDAErrorWithLine("downsweep phase failed!");
			}
			//read data back
			cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAErrorWithLine("memcpy back failed!");
			timer().endGpuTimer();
		}

		/*Copy of scan but only works with cuda pointers*/
		void dev_scan(int n, int *dev_odata) {
			// allocate pointers to memory and copy data over
			int blocks = 0;
			int closest_pow2 = 1 << ilog2ceil(n);
			// reduce phase
			// so we dont need to do the last round of computation because we zero it anyway
			for (int d = 0; d <= ilog2ceil(closest_pow2) - 2; d++) {
				// compute number of threads to spawn
				blocks = (closest_pow2 / (1 << (d + 1)) + block_size - 1) / block_size;
				reduce_parallel << <blocks, block_size >> > (closest_pow2, dev_odata, d);
				checkCUDAErrorWithLine("reduce phase failed!");
			}
			// down-sweep phase
			// zero last value
			cudaMemset(dev_odata + (closest_pow2 - 1), 0, 1 * sizeof(int));
			for (int d = ceil(log2(closest_pow2) - 1); d >= 0; d--) {
				blocks = (closest_pow2 / (1 << (d + 1)) + block_size - 1) / block_size;
				downsweep_parallel << <blocks, block_size >> > (closest_pow2, dev_odata, d);
				checkCUDAErrorWithLine("downsweep phase failed!");
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
			timer().startGpuTimer();
			int closest_pow2 = 1<<ilog2ceil(n);
			int *dev_idata, *dev_odata, *dev_mask, *dev_indices;
			int blocks = ceil((closest_pow2 + block_size - 1) / block_size);
			cudaMalloc((void**)&dev_idata, closest_pow2 * sizeof(int));
			cudaMalloc((void**)&dev_odata, closest_pow2 * sizeof(int));
			cudaMalloc((void**)&dev_mask, closest_pow2 * sizeof(int));
			cudaMalloc((void**)&dev_indices, closest_pow2 * sizeof(int));
			// copy over idata
			cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAErrorWithLine("mask gen failed!");
			Common::kernMapToBoolean <<<blocks, block_size >> > (n, dev_mask, dev_idata);
			checkCUDAErrorWithLine("mask gen failed!");
			// scan the mask array (can be done in parallel by using a balanced binary tree)
			cudaMemcpy(dev_indices, dev_mask, closest_pow2 * sizeof(int), cudaMemcpyDeviceToDevice);
			dev_scan(closest_pow2, dev_indices);
			checkCUDAErrorWithLine("dev scan failed!");
			// Scatter array (go to each position and copy the value)
			Common::kernScatter<<<blocks, block_size >> > (closest_pow2, dev_odata, dev_idata, dev_mask, dev_indices);

			checkCUDAErrorWithLine("scatter failed!");
			//read data back
			int res;
			cudaMemcpy(&res, dev_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
			res = idata[n - 1] ? res + 1 : res;
			cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
			cudaFree(dev_odata);
			cudaFree(dev_idata);
			cudaFree(dev_mask);
			cudaFree(dev_indices);
			timer().endGpuTimer();
			return res;
		}
	}
}
