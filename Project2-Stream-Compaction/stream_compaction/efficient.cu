#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include <iostream>
#include <stream_compaction/cpu.h>
#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)


namespace StreamCompaction {
	#define block_size 256
	using StreamCompaction::Common::PerformanceTimer;
	
	namespace Efficient {
		PerformanceTimer& timer()
		{
			static PerformanceTimer timer;
			return timer;
		}
		/*2 scan phases, see (link)[https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html] for more details*/
		__global__ void reduce_parallel(unsigned long long int n, long long *data, long long d) {
			unsigned long long int tmp_d = 1 << (d + 1);
			unsigned long long int index = (blockDim.x * blockIdx.x + threadIdx.x)*tmp_d;
			if (index >= n )
				return;
			data[index + tmp_d - 1] += data[index + (tmp_d>>1) - 1];
		}

		__global__ void downsweep_parallel(unsigned long long int n, long long *data, long long d) {
			unsigned long long int new_d = 1<<(d + 1);
			unsigned long long int index = (blockDim.x * blockIdx.x + threadIdx.x)*new_d;
			if (index >= n)
				return;
			int t = data[index + (new_d>>1) - 1];
			data[index + (new_d>>1) - 1] = data[index + new_d - 1];
			data[index + new_d - 1] += t;
		}
		/**
		 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
		 */
		void scan(unsigned long long int n, long long *odata, const long long *idata) {
			timer().startGpuTimer();
			// allocate pointers to memory and copy data over
			long long *dev_odata;
			unsigned long long int blocks = 0;
			unsigned long long int closest_pow2 = 1<<ilog2ceil(n);
			cudaMalloc((void**)&dev_odata, closest_pow2 * sizeof(long long));
			checkCUDAErrorWithLine("malloc failed!");
			// we can use dev_odata to hold idata (because we do the scan inplace)
			cudaMemcpy(dev_odata, idata, n * sizeof(long long), cudaMemcpyHostToDevice);
			checkCUDAErrorWithLine("memcpy failed!");
			// reduce phase
			// so we dont need to do the last round of computation because we zero it anyway
			for (long long d = 0; d <= ilog2ceil(closest_pow2) - 2; d++) {
				// compute number of threads to spawn
				blocks = ceil((closest_pow2 / (1<<(d + 1)) + block_size - 1) / block_size);
				reduce_parallel <<<blocks, block_size >>> (closest_pow2, dev_odata, d);
				checkCUDAErrorWithLine("reduce phase failed!");
			}
			// down-sweep phase
			// zero last value
			cudaMemset(dev_odata + (closest_pow2 - 1), 0, 1 * sizeof(long long));
			for (long long d = ilog2ceil(closest_pow2) - 1; d >= 0; d--) {
				blocks = ceil((closest_pow2 / (1 << (d + 1)) + block_size - 1) / block_size);
				downsweep_parallel <<<blocks, block_size >>> (closest_pow2, dev_odata, d);
				checkCUDAErrorWithLine("downsweep phase failed!");
			}
			//read data back
			cudaMemcpy(odata, dev_odata, n * sizeof(long long), cudaMemcpyDeviceToHost);
			checkCUDAErrorWithLine("memcpy back failed!");
			cudaFree(dev_odata);
			timer().endGpuTimer();
		}

		/*Copy of scan but only works with cuda pointers*/
		void dev_scan(unsigned long long int n, long long *dev_odata) {
			// allocate pointers to memory and copy data over
			unsigned long long int blocks = 0;
			unsigned long long int closest_pow2 = 1 << ilog2ceil(n);
			// reduce phase
			// so we dont need to do the last round of computation because we zero it anyway
			for (long long d = 0; d <= ilog2ceil(closest_pow2) - 2; d++) {
				// compute number of threads to spawn
				blocks = (closest_pow2 / (1 << (d + 1)) + block_size - 1) / block_size;
				reduce_parallel << <blocks, block_size >> > (closest_pow2, dev_odata, d);
				checkCUDAErrorWithLine("reduce phase failed!");
			}
			// down-sweep phase
			// zero last value
			cudaMemset(dev_odata + (closest_pow2 - 1), 0, 1 * sizeof(long long)); // might not actually work if not int, todo check
			for (long long d = ceil(log2(closest_pow2) - 1); d >= 0; d--) {
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
		unsigned long long int compact(unsigned long long int n, long long *odata, const long long *idata) {
			timer().startGpuTimer();
			unsigned long long int closest_pow2 = 1<<ilog2ceil(n);
			long long *dev_idata, *dev_odata, *dev_mask, *dev_indices;
			unsigned long long int blocks = ceil((closest_pow2 + block_size - 1) / block_size);
			cudaMalloc((void**)&dev_idata, closest_pow2 * sizeof(long long));
			cudaMalloc((void**)&dev_odata, closest_pow2 * sizeof(long long));
			cudaMalloc((void**)&dev_mask, closest_pow2 * sizeof(long long));
			cudaMalloc((void**)&dev_indices, closest_pow2 * sizeof(long long));
			// copy over idata
			cudaMemcpy(dev_idata, idata, n * sizeof(long long), cudaMemcpyHostToDevice);
			checkCUDAErrorWithLine("mask gen failed!");
			Common::kernMapToBoolean <<<blocks, block_size >> > (n, dev_mask, dev_idata);
			checkCUDAErrorWithLine("mask gen failed!");
			// scan the mask array (can be done in parallel by using a balanced binary tree)
			cudaMemcpy(dev_indices, dev_mask, closest_pow2 * sizeof(long long), cudaMemcpyDeviceToDevice);
			dev_scan(closest_pow2, dev_indices);
			checkCUDAErrorWithLine("dev scan failed!");
			// Scatter array (go to each position and copy the value)
			Common::kernScatter<<<blocks, block_size >> > (closest_pow2, dev_odata, dev_idata, dev_mask, dev_indices);

			checkCUDAErrorWithLine("scatter failed!");
			//read data back
			long long res;
			cudaMemcpy(&res, dev_indices + n - 1, sizeof(long long), cudaMemcpyDeviceToHost);
			res = idata[n - 1] ? res + 1 : res;
			cudaMemcpy(odata, dev_odata, n * sizeof(long long), cudaMemcpyDeviceToHost);
			cudaFree(dev_odata);
			cudaFree(dev_idata);
			cudaFree(dev_mask);
			cudaFree(dev_indices);
			timer().endGpuTimer();
			return res;
		}
	}
	namespace SharedMemory {
		PerformanceTimer& timer()
		{
			static PerformanceTimer timer;
			return timer;
		}
		#define NUM_BANKS 16
		#define LOG_NUM_BANKS 4
		#define CONFLICT_FREE_OFFSET(n) \
			((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))
		__global__ void dev_scan(unsigned long long int n, long long *dev_odata, long long *dev_idata, long long *dev_block_sum)
		{
			/* Extriemly heavly based on https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html
			   Main change is to use block_size instead of n and then put the sums of each block into a new array and scaning the array
			*/
			// Declare Share Memory
			__shared__ long long temp[block_size + NUM_BANKS];
			unsigned long long int thid = threadIdx.x;
			unsigned long long int bid = blockIdx.x;
			unsigned long long int scan_offset = bid * block_size;
			int offset = 1; // to make this an exclusive scan
			unsigned long long int ai = thid<<1;
			unsigned long long int bi = ai + 1;
			unsigned long long int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
			unsigned long long int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
			temp[ai + bankOffsetA] = dev_idata[ai + scan_offset];
			temp[bi + bankOffsetB] = dev_idata[bi + scan_offset];
			for (long long int d = block_size >> 1; d > 0; d >>= 1)                    // build sum in place up the tree
			{
				__syncthreads();
				if (thid < d)
				{
					unsigned long long int ai = offset * ((thid<<1) + 1) - 1;
					unsigned long long int bi = offset * ((thid << 1) + 2) - 1;
					ai += CONFLICT_FREE_OFFSET(ai);
					bi += CONFLICT_FREE_OFFSET(bi);
					temp[bi] += temp[ai];
				}
				offset<<=1;
			}
			__syncthreads();
			if (thid == 0) { 
				// place final sums of each block into extra array
				dev_block_sum[bid] = temp[block_size - 1 + CONFLICT_FREE_OFFSET(block_size - 1)];
				// zero last x cells because we are going to shift them when we do the downsweep
				temp[block_size - 1 + CONFLICT_FREE_OFFSET(block_size - 1)] = 0; 
			}
			// downsweep
			for (unsigned long long int d = 1; d < block_size; d<<=1) // traverse down tree & build scan
			{
				offset >>= 1;
				__syncthreads();
				if (thid < d)
				{
					unsigned long long int ai = offset * ((thid << 1) + 1) - 1;
					unsigned long long int bi = offset * ((thid << 1) + 2) - 1;
					ai += CONFLICT_FREE_OFFSET(ai);
					bi += CONFLICT_FREE_OFFSET(bi);
					long long t = temp[ai];
					temp[ai] = temp[bi];
					temp[bi] += t;
				}
			}
			__syncthreads();
			dev_odata[ai + scan_offset] = temp[ai + bankOffsetA];
			dev_odata[bi + scan_offset] = temp[bi + bankOffsetB];
		}
		__global__ void add_offset(unsigned long long int n, long long *data, long long *dev_block_offset) {
			unsigned long long int bid = blockIdx.x;
			unsigned long long int index = blockDim.x * bid + threadIdx.x;
			if (index >= n)
				return;
			// add value to current section
			if (bid != 0) // to save a bunch of useless reads and write
				data[index] += dev_block_offset[bid];
		}
		void scan(unsigned long long int n, long long *odata, long long *idata) {
			timer().startGpuTimer();
			// allocate pointers to memory and copy data over
			long long *dev_odata, *dev_idata, *dev_block_sum, *dev_block_offset;
			long long *block_sum, *block_offset;
			unsigned long long int closest_pow2 = 1 << ilog2ceil(n);
			unsigned long long int blocks = ceil((closest_pow2 + block_size - 1) / block_size);
			// allocate buffers
			cudaMalloc((void**)&dev_odata, closest_pow2 * sizeof(long long));
			cudaMalloc((void**)&dev_idata, closest_pow2 * sizeof(long long));
			// allocate block global buffers
			cudaMalloc((void**)&dev_block_sum, blocks * sizeof(long long)); // each block gets 1 number to fill in 
			cudaMalloc((void**)&dev_block_offset, blocks * sizeof(long long));
			// cpu scan buffers
			block_sum = new long long[blocks]();
			block_offset = new long long[blocks]();
			checkCUDAErrorWithLine("malloc failed!");
			// copy over raw data
			cudaMemcpy(dev_idata, idata, n * sizeof(long long), cudaMemcpyHostToDevice);
			checkCUDAErrorWithLine("memcpy failed!");
			// large prescan, pre alloc shared memory
			dev_scan<<<blocks, (block_size >> 1)>>>(closest_pow2, dev_odata, dev_idata, dev_block_sum);
			checkCUDAErrorWithLine("prescan fn failed!");
			// cpu scan the remaining blocks, because otherwise it could become recursive for large numbers
			cudaMemcpy(block_sum, dev_block_sum, blocks * sizeof(long long), cudaMemcpyDeviceToHost);
			checkCUDAErrorWithLine("memcpy to cpu failed!");
			StreamCompaction::CPU::scan(blocks, block_offset, block_sum);
			// copy data back over to cuda
			cudaMemcpy(dev_block_offset, block_offset, blocks * sizeof(long long), cudaMemcpyHostToDevice);
			checkCUDAErrorWithLine("memcpy from cpu failed!");
			// add dev_block_offset to each block
			add_offset <<<blocks, block_size>>> (n, dev_odata, dev_block_offset);
			checkCUDAErrorWithLine("add_offset fn failed!");
			cudaMemcpy(odata, dev_odata, n * sizeof(long long), cudaMemcpyDeviceToHost);
			checkCUDAErrorWithLine("memcpy back failed!");
			// free memory
			cudaFree(dev_odata);
			cudaFree(dev_idata);
			cudaFree(dev_block_sum);
			cudaFree(dev_block_offset);
			delete[] block_sum;
			delete[] block_offset;
			timer().endGpuTimer();
		}
		void dev_scan(unsigned long long int n, long long *dev_odata, long long *dev_idata) {
			// allocate pointers to memory and copy data over
			long long *dev_block_sum, *dev_block_offset;
			long long *block_sum, *block_offset;
			unsigned long long int closest_pow2 = 1 << ilog2ceil(n);
			unsigned long long int blocks = ceil((closest_pow2 + block_size - 1) / block_size);
			// allocate buffers
			// allocate block global buffers
			cudaMalloc((void**)&dev_block_sum, blocks * sizeof(long long)); // each block gets 1 number to fill in 
			cudaMalloc((void**)&dev_block_offset, blocks * sizeof(long long));
			// cpu scan buffers
			block_sum = new long long[blocks]();
			block_offset = new long long[blocks]();
			checkCUDAErrorWithLine("malloc failed!");
			// copy over raw data
			// large prescan, pre alloc shared memory
			dev_scan << <blocks, (block_size >> 1) >> > (n, dev_odata, dev_idata, dev_block_sum); //todo maybe change back n to closest_pow2
			checkCUDAErrorWithLine("prescan fn failed!");
			// cpu scan the remaining blocks, because otherwise it could become recursive for large numbers
			cudaMemcpy(block_sum, dev_block_sum, blocks * sizeof(long long), cudaMemcpyDeviceToHost);
			checkCUDAErrorWithLine("memcpy to cpu failed!");
			StreamCompaction::CPU::scan(blocks, block_offset, block_sum);
			// copy data back over to cuda
			cudaMemcpy(dev_block_offset, block_offset, blocks * sizeof(long long), cudaMemcpyHostToDevice);
			checkCUDAErrorWithLine("memcpy from cpu failed!");
			// add dev_block_offset to each block
			add_offset << <blocks, block_size >> > (n, dev_odata, dev_block_offset);
			checkCUDAErrorWithLine("add_offset fn failed!");
			// free memory
			cudaFree(dev_block_sum);
			cudaFree(dev_block_offset);
			delete[] block_sum;
			delete[] block_offset;
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
		unsigned long long int compact(unsigned long long int n, long long *odata, const long long *idata) {
			timer().startGpuTimer();
			unsigned long long int closest_pow2 = 1 << ilog2ceil(n);
			long long *dev_idata, *dev_odata, *dev_mask, *dev_indices;
			unsigned long long int blocks = ceil((closest_pow2 + block_size - 1) / block_size);
			cudaMalloc((void**)&dev_idata, closest_pow2 * sizeof(long long));
			cudaMalloc((void**)&dev_odata, closest_pow2 * sizeof(long long));
			cudaMalloc((void**)&dev_mask, closest_pow2 * sizeof(long long));
			cudaMalloc((void**)&dev_indices, closest_pow2 * sizeof(long long));
			// copy over idata
			cudaMemcpy(dev_idata, idata, n * sizeof(long long), cudaMemcpyHostToDevice);
			checkCUDAErrorWithLine("mask gen failed!");
			Common::kernMapToBoolean << <blocks, block_size >> > (n, dev_mask, dev_idata);
			checkCUDAErrorWithLine("mask gen failed!");
			// scan the mask array (can be done in parallel by using a balanced binary tree)
			cudaMemcpy(dev_indices, dev_mask, closest_pow2 * sizeof(long long), cudaMemcpyDeviceToDevice);
			dev_scan(closest_pow2, dev_indices, dev_mask);
			checkCUDAErrorWithLine("dev scan failed!");
			// Scatter array (go to each position and copy the value)
			Common::kernScatter << <blocks, block_size >> > (closest_pow2, dev_odata, dev_idata, dev_mask, dev_indices);

			checkCUDAErrorWithLine("scatter failed!");
			//read data back
			long long res;
			cudaMemcpy(&res, dev_indices + n - 1, sizeof(long long), cudaMemcpyDeviceToHost);
			res = idata[n - 1] ? res + 1 : res;
			cudaMemcpy(odata, dev_odata, n * sizeof(long long), cudaMemcpyDeviceToHost);
			cudaFree(dev_odata);
			cudaFree(dev_idata);
			cudaFree(dev_mask);
			cudaFree(dev_indices);
			timer().endGpuTimer();
			return res;
		}
	}
}
