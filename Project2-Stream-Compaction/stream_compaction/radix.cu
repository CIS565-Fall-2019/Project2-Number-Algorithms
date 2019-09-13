#include <cuda.h>
#include <cuda_runtime.h>
#include "radix.h"
#include "stream_compaction/efficient.h"
#include "common.h"


namespace Sorting {
	namespace Radix {
		#define block_size 128
		__global__ void mask_generation(int n, int *dev_mask, int *dev_idata, int bitmask) {
			int index = blockDim.x * blockIdx.x + threadIdx.x;
			if (index >= n)
				return;
			dev_mask[index] = !((bool)(dev_idata[index] & bitmask));
		}
		__global__ void true_index_generation(int n, int *dev_t, int *dev_f, int total_falses) {
			int index = blockDim.x * blockIdx.x + threadIdx.x;
			if (index >= n)
				return;
			dev_t[index] = index - dev_f[index] + total_falses;
		}
		__global__ void reshuffle_mask(int n, int *dev_odata, int *dev_idata, int *dev_t, int *dev_f, int mask) {
			int index = blockDim.x * blockIdx.x + threadIdx.x;
			if (index >= n)
				return;
			int data = dev_idata[index];
			if ((bool)(data & mask)) 
				dev_odata[dev_t[index]] = data;
			else
				dev_odata[dev_f[index]] = data;
		}
		int countBits(int n)
		{
			int count = 0;
			// While loop will run until we get n = 0
			while (n)
			{
				count++;
				// We are shifting n to right by 1 
				// place as explained above
				n = n >> 1;
			}
			return count;
		}
		void sort(int n, int *odata, int *idata, int max_value) {
			int loop_count = countBits(max_value);//std::numeric_limits<int>::digits;
			int mask = 1, total_falses, tmp;
			int blocks = ceil((n + block_size - 1) / block_size);
			int *dev_data, *dev_f, *dev_t, *dev_data2;
			int *dev_mask;
			cudaMalloc((void**)&dev_data, n*sizeof(int));
			cudaMalloc((void**)&dev_data2, n * sizeof(int));
			cudaMalloc((void**)&dev_mask, n * sizeof(int));
			cudaMalloc((void**)&dev_f, n * sizeof(int));
			cudaMalloc((void**)&dev_t, n * sizeof(int));
			// copy over idata
			cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			for (int i = 0; i < loop_count; i++, mask <<= 1) {
				mask_generation <<<blocks, block_size >>> (n, dev_mask, dev_data, mask);
				// scan to get false position indices
				StreamCompaction::SharedMemory::dev_scan(n, dev_f, dev_mask);
				cudaMemcpy(&total_falses, dev_f + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
				cudaMemcpy(&tmp, dev_mask + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
				total_falses += tmp;
				// compute true position indices
				true_index_generation <<<blocks, block_size >>> (n, dev_t, dev_f, total_falses);
				// reshuffle data using correct positions
				reshuffle_mask << <blocks, block_size >> > (n, dev_data2, dev_data, dev_t, dev_f, mask);
				// copy data2 over to data
				cudaMemcpy(dev_data, dev_data2, n * sizeof(int), cudaMemcpyDeviceToDevice);
			}
			// we are done sorting, copy back
			cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
			// free everything
			cudaFree(dev_data);
			cudaFree(dev_data2);
			cudaFree(dev_mask);
			cudaFree(dev_f);
			cudaFree(dev_t);
		}
	}
}