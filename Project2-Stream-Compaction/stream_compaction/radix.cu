#include <cuda.h>
#include <cuda_runtime.h>
#include "radix.h"
#include "stream_compaction/efficient.h"
#include "common.h"


namespace Sorting {
	namespace Radix {
		#define block_size 128
		__global__ void mask_generation(unsigned long long int n, long long *dev_mask, long long *dev_idata, int bitmask) {
			unsigned long long int index = blockDim.x * blockIdx.x + threadIdx.x;
			if (index >= n)
				return;
			dev_mask[index] = !((bool)(dev_idata[index] & bitmask));
		}
		__global__ void true_index_generation(unsigned long long int n, long long *dev_t, long long *dev_f, long long total_falses) {
			unsigned long long int index = blockDim.x * blockIdx.x + threadIdx.x;
			if (index >= n)
				return;
			dev_t[index] = index - dev_f[index] + total_falses;
		}
		__global__ void reshuffle_mask(unsigned long long int n, long long *dev_odata, long long *dev_idata, long long *dev_t, long long *dev_f, int bitmask) {
			unsigned long long int index = blockDim.x * blockIdx.x + threadIdx.x;
			if (index >= n)
				return;
			int data = dev_idata[index];
			if ((bool)(data & bitmask))
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
		void sort(unsigned long long int n, long long *odata, long long *idata, int max_value) {
			timer().startGpuTimer();
			int loop_count = countBits(max_value);//std::numeric_limits<int>::digits;
			int mask = 1;
			long long int total_falses, tmp;
			unsigned long long int blocks = ceil((n + block_size - 1) / block_size);
			long long *dev_data, *dev_f, *dev_t, *dev_data2;
			long long *dev_mask;
			cudaMalloc((void**)&dev_data, n*sizeof(long long));
			cudaMalloc((void**)&dev_data2, n * sizeof(long long));
			cudaMalloc((void**)&dev_mask, n * sizeof(long long));
			cudaMalloc((void**)&dev_f, n * sizeof(long long));
			cudaMalloc((void**)&dev_t, n * sizeof(long long));
			// copy over idata
			cudaMemcpy(dev_data, idata, n * sizeof(long long), cudaMemcpyHostToDevice);
			for (int i = 0; i < loop_count; i++, mask <<= 1) {
				mask_generation <<<blocks, block_size >>> (n, dev_mask, dev_data, mask);
				// scan to get false position indices
				StreamCompaction::SharedMemory::dev_scan(n, dev_f, dev_mask);
				cudaMemcpy(&total_falses, dev_f + n - 1, sizeof(long long), cudaMemcpyDeviceToHost);
				cudaMemcpy(&tmp, dev_mask + n - 1, sizeof(long long), cudaMemcpyDeviceToHost);
				total_falses += tmp;
				// compute true position indices
				true_index_generation <<<blocks, block_size >>> (n, dev_t, dev_f, total_falses);
				// reshuffle data using correct positions
				reshuffle_mask << <blocks, block_size >> > (n, dev_data2, dev_data, dev_t, dev_f, mask);
				// copy data2 over to data
				cudaMemcpy(dev_data, dev_data2, n * sizeof(long long), cudaMemcpyDeviceToDevice);
			}
			// we are done sorting, copy back
			cudaMemcpy(odata, dev_data, n * sizeof(long long), cudaMemcpyDeviceToHost);
			// free everything
			cudaFree(dev_data);
			cudaFree(dev_data2);
			cudaFree(dev_mask);
			cudaFree(dev_f);
			cudaFree(dev_t);
			timer().endGpuTimer();
		}
		using StreamCompaction::Common::PerformanceTimer;
		PerformanceTimer& timer()
		{
			static PerformanceTimer timer;
			return timer;
		}
	}
}