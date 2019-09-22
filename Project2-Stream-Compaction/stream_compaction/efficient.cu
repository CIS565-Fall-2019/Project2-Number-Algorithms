#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

/*! Block size used for CUDA kernel launch*/
#define blockSize 512
int *dev_idata;
int *dev_odata;
int *dev_mask;
namespace StreamCompaction {
	namespace Efficient {
		using StreamCompaction::Common::PerformanceTimer;
		PerformanceTimer& timer()
		{
			static PerformanceTimer timer;
			return timer;
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

		__global__ void kernUpsweep(int n, int d, int *dev_odata, int incr, int twod) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			//also return if index is not a multiple of the incr
			if (index >= n || (index) % incr != 0) {
				return;
			}
			//if we reached here, index+1 must be a multiple of incr (2^(d+1))
			dev_odata[index + incr - 1] += dev_odata[index + twod - 1];
			dev_odata[n - 1] = 0;
		}
		__global__ void kernDownsweep(int n, int d, int *dev_odata, int incr, int twod) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			//also return if index is not a multiple of the incr
			if (index >= n || (index) % incr != 0) {
				return;
			}
			//if we reached here, index+1 must be a multiple of incr (2^(d+1))
			int t = dev_odata[index + twod - 1];
			dev_odata[index + twod - 1] = dev_odata[index + incr - 1];
			dev_odata[index + incr - 1] += t;
		}

		__global__ void kernMapToBoolean(int n, int *mask, int *idata) {
			//dev_odata contains idata
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= n) {
				return;
			}
			if (idata[index] != 0) {
				mask[index] = 1;
			}
			else {
				mask[index] = 0;
			}
		}

		__global__ void kernScatter(int n, int *mask, int *odata, int *idata) {
			//odata now contains scan result
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= n){
				return;
			}
			int shouldInclude = mask[index];
			if (shouldInclude) {
				int newIdx = odata[index];
				odata[newIdx] = idata[index];
			}
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			int malloc_size = nextPowerOf2(n);
			//CUDA Malloc buffers
			cudaMalloc((void**)&dev_odata, malloc_size * sizeof(int));
			checkCUDAError("cudaMalloc dev_odata failed!");

			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
			int max_level = ilog2ceil(n);
			int incr = 0;
			int twod = 0;

            timer().startGpuTimer();
			//Copy idata into dev_odata
			cudaMemcpy(dev_odata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy dev_odata failed!");

			//Upsweep
			for (int d = 0; d < max_level; d++) {
				incr = pow(2, d + 1);
				twod = pow(2, d);
				kernUpsweep<<<fullBlocksPerGrid, blockSize >>>(malloc_size, d, dev_odata, incr, twod);
			}

			//Downsweep
			for (int d = max_level-1; d >= 0; d--) {
				incr = pow(2, d + 1);
				twod = pow(2, d);
				kernDownsweep<<<fullBlocksPerGrid, blockSize >>>(malloc_size, d, dev_odata, incr, twod);
			}
			cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            timer().endGpuTimer();

			//Free Memory
			cudaFree(dev_odata);
        }

        void scan_notimer(int n, int malloc_size) {
			dim3 fullBlocksPerGrid((malloc_size + blockSize - 1) / blockSize);
			int max_level = ilog2ceil(n);
			int incr = 0;
			int twod = 0;

			//Upsweep
			for (int d = 0; d < max_level; d++) {
				incr = pow(2, d + 1);
				twod = pow(2, d);
				kernUpsweep<<<fullBlocksPerGrid, blockSize >>>(malloc_size, d, dev_odata, incr, twod);
			}

			//Downsweep
			for (int d = max_level-1; d >= 0; d--) {
				incr = pow(2, d + 1);
				twod = pow(2, d);
				kernDownsweep<<<fullBlocksPerGrid, blockSize >>>(malloc_size, d, dev_odata, incr, twod);
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
			int malloc_size = nextPowerOf2(n);
			//CUDA Malloc buffers
			cudaMalloc((void**)&dev_odata, malloc_size * sizeof(int));
			checkCUDAError("cudaMalloc dev_odata failed!");
			cudaMalloc((void**)&dev_odata, malloc_size * sizeof(int));
			checkCUDAError("cudaMalloc dev_odata failed!");
			cudaMalloc((void**)&dev_mask, malloc_size * sizeof(int));
			checkCUDAError("cudaMalloc  failed!");

			//Memcpy idata into dev_odata for starters
			cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy dev_idata failed!");
			cudaMemcpy(dev_odata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy dev_odata failed!");

            timer().startGpuTimer();
			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
			//1: Compute mask (Temporary Array)
			kernMapToBoolean<<<fullBlocksPerGrid, blockSize>>>(n, dev_mask, dev_odata);

			//2: Exclusive Scan on TempArray
			scan_notimer(n, malloc_size);

			//2.5: Get Count from dev_odata
			int tempcount[1];
			cudaMemcpy(&tempcount, dev_odata + n - 1, 1 * sizeof(int), cudaMemcpyDeviceToHost);
			int count = idata[n - 1] == 0 ? tempcount[0] : tempcount[0] + 1;

			//3: Scatter
			kernScatter<<<fullBlocksPerGrid, blockSize>>>(n, dev_mask, dev_odata, dev_idata);

            timer().endGpuTimer();

			free(dev_mask);
			free(dev_odata);
            return count;
        }
    }
}
