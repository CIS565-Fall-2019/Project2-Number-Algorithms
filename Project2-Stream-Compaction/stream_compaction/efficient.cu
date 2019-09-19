#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

		__global__ void kernScanUpsweep(int n, int iteration, int *buffer) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= n) {
				return;
			}

			int power1 = ceil(pow(2.0, double(iteration + 1)));
			if (index % power1 == 0) {
				int power2 = ceil(pow(2.0, double(iteration)));
				buffer[index + power1 - 1] += buffer[index + power2 - 1];
			}
		}

		__global__ void kernScanDownsweep(int n, int iteration, int *buffer) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= n) {
				return;
			}

			int power1 = ceil(pow(2.0, double(iteration + 1)));
			if (index % power1 == 0) {
				int power2 = ceil(pow(2.0, double(iteration)));

				int temp = buffer[index + power2 - 1];
				buffer[index + power2 - 1] = buffer[index + power1 - 1];
				buffer[index + power1 - 1] += temp;
			}
		}

		// finds the next power of 2 greater than or equal to n
		int nextPowerOfTwo(int n) {
			if (n && !(n & (n - 1)))
				return n;

			int count = 0;
			while (n != 0) {
				n >>= 1;
				count++;
			}

			return 1 << count;
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			int nPowerOfTwo = nextPowerOfTwo(n);

			int *dev_buffer;

			// malloc device buffer
			cudaMalloc((void**)&dev_buffer, nPowerOfTwo * sizeof(int));
			checkCUDAError("cudaMalloc dev_buffer failed!");

			// copy input to device buffer
			cudaMemcpy(dev_buffer, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy dev_buffer idata failed!");

			// fill rest of device buffer with zero
			cudaMemset(dev_buffer + n, 0, (nPowerOfTwo - n) * sizeof(int));
			checkCUDAError("cudaMemset dev_buffer failed!");

			timer().startGpuTimer();

			// upsweep
			dim3 gridSize = dim3((nPowerOfTwo + blockSize - 1) / blockSize, 1, 1);
			for (int d = 0; d < ilog2ceil(nPowerOfTwo); d++) {
				kernScanUpsweep<<<gridSize, blockSize>>>(nPowerOfTwo, d, dev_buffer);
				checkCUDAError("kernScanUpsweep failed!");
			}

			cudaDeviceSynchronize();

			// set root to zero
			cudaMemset(dev_buffer + nPowerOfTwo - 1, 0, 1 * sizeof(int));
			checkCUDAError("cudaMemset dev_buffer failed!");

			// downsweep
			for (int d = ilog2ceil(nPowerOfTwo) - 1; d >= 0; d--) {
				kernScanDownsweep<<<gridSize, blockSize>>>(nPowerOfTwo, d, dev_buffer);
				checkCUDAError("kernScanDownsweep failed!");
			}

			cudaDeviceSynchronize();
			timer().endGpuTimer();

			// copy output to host
			cudaMemcpy(odata, dev_buffer, n * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy odata dev_buffer failed!");

			cudaFree(dev_buffer);
			checkCUDAError("cudaFree failed!");
        }

		/**
		 * Performs prefix-sum (aka scan) on idata, storing the result into odata, NO TIMER
		 */
		void scanNoTimer(int n, int *odata, const int *idata) {
			int nPowerOfTwo = nextPowerOfTwo(n);

			int *dev_buffer;

			// malloc device buffer
			cudaMalloc((void**)&dev_buffer, nPowerOfTwo * sizeof(int));
			checkCUDAError("cudaMalloc dev_buffer failed!");

			// copy input to device buffer
			cudaMemcpy(dev_buffer, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy dev_buffer idata failed!");

			// fill rest of device buffer with zero
			cudaMemset(dev_buffer + n, 0, (nPowerOfTwo - n) * sizeof(int));
			checkCUDAError("cudaMemset dev_buffer failed!");

			// upsweep
			dim3 gridSize = dim3((n + blockSize - 1) / blockSize, 1, 1);
			for (int d = 0; d < ilog2ceil(nPowerOfTwo); d++) {
				kernScanUpsweep<<<gridSize, blockSize>>>(nPowerOfTwo, d, dev_buffer);
				checkCUDAError("kernScanUpsweep failed!");
			}

			cudaDeviceSynchronize();

			// set root to zero
			cudaMemset(dev_buffer + nPowerOfTwo - 1, 0, 1 * sizeof(int));
			checkCUDAError("cudaMemset dev_buffer failed!");

			// downsweep
			for (int d = ilog2ceil(nPowerOfTwo) - 1; d >= 0; d--) {
				kernScanDownsweep<<<gridSize, blockSize>>>(nPowerOfTwo, d, dev_buffer);
				checkCUDAError("kernScanDownsweep failed!");
			}

			cudaDeviceSynchronize();

			// copy output to host
			cudaMemcpy(odata, dev_buffer, n * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy odata dev_buffer failed!");

			cudaFree(dev_buffer);
			checkCUDAError("cudaFree failed!");
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

			int *dev_in;
			int *dev_out;
			int *dev_bools;
			int *dev_indices;
            
			// malloc device buffers
			cudaMalloc((void**)&dev_in, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_input failed!");

			cudaMalloc((void**)&dev_out, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_out failed!");

			cudaMalloc((void**)&dev_bools, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_bools failed!");

			cudaMalloc((void**)&dev_indices, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_indices failed!");

			// copy input to device buffer
			cudaMemcpy(dev_in, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy dev_in idata failed!");

			timer().startGpuTimer();

			// compute bools buffer
			Common::kernMapToBoolean<<<fullBlocksPerGrid , blockSize>>>(n, dev_bools, dev_in);
			checkCUDAError("kernMapToBoolean failed!");

			cudaDeviceSynchronize();

			// copy bools to host
			int *bools = new int[n];
			cudaMemcpy(bools, dev_bools, n * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy bools dev_bools failed!");

			// run exclusive scan on bools
			int *indices = new int[n];
			scanNoTimer(n, indices, bools);
			int outputSize = bools[n - 1] == 0 ? indices[n - 1] : indices[n - 1] + 1;

			// copy indices to device
			cudaMemcpy(dev_indices, indices, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy dev_indices indices failed!");

			// scatter
			Common::kernScatter<<<fullBlocksPerGrid, blockSize>>>(n, dev_out, dev_in, dev_bools, dev_indices);
			cudaDeviceSynchronize();

			timer().endGpuTimer();

			// copy output to host
			cudaMemcpy(odata, dev_out, outputSize * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy odata dev_out failed!");

			cudaFree(dev_in);
			cudaFree(dev_out);
			cudaFree(dev_bools);
			cudaFree(dev_indices);
			checkCUDAError("cudaFree failed!");
			
            return outputSize;
        }
    }
}
