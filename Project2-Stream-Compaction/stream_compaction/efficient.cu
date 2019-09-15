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
		int *dev_odata;
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */

		void printArray(int n, int *a, bool abridged = false) {
			printf("    [ ");
			for (int i = 0; i < n; i++) {
				if (abridged && i + 2 == 15 && n > 16) {
					i = n - 2;
					printf("... ");
				}
				printf("%3d ", a[i]);
			}
			printf("]\n");
		}

		__global__ void up_sweep(int N, int *Dev_odata, int d) {

			int index = threadIdx.x + (blockIdx.x * blockDim.x);

			index = index * (1 << (d + 1));

			if (index > N-1) {
				return;
			}

			if (((index + (1 << (d)) - 1) < N) && ((index + (1 << (d + 1)) - 1) < N)) {

				Dev_odata[index + (1 << (d + 1)) - 1] += Dev_odata[index + (1 << (d)) - 1];
			}

			

		}


		__global__ void down_sweep(int N, int *Dev_odata, int d) {

			int index = threadIdx.x + (blockIdx.x * blockDim.x);


			index = index * (1 << (d + 1));


			if (index > N-1) {
				return;
			}


			if (((index + (1 << (d)) - 1) < N) && ((index + (1 << (d + 1)) - 1) < N)) {

				int t = Dev_odata[index + (1 << (d)) - 1];
				Dev_odata[index + (1 << (d)) - 1] = Dev_odata[index + (1 << (d + 1)) - 1];
				Dev_odata[index + (1 << (d + 1)) - 1] += t;
			}

			
		}
		/**
		 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
		 */
		void scan(int n, int *odata, const int *idata) {


			bool timer_started = false;
			try {
				timer().startGpuTimer();
			}
			catch (const std::exception& e) {
				timer_started = true;
			}

			int blockSize = 32;
			//printArray(n, idata);
			//int new_n = n;
			n = 1 << ilog2ceil(n); // make n something that is power of 2

			cudaMalloc((void**)&dev_odata, n * sizeof(int));
			cudaMemcpy(dev_odata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);



			for (int d = 0; d <= ((ilog2ceil(n)) - 1); d++) {
				int count_thread = 1 << ((ilog2ceil(n) - d - 1));   // i need ceil(n/d) threads total
				dim3 fullBlocksPerGrid(((count_thread)+blockSize -1)/ blockSize);
				up_sweep << <fullBlocksPerGrid, blockSize >> > (n, dev_odata, d);
			}

			cudaMemset(n + dev_odata - 1, 0, sizeof(int));

			for (int d = ilog2ceil(n) - 1; d >= 0; d--) {
				int count_thread = 1 << ((ilog2ceil(n) - d - 1));   // i need ceil(n/d) threads total
				dim3 fullBlocksPerGrid(((count_thread)+blockSize - 1) / blockSize);
				down_sweep << <fullBlocksPerGrid, blockSize >> > (n, dev_odata, d);
			}

			
			cudaMemcpy(odata, dev_odata, sizeof(int) * (n), cudaMemcpyDeviceToHost);
			//cudaMemcpy(dev_odata, dev_idata, sizeof(int) * n, cudaMemcpyDeviceToDevice);
			//odata[0] = 0;
			//printArray(n, odata);
			if (timer_started == false) {
				timer().endGpuTimer();
			}
			cudaFree(dev_odata);

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
            // TODO
            timer().endGpuTimer();
            return -1;
        }
    }
}
