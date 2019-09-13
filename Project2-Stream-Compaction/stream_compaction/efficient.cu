#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "device_launch_parameters.h"
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

		__global__ void kernelUpSweepStep(int n, int d, int* cdata) {
			int k = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (k > n)
				return;
			int cur_step_size = 1 << (d + 1);
			int prev_step_size = 1 << d;
			if (k % cur_step_size == 0)
				cdata[k + cur_step_size - 1] += cdata[k + prev_step_size - 1];
		}

		__global__ void kernelDownSweepStep(int n, int d, int* cdata) {
			int k = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (k > n)
				return;
			int cur_step = 1 << (d + 1);
			int left_step = 1 << d;

			if (k % cur_step == 0) {
				int temp = cdata[k + left_step - 1];
				cdata[k + left_step - 1] = cdata[k + cur_step - 1];
				cdata[k + cur_step - 1] += temp;
			}
		}
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

		void printCudaArray(int size, int* cdata) {
			int *d_data = new int[size];
			cudaMemcpy(d_data, cdata, size * sizeof(int), cudaMemcpyDeviceToHost);
			printArray(size, d_data, true);
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
			int power_size = pow(2, ilog2ceil(n));
			int *cdata;
			cudaMalloc((void**)&cdata, power_size * sizeof(int));
			checkCUDAErrorFn("cudaMalloc adata failed!");

			cudaMemset(cdata, 0, power_size * sizeof(int));
			cudaMemcpy(cdata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			dim3 fullBlocksPerGrid((power_size + blockSize - 1) / blockSize);

			//Up Sweep
			for (int d = 0; d < ilog2ceil(power_size) ; d++) {
				kernelUpSweepStep <<<fullBlocksPerGrid, blockSize >>> (power_size, d, cdata);
			}

			//Down Sweep
			cudaMemset(cdata + power_size - 1, 0, sizeof(int));

			for (int d = ilog2(power_size) - 1; d >= 0; d--) {
				kernelDownSweepStep << <fullBlocksPerGrid, blockSize >> > (power_size, d, cdata);
			}
			cudaMemcpy(odata, cdata, sizeof(int) * power_size, cudaMemcpyDeviceToHost);
			cudaFree(cdata);
			timer().endGpuTimer();
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
