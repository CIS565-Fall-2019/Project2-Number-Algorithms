#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include "device_launch_parameters.h"


#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

#define blockSize 128

int* dev_idata;
int* padded_idata;

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
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

		__global__ void upSweep(int n, int d, int* A) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) {
				return;
			}
			int stride = 1 << (d+1);
			int other_index = 1 << d;
			if ((index) % stride == 0) {
				A[index + stride - 1] += A[index + other_index - 1];
			}
		}

		__global__ void downSweep(int n, int d, int* A) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) {
				return;
			}
			int left_index = 1 << (d);
			int right_index = 1 << (d + 1);
			if (index % right_index == 0) {
				int temp = A[index + left_index - 1];
				A[index + left_index - 1] = A[index + right_index - 1];
				A[index + right_index - 1] += temp;
			}
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
			int padded_size = 1 << (ilog2ceil(n));

			
			cudaMalloc((void**)&padded_idata, padded_size * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc padded_idata failed!");

			cudaMemset(padded_idata, 0, padded_size * sizeof(int));
			cudaMemcpy(padded_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

			dim3 fullBlocksPerGrid((padded_size + blockSize - 1) / blockSize);
			int iterations = ilog2(padded_size);

			//Up-Sweep
			for (int d = 0; d < iterations; d++) {
				upSweep << <fullBlocksPerGrid, blockSize >> > (padded_size, d, padded_idata);
			}
			
			//Down-Sweep
			cudaMemset(padded_idata + (padded_size - 1), 0, sizeof(int));
			
			for (int d = iterations - 1; d >= 0; d--) {
				downSweep << <fullBlocksPerGrid, blockSize >> > (padded_size, d, padded_idata);
			}
			cudaMemcpy(odata, padded_idata, sizeof(int) * n, cudaMemcpyDeviceToHost);

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

			cudaMalloc((void**)&dev_idata, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_idata failed!");

			cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

			int* dev_bools;
			cudaMalloc((void**)&dev_bools, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_bools failed!");
			
			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
			
			StreamCompaction::Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> > (n, dev_bools, dev_idata);

			int *bools;
			cudaMemcpy(bools, dev_bools, sizeof(int) * n, cudaMemcpyDeviceToHost);

			int* indices;
			
			scan(n, indices, bools);

			int *dev_indices;
			cudaMalloc((void**)&dev_bools, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_bools failed!");
			cudaMemcpy(indices, dev_indices, sizeof(int) * n, cudaMemcpyHostToDevice);

			int *dev_odata;
			cudaMalloc((void**)&dev_odata, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_odata failed!");

			StreamCompaction::Common::kernScatter << <fullBlocksPerGrid, blockSize >> > (n, dev_odata, dev_idata, dev_bools, dev_indices);

            timer().endGpuTimer();
            return -1;
        }
    }
}
