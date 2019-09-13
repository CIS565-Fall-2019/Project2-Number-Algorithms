#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <bitset>
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

		int nextPowerOfTwo(int in) {
			int out = 0;
			float log = log2(in);

			// If this is true, the number IS a power of 2
			if (ceil(log) == floor(log)) {
				out = in;
			}
			else {
				// Not a power of two, grab the next one up.
				out = 1;
				do {
					out = out << 1;
				} while (out < in);
			}

			return out;
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			// Efficient algorithm uses balanded binary trees and two phases: upsweep and downsweep.
			// This can be performed inplace.

			// 0) Correct length to be Power of 2
			const int N = nextPowerOfTwo(n); // Returns 'n' if input is already a power of 2.

			// 1) Initialize Memory
			int* dev_data = 0;
			cudaMalloc(&dev_data, N * sizeof(int));
			cudaMemset(dev_data + n, 0, (N - n) * sizeof(int));
			checkCUDAError("CUDA memset 1 failed");
			cudaMemcpy(dev_data, idata, n * sizeof(int), ::cudaMemcpyHostToDevice);

			// 2) Upsweep
			timer().startGpuTimer();
			int* INSPECT = (int*)malloc(N * sizeof(int));
			for (int d = 0; d <= ilog2ceil(N) - 1; d++) {
				const int NUM_THREADS = N;
				const int NUM_BLOCKS = 1;

				// Kernel Call
				cudaMemcpy(INSPECT, dev_data, N * sizeof(int), ::cudaMemcpyDeviceToHost);
				kernWorkEffScanUpsweep<<<NUM_BLOCKS, NUM_THREADS>>>(n, d, dev_data, dev_data);
				cudaMemcpy(INSPECT, dev_data, N * sizeof(int), ::cudaMemcpyDeviceToHost);
				cudaDeviceSynchronize();
			}

			// 3) Downsweep
			cudaMemset(dev_data + (N-1), 0, 1*sizeof(int));
			checkCUDAError("CuDA memset 2 failed");
			for (int d = ilog2ceil(N) - 1; d >= 0; d--) {
				const int NUM_THREADS = N;
				const int NUM_BLOCKS = 1;

				// Kernel Call
				cudaMemcpy(INSPECT, dev_data, N * sizeof(int), ::cudaMemcpyDeviceToHost);
				kernWorkEffScanDownsweep << <NUM_BLOCKS, NUM_THREADS >> > (n, d, dev_data, dev_data);
				cudaMemcpy(INSPECT, dev_data, N * sizeof(int), ::cudaMemcpyDeviceToHost);
				cudaDeviceSynchronize();
			}
			
			// 4) Cleanup
			timer().endGpuTimer();
			cudaMemcpy(odata, dev_data, N * sizeof(int), ::cudaMemcpyDeviceToHost);
			cudaFree(dev_data);

			return;
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

		__global__ void kernWorkEffScanUpsweep(const int N, const int D, int *out, const int* in) {
			int k = threadIdx.x + (blockIdx.x * blockDim.x);
			if (k >= N) {
				return;
			}

			if (k % (int)powf(2, D + 1) == 0) {
				out[k + (int)powf(2, D + 1) - 1] += in[k + (int)powf(2, D) - 1];
			}
		}

		__global__ void kernWorkEffScanDownsweep(const int N, const int D, int *out, const int* in) {
			int k = threadIdx.x + (blockIdx.x * blockDim.x);
			if (k >= N) {
				return;
			}

			if (k % (int)powf(2, D + 1) == 0) {
				int tmp = in[k + (int)powf(2, D) - 1];
				out[k + (int)powf(2, D) - 1] = out[k + (int)powf(2, D + 1) - 1];
				out[k + (int)powf(2, D + 1) - 1] += tmp;
			}
		}
    }
}
