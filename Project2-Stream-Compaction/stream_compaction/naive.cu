#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

		__global__ void kernInclusiveScanIteration(int n, int iteration, int *out, int *in) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= n) {
				return;
			}
			
			int nextIndex = ceil(pow(2.0, double(iteration - 1))); // encountered rounding issue at 2048, should fix it
			if (index >= nextIndex) {
				out[index] = in[index - nextIndex] + in[index];
			}
		}

		__global__ void kernShiftRight(int n, int *out, int *in) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= n) {
				return;
			}

			out[index] = index == 0 ? 0 : in[index - 1];
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			int *dev_in;
			int *dev_out;

			// malloc device buffers
			cudaMalloc((void**)&dev_in, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_in failed!");
			cudaMalloc((void**)&dev_out, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_out failed!");

			// copy input to device
			cudaMemcpy(dev_in, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy dev_in idata failed!");
			cudaMemcpy(dev_out, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy dev_out idata failed!");

			// perform inclusive scan
			timer().startGpuTimer();

			dim3 gridSize = dim3((n + blockSize - 1) / blockSize, 1, 1);
			for (int d = 1; d <= ilog2ceil(n); d++) {
				// run one iteration
				kernInclusiveScanIteration<<<gridSize, blockSize>>>(n, d, dev_out, dev_in);
				checkCUDAError("kernInclusiveScanIteration failed!");

				// copy out to in
				cudaMemcpy(dev_in, dev_out, n * sizeof(int), cudaMemcpyDeviceToDevice);
				checkCUDAError("cudaMemcpy dev_in dev_out failed!");
			}

			// convert to exclusive scan
			kernShiftRight<<<gridSize, blockSize>>>(n, dev_out, dev_in);
			checkCUDAError("kernShiftRight failed!");

			timer().endGpuTimer();

			// copy output to host
			cudaMemcpy(odata, dev_out, n * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy odata dev_out failed!");

			cudaFree(dev_in);
			cudaFree(dev_out);
			checkCUDAError("cudaFree failed!");
        }
    }
}
