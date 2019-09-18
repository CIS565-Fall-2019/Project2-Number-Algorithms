#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "radix.h"
#include "efficient.h"

#define blockSize 128


namespace StreamCompaction {
    namespace Radix {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__

		__global__ void kernComputeE(int n, int bitPos, int *input, int *e) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;

			if (index < n) {
				e[index] = (((input[index] >> bitPos) & 1) == 0) ? 1 : 0;
			}

		}

		__global__ void kernComputeTotalFalses(int n, int* totalFalses, int *e, int *f) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;

			if (index < n) {
				*totalFalses = e[n - 1] + f[n - 1];
			}

		}

		__global__ void kernComputeD(int n, int *e, int *t, int *f, int *d) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;

			if (index < n) {
				d[index] = e[index] ? f[index] : t[index];
			}

		}

		__global__ void kernComputeT(int n, int *totalFalses, int *t, int *f) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;

			if (index < n) {
				t[index] = index - f[index] + (*totalFalses);
			}

		}



		/**
		 * Performs scatter on an array. That is, for each element in idata,
		 */
		__global__ void kernScatter(int n, int *odata,
			const int *idata, const int *indices) {
			// TODO
			int index = threadIdx.x + (blockDim.x * blockIdx.x);
			if (index >= n) {
				return;
			}
			odata[indices[index]] = idata[index];

		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void radix(int n, int *odata, const int *idata) {

			int *devRadixData;
			cudaMalloc((void **)&devRadixData, n * sizeof(int));
			checkCUDAError("cudaMalloc");
			cudaMemcpy(devRadixData, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy");

			int *e;
			cudaMalloc((void **)&e, n * sizeof(int));
			checkCUDAError("cudaMalloc");

			int *f;
			int newSize = 1 << ilog2ceil(n);
			cudaMalloc((void **)&f, newSize * sizeof(int));
			checkCUDAError("cudaMalloc");


			int *t;
			cudaMalloc((void **)&t, n * sizeof(int));
			checkCUDAError("cudaMalloc");

			int *d;
			cudaMalloc((void **)&d, n * sizeof(int));
			checkCUDAError("cudaMalloc");

			int *output;
			cudaMalloc((void **)&output, n * sizeof(int));
			checkCUDAError("cudaMalloc");

			int *totalFalses;
			cudaMalloc((void **)&totalFalses, 1 * sizeof(int));
			checkCUDAError("cudaMalloc");

			int gridRows = (n + blockSize - 1) / blockSize;

			timer().startGpuTimer();

			for (int bitPos = 0; bitPos < 6; bitPos++) {

				kernComputeE<<<gridRows, blockSize>>>(n, bitPos, devRadixData, e);
				StreamCompaction::Efficient::scanForRadix(n, f, e, blockSize);
				kernComputeTotalFalses <<< gridRows, blockSize >> > (n, totalFalses, e, f);
				kernComputeT << < gridRows, blockSize >> > (n, totalFalses, t, f);
				kernComputeD << < gridRows, blockSize >> > (n, e, t, f, d);
				kernScatter << < gridRows, blockSize >> > (n, output, devRadixData, d);
				cudaMemcpy(devRadixData, output, n * sizeof(int), cudaMemcpyDeviceToDevice);
			}
			timer().endGpuTimer();
			cudaMemcpy(odata, devRadixData, n * sizeof(int), cudaMemcpyDeviceToHost);
        }
    }
}
