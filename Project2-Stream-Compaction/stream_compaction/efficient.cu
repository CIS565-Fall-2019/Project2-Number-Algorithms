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

		__global__ 
		void kernReduction(int n, unsigned int d, int *idata)
		{
			int k = blockIdx.x * blockDim.x + threadIdx.x;
			unsigned int offset = 1 << d;

			if (k >= n || k % (offset << 1) !=0 ) return;

			idata[k + (offset << 1) - 1] += idata[k + offset - 1];
		}

		__global__
			void kernDownSweep(int n, unsigned int d, int* idata)
		{
			int k = blockIdx.x * blockDim.x + threadIdx.x;

			unsigned int offset = 1 << d;
			if (k >= n || k % (offset << 1) != 0) return;

			int tmp = idata[k + offset - 1];	// Save left child
			idata[k + offset - 1] = idata[k + (offset << 1) - 1];	// Set left child to this node's value
			idata[k + (offset << 1) - 1] += tmp;	// Set right child to old left value + this node's value
		}

		__global__ void kernSetZero(int n, int* idata)
		{
			idata[n - 1] = 0;
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
			int* d_idata;
			int maxD = ilog2ceil(n);
			int nCeil = 1 << maxD;

			cudaMalloc(&d_idata, nCeil * sizeof(int));
			cudaMemset(d_idata, 0, nCeil * sizeof(int));
			cudaMemcpy(d_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

			int threadsPerBlock = 512;
			int blockSize = (nCeil + threadsPerBlock - 1) / threadsPerBlock;

			// Parallel Reduction
			for (int d = 0; d < maxD; d++) {
				kernReduction << <blockSize, threadsPerBlock >> > (n, d, d_idata);
				cudaDeviceSynchronize();
			}

			kernSetZero << <1, 1 >> > (nCeil, d_idata);
			cudaDeviceSynchronize();
			// Down Sweep
			for (int d = maxD-1; d >= 0; d--) {
				kernDownSweep << <blockSize, threadsPerBlock >> > (n, d, d_idata);
				cudaDeviceSynchronize();
			}

			cudaMemcpy(odata, d_idata, n * sizeof(int), cudaMemcpyDeviceToHost);
			cudaFree(d_idata);

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
