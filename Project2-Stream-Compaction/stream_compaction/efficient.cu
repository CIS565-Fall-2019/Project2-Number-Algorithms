#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
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

		int *dev_arrayA;
		int *dev_arrayB;

		int *dev_bools;
		int *dev_boolScans;

		int *dev_idata;
		int *dev_odata;
		
		int * dev_indices;
			   		 
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


		__global__ void kernEffScanUpSweep(int N, int pow2d, int pow2d1, int* arrA) {
			int k = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (k >= N) return;

			if ((k % pow2d1) == 0 && (k + pow2d1 - 1)<N && (k + pow2d - 1)<N ){
				arrA[k + pow2d1 - 1] += arrA[k + pow2d - 1];
			}
		}

		__global__ void kernEffScanDownSweep(int N, int pow2d, int pow2d1, int* arrA) {
			int k = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (k >= N) return;
			
			int tmp = 0;
			
			if ((k % pow2d1) == 0 && (k + pow2d1 - 1) < N && (k + pow2d - 1) < N) {
				tmp = arrA[k + pow2d -1];
				arrA[k + pow2d - 1] = arrA[k + pow2d1 - 1];
				arrA[k + pow2d1 - 1] += tmp;
			}
		}

		__global__ void kernInitZero(int N, int* array) {
			
			int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
			
			if (tid < N) {
				array[tid] = 0;
			}
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			bool tmp = true;
			try {
				timer().startCpuTimer();
			}
			catch (const std::runtime_error& e) {
				tmp = false;
			}

            // TODO
			int n_new = n;

			//check for non-2powerN
			if (1 << ilog2ceil(n) != n)
				n_new = (1 << ilog2ceil(n));

			int fullBlocksPerGrid((n_new + blockSize - 1) / blockSize);

			cudaMalloc((void**)&dev_arrayA, n_new * sizeof(int));
			checkCUDAErrorFn("cudaMalloc dev_arrayA failed!");

			//Initialize to Zero
			kernInitZero <<<fullBlocksPerGrid, blockSize >>> (n_new, dev_arrayA);
			checkCUDAErrorFn("kernInitZero failed!");

			// Fill dev_arrayA with idata
			cudaMemcpy(dev_arrayA, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAErrorFn("cudaMemcpyToSymbol from idata to dev_arrayA failed!");

			// Upstream
			int pow2d1 = 0;
			int pow2d = 0;
			for (int d = 0; d <= ilog2ceil(n_new)-1; d++) {
				pow2d = 1 << (d);
				pow2d1 = 1 << (d+1);
				kernEffScanUpSweep << <fullBlocksPerGrid, blockSize >> > (n_new, pow2d, pow2d1, dev_arrayA);
				checkCUDAErrorFn("kernEffScanUpSweep failed!");		
			}

			// Downstream
			int *zero = new int[1];
			zero[0] = 0;
			cudaMemcpy(dev_arrayA + n_new-1, zero, 1*sizeof(int), cudaMemcpyHostToDevice);

			for (int d = ilog2ceil(n_new)-1; d >= 0; d--) {
				pow2d = 1 << (d);
				pow2d1 = 1 << (d + 1);
				kernEffScanDownSweep << <fullBlocksPerGrid, blockSize >> > (n_new, pow2d, pow2d1, dev_arrayA);
				checkCUDAErrorFn("kernGenerateRandomPosArray failed!");
			}

			// Copy back to cpu
			cudaMemcpy(odata, dev_arrayA, n*sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAErrorFn("cudaMemcpyFromSymbol from dev_arrayA to odata failed!");
			
			//printf("BBT Scan Final Computed : \n");
			//printArray(n, odata, true);

			if (tmp == true) timer().endCpuTimer();
			cudaFree(dev_arrayA);
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
			
			//Compute bools
			int fullBlocksPerGrid((n + blockSize - 1) / blockSize);

			cudaMalloc((void**)&dev_bools, n * sizeof(int));
			checkCUDAErrorFn("cudaMalloc dev_bools failed!");

			cudaMalloc((void**)&dev_idata, n*sizeof(int));
			checkCUDAErrorFn("cudaMalloc dev_arrayA failed!");

			cudaMemcpy(dev_idata, idata, n*sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAErrorFn("cudaMemcpyToSymbol from idata to dev_arrayA failed!");

			Common::kernMapToBoolean<<<fullBlocksPerGrid, blockSize >>>(n, dev_bools, dev_idata);
			checkCUDAErrorFn("kernMapToBoolean failed!");

			//compute scans
			int * indices = new int[n];
			int * bools = new int[n];

			cudaMemcpy(bools, dev_bools, n*sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAErrorFn("cudaMemcpyToSymbol from bools to dev_bools failed!");

			scan(n, indices, bools);

			cudaMalloc((void**)&dev_indices, n*sizeof(int));
			checkCUDAErrorFn("cudaMalloc dev_indices failed!");

			cudaMemcpy(dev_indices, indices, n*sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAErrorFn("cudaMemcpyToSymbol from indices to dev_indices failed!");

			cudaMalloc((void**)&dev_odata, n*sizeof(int));
			checkCUDAErrorFn("cudaMalloc dev_indices failed!");

			cudaMemcpy(dev_odata, odata, n*sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAErrorFn("cudaMemcpyToSymbol from indices to dev_indices failed!");

			//scatter
			Common::kernScatter<<<fullBlocksPerGrid, blockSize >>>(n, dev_odata, dev_idata, dev_bools, dev_indices);
			checkCUDAErrorFn("kernScatter failed!");

			// Copy back to cpu
			cudaMemcpy(odata, dev_odata, n*sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAErrorFn("cudaMemcpyFromSymbol from dev_odata to odata failed!");

			//printf("GPU Compaction : \n");
			//printArray(indices[n - 1], odata, true);

            timer().endGpuTimer();

			cudaFree(dev_bools);
			cudaFree(dev_idata);
			cudaFree(dev_indices);
			cudaFree(dev_odata);

			return indices[n-1];
        }
    }
}
