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
		__global__ void kernNaiveScan(int n, int offset, int *odata, const int *idata) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= n) {
				return;
			}

		/*	if (index < offset) {
				odata[index] = idata[index];
			}
			else {
				odata[index] = idata[index - offset] + idata[index];
			}*/

			odata[index] = (index < offset) ? idata[index] : idata[index - offset] + idata[index];
		}


		__global__ void kernMakeExclusive(int n, int *odata, const int *idata) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= n) {
				return;
			}
			odata[index] = (index == 0) ? 0 : idata[index - 1];
		}

	    void printDeviceArr(int n, int* device_arr) {
		    int* arr = (int*)malloc(sizeof(int)*n);
			cudaMemcpy(arr, device_arr, sizeof(int) *n, cudaMemcpyDeviceToHost);
			printf("\n [");
			for (int i = -1; i < n; i++) {
				printf("%d, ", arr[i]);
			}
			printf("]\n");
			free(arr);
		}
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			dim3 threadsPerBlock(blockSize);
			dim3 blocksPerGrid((n + blockSize - 1) / blockSize);

			int* A;
			int* B;

			cudaMalloc((void**)&A, sizeof(int) * n);
			checkCUDAError("cuda Error Allocating A");
			cudaMalloc((void**)&B, sizeof(int) * n);
			checkCUDAError("cuda Error Allocating B");
			cudaMemcpy(A, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("Error Copying data to A");

            timer().startGpuTimer();
			for (int offset = 1; offset < n; offset *= 2) {
				kernNaiveScan<<<blocksPerGrid,threadsPerBlock>>>(n, offset, B, A);
                
				cudaMemcpy(A, B, n * sizeof(int), cudaMemcpyDeviceToDevice);
			    checkCUDAError("Error Copying data to A");
				
			}
			kernMakeExclusive << <blocksPerGrid, threadsPerBlock >> > (n, B, A);
			    checkCUDAError("ERror in kernMakeExclusive");
		    cudaMemcpy(odata, B, n * sizeof(int), cudaMemcpyDeviceToHost);
			    checkCUDAError("ERror in Copying back to host");
             // TODO
            timer().endGpuTimer();

			cudaFree(A);
			cudaFree(B);
        }
    }
}
