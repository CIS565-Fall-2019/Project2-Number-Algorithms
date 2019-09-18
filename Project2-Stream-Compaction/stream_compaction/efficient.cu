#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#define blockSize 512
namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

		__global__ void kernUpsweep(int n, int d, int* idata) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= n) {
				return;
			}
			int offset  =  (1 << (d + 1)); // implementing 2^d+1 incrementing 
			int k = index * offset;
			if (k >= n) {
				return;
			}

			idata[k + offset - 1] += idata[k + (1 << d) - 1];
		}

		__global__ void kernDownsweep(int n, int d, int* idata) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= n) {
				return;
			}
			int offset = (1 << (d + 1)); // implementing 2^d+1 incrementing 
			int k = index * offset;
			if (k >= n) {
				return;
			}

			int t = idata[k + (1 << d) - 1];
			idata[k + (1 << d) - 1] = idata[k + offset - 1];
			idata[k + offset - 1] += t;
		}


		void printDeviceArr(int n, int* device_arr) {
			int* arr = (int*)malloc(sizeof(int)*n);
			cudaMemcpy(arr, device_arr, sizeof(int) *n, cudaMemcpyDeviceToHost);
			printf("\n [");
			for (int i = 0; i < n; i++) {
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
			int npt = 1 << ilog2ceil(n);

			int * dev_idata;
			cudaMalloc((void **)&dev_idata, npt * sizeof(int));
			checkCUDAError("Error: Cuda Malloc for input data on device");
			if (npt != n) {
				int *new_cpu_arr = (int*) malloc(sizeof(int) * npt);
				memset(new_cpu_arr, 0, sizeof(int) * npt);
				memcpy(new_cpu_arr, idata, sizeof(int) * n);
			    cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
				checkCUDAError("Cuda Error on non-power of two array");
			    free(new_cpu_arr);
			} else { 
			    cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
				checkCUDAError("Cuda Error on power of two array");
			}

            timer().startGpuTimer();
			for (int d = 0; d < ilog2ceil(npt); d++) {
				int updateThreadUtil = npt / (1 << (npt + 1));
				dim3 blocksPerGrid((updateThreadUtil + blockSize - 1) / blockSize);
				kernUpsweep << <blocksPerGrid, threadsPerBlock >> > (npt, d, dev_idata);
			}

			int z = 0;
			cudaMemcpy(&dev_idata[npt - 1], &z, sizeof(int), cudaMemcpyHostToDevice);
			for (int d = ilog2(npt) - 1; d >= 0; d--) {
				int updateThreadUtil = npt / (1 << (npt + 1));
				dim3 blocksPerGrid((updateThreadUtil + blockSize - 1) / blockSize);
				kernDownsweep << <blocksPerGrid, threadsPerBlock >> > (npt, d, dev_idata);
			}
            // TODO
            timer().endGpuTimer();
			cudaMemcpy(odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToHost);
			cudaFree(dev_idata);
        }

        void compactScan(int n, int *odata, const int *idata) {

			dim3 threadsPerBlock(blockSize);
			int npt = 1 << ilog2ceil(n);

			int * dev_idata;
			cudaMalloc((void **)&dev_idata, npt * sizeof(int));
			checkCUDAError("Error: Cuda Malloc for input data on device");
			if (npt != n) {
				int *new_cpu_arr = (int*) malloc(sizeof(int) * npt);
				memset(new_cpu_arr, 0, sizeof(int) * npt);
				memcpy(new_cpu_arr, idata, sizeof(int) * n);
			    cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
				checkCUDAError("Cuda Error on non-power of two array");
			} else { 
			    cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
				checkCUDAError("Cuda Error on power of two array");
			}

			for (int d = 0; d < ilog2ceil(npt); d++) {
				int updateThreadUtil = npt / (1 << (npt + 1));
				dim3 blocksPerGrid((updateThreadUtil + blockSize - 1) / blockSize);
				kernUpsweep << <blocksPerGrid, threadsPerBlock >> > (npt, d, dev_idata);
			}

			int z = 0;
			cudaMemcpy(&dev_idata[npt - 1], &z, sizeof(int), cudaMemcpyHostToDevice);
			for (int d = ilog2(npt) - 1; d >= 0; d--) {
				int updateThreadUtil = npt / (1 << (npt + 1));
				dim3 blocksPerGrid((updateThreadUtil + blockSize - 1) / blockSize);
				kernDownsweep << <blocksPerGrid, threadsPerBlock >> > (npt, d, dev_idata);
			}
            // TODO
			cudaMemcpy(odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToHost);
			cudaFree(dev_idata);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.

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
			dim3 threadsPerBlock(blockSize);
			dim3 blocksPerGrid((n + blockSize - 1) / blockSize);

			int *dev_idata, *dev_odata, *dev_bool;
			cudaMalloc((void**)&dev_idata, sizeof(int) * n);
			checkCUDAError("Malloc Error dev_idata");
			cudaMalloc((void**)&dev_odata, sizeof(int) * n);
			checkCUDAError("Malloc Error dev_odata");
			cudaMalloc((void**)&dev_bool,  sizeof(int) * n);
			checkCUDAError("Malloc Error dev_bool");
			
			cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
			checkCUDAError("memcpy error dev_idata", __LINE__);


            timer().startGpuTimer();
			Common::kernMapToBoolean << <blocksPerGrid, threadsPerBlock >> > (n, dev_bool, dev_idata);
			checkCUDAError("map to boolean error", __LINE__);
			int* host_bool = new int[n];
			cudaMemcpy(host_bool, dev_bool, n * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("memcpy erroyr host_bool", __LINE__);
			
			int* indices = new int[n];

			compactScan(n, indices, host_bool);
			int* dev_indices;
			cudaMalloc((void**)&dev_indices, sizeof(int) * n);
			checkCUDAError("malloc error dev_indices", __LINE__);
			cudaMemcpy(dev_indices, indices, sizeof(int)*n, cudaMemcpyHostToDevice);
			checkCUDAError("memcpy error dev_indices", __LINE__);

			Common::kernScatter << <blocksPerGrid, threadsPerBlock >> > (n, dev_odata, dev_idata, dev_bool, dev_indices);
			checkCUDAError("kernScatter Error", __LINE__);

			cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);
			checkCUDAError("memcpy error dev_odata to odata", __LINE__);

            // TODO
            timer().endGpuTimer();
			int count = (host_bool[n - 1]) ? indices[n - 1] + 1 : indices[n-1];
			delete[] host_bool;
			delete[] indices;
			cudaFree(dev_bool);
			cudaFree(dev_idata);
			cudaFree(dev_indices);
			cudaFree(dev_odata);
            return count;
        }
    }
}
