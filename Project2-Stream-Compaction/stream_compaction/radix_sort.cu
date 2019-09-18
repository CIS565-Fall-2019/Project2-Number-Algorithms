#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "radix_sort.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stream_compaction/efficient.h>

__global__ void Compute_e(int n, int bit, int *e_array, const int *i_array) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx < n) {
		int b = (i_array[idx] >> bit) & 1;
		e_array[idx] = (b == 0 ? 1 : 0);
	}
}

__global__ void Compute_t(int n, int *totalFalses, int *t_array, const int *f_array) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx < n) {
		t_array[idx] = idx - f_array[idx] + *totalFalses;
	}
}

__global__ void Compute_d(int n, int *d_array, const int *e_array, const int *f_array, const int *t_array) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx < n) {
		d_array[idx] = e_array[idx] ? f_array[idx] : t_array[idx];
	}
}

__global__ void Scatter(int n, int *o_array, const int *d_array, const int *i_array) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx < n) {
		o_array[d_array[idx]] = i_array[idx];
	}
}

__global__ void Add(int *arr1, int *arr2, int idx, int *totalFalses) {
	*totalFalses = arr1[idx] + arr2[idx];
}

namespace StreamCompaction {
	namespace RadixSort {
		void sort(int n, int *odata, const int *idata, int blockSize) {
			int *i_array, *e_array, *f_array, *t_array, *d_array, *o_array, *totalFalses;
			// Memory Allocation
			cudaMalloc((void**)&i_array, n * sizeof(int));
			checkCUDAErrorFn("cudaMalloc i_array failed!");

			cudaMalloc((void**)&e_array, n * sizeof(int));
			checkCUDAErrorFn("cudaMalloc e_array failed!");

			cudaMalloc((void**)&f_array, n * sizeof(int));
			checkCUDAErrorFn("cudaMalloc f_array failed!");

			cudaMalloc((void**)&t_array, n * sizeof(int));
			checkCUDAErrorFn("cudaMalloc t_array failed!");

			cudaMalloc((void**)&d_array, n * sizeof(int));
			checkCUDAErrorFn("cudaMalloc d_array failed!");

			cudaMalloc((void**)&o_array, n * sizeof(int));
			checkCUDAErrorFn("cudaMalloc o_array failed!");

			cudaMalloc((void**)&totalFalses, sizeof(int));
			checkCUDAErrorFn("cudaMalloc totalFalses failed!");

			cudaMemcpy(i_array, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

			dim3 fullBlocks((n + blockSize - 1) / blockSize);
			for (int i = 0; i < 10; i++) {
				Compute_e << < fullBlocks, blockSize >> > (n, i, e_array, i_array);
				StreamCompaction::Efficient::scanEfficientCUDA(n, f_array, e_array);
				Add << <1, 1 >> > (e_array, f_array, n - 1, totalFalses);
				Compute_t << < fullBlocks, blockSize >> > (n, totalFalses, t_array, f_array);
				Compute_d << < fullBlocks, blockSize >> > (n, d_array, e_array, f_array, t_array);
				Scatter << < fullBlocks, blockSize >> > (n, o_array, d_array, i_array);
				cudaMemcpy(i_array, o_array, sizeof(int) * n, cudaMemcpyDeviceToDevice);
			}
			cudaMemcpy(odata, o_array, sizeof(int) * n, cudaMemcpyDeviceToHost);

			// Freeing the Memory
			cudaFree(i_array);
			cudaFree(e_array);
			cudaFree(f_array);
			cudaFree(t_array);
			cudaFree(d_array);
			cudaFree(o_array);
		}

	}
}