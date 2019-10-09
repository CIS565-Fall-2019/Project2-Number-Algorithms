#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "common.h"
#include "radix.h"
#include "efficient.h"
#include "device_launch_parameters.h"
#include <math.h>

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)
#define MAX_BIT 6
#define blockSize 128

int* data;
int* output;

int* B;
int* E;
int* F;
int* T;
int* D;

namespace StreamCompaction {
	namespace Radix {
		using StreamCompaction::Common::PerformanceTimer;
		PerformanceTimer& timer()
		{
			static PerformanceTimer timer;
			return timer;
		}


		__global__ void compute_b_e(int n, int position, int* input, int* B, int* E) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) {
				return;
			}
			int value = input[index];
			int bit = (value >> position) & 1;
			if (bit == 0) {
				B[index] = 0;
				E[index] = 1;
			}
			else {
				B[index] = 1;
				E[index] = 0;
			}
		}

		__global__ void compute_total_falses(int n, int* totalFalses, int* E, int* F) {
			*totalFalses = E[n - 1] + F[n - 1];
		}

		__global__ void compute_t(int n, int* F, int* totalFalses, int* T) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) {
				return;
			}
			T[index] = index - F[index] + (*totalFalses);
		}

		__global__ void compute_d(int n, int* B, int* T, int* F, int* D) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) {
				return;
			}
			D[index] = B[index] ? T[index] : F[index];
		}

		__global__ void scatter(int n, int* indices, int* inp, int* op) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) {
				return;
			}
			op[indices[index]] = inp[index];
		}
		
		void sort(int n, int* odata, int*idata) {
			cudaMalloc((void**)&data, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc data failed!");

			cudaMalloc((void**)&output, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc output failed!");

			cudaMemcpy(data, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

			cudaMalloc((void**)&B, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc B failed!");

			cudaMalloc((void**)&E, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc E failed!");

			cudaMalloc((void**)&F, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc F failed!");

			cudaMalloc((void**)&T, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc T failed!");

			cudaMalloc((void**)&D, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc D failed!");

			int* totalFalses;
			cudaMalloc((void**)&totalFalses, sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc totalFalses failed!");
			
			timer().startGpuTimer();
			
			for (int i = 0; i < MAX_BIT; i++) {
				dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
				compute_b_e << <fullBlocksPerGrid, blockSize >>> (n, i, data, B, E);

				//Scan E and store in F
				StreamCompaction::Efficient::scan_device(n, F, E, blockSize);

				compute_total_falses << <1,1>> > (n, totalFalses, E, F);

				compute_t << <fullBlocksPerGrid, blockSize >> > (n, F, totalFalses, T);

				compute_d << <fullBlocksPerGrid, blockSize >> > (n, B, T, F, D);

				//Scatter as per D
				scatter << <fullBlocksPerGrid, blockSize >> > (n, D, data, output);

				//Copy output back to input
				cudaMemcpy(data, output, sizeof(int) * n, cudaMemcpyDeviceToDevice);
			}

			timer().endGpuTimer();

			cudaMemcpy(odata, output, sizeof(int) * n, cudaMemcpyDeviceToHost);
		}
	}
}