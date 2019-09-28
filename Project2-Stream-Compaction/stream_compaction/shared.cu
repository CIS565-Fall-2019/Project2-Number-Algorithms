#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "shared.h"
#include "efficient.h"
#include "device_launch_parameters.h"


#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

//#define blockSize 128

int* dev_idata_shared;
int* padded_idata_shared;

namespace StreamCompaction {
	namespace Shared {
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
			int stride = 1 << (d + 1);
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


		__global__ void scan_array(int n, int* A, int* B, int* intermediate) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);

			if (index >= n) {
				return;
			}
			int BLOCKSIZE = blockDim.x;
			__shared__ int s[512];

			int tid = threadIdx.x;
			//Copy Global Memory to Shared
			s[tid] = A[threadIdx.x + (blockIdx.x * BLOCKSIZE)];

			int iterations = ilog2ceil(BLOCKSIZE);

			//Up Sweep
			for (int d = 0; d < ilog2(BLOCKSIZE); d++) {
				__syncthreads();
				int stride = 1 << (d + 1);
				int other_index = 1 << d;
				if ((tid) % stride == 0) {
					s[tid + stride - 1] += s[tid + other_index - 1];
				}
			}
			//__syncthreads();

			if (threadIdx.x == BLOCKSIZE - 1) {
				//Add last element of block (upsweep) to intermediate array
				intermediate[blockIdx.x] = s[BLOCKSIZE - 1];
				s[BLOCKSIZE - 1] = 0;
			}
			
			//__syncthreads();
			
			//Down Sweep
			for (int d = iterations - 1; d >= 0; d--) {
				__syncthreads();
				int left_index = 1 << (d);
				int right_index = 1 << (d + 1);
				if (tid % right_index == 0) {
					int temp = s[tid + left_index - 1];
					s[tid + left_index - 1] = s[tid + right_index - 1];
					s[tid + right_index - 1] += temp;
				}
			}

			//Copy Result Back to Global Memory
			 B[threadIdx.x + (blockIdx.x * BLOCKSIZE)] = s[threadIdx.x];
			
		}

		__global__ void scan_array_old(int n, int* A) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);

			if (index >= n) {
				return;
			}

			int iterations = ilog2ceil(n);

			//Up Sweep
			for (int d = 0; d < ilog2(n); d++) {
				__syncthreads();
				int stride = 1 << (d + 1);
				int other_index = 1 << d;
				if ((index) % stride == 0) {
					A[index + stride - 1] += A[index + other_index - 1];
				}
			}
			__syncthreads();
			//Down Sweep
			A[n - 1] = 0;

			for (int d = iterations - 1; d >= 0; d--) {
				__syncthreads();
				int left_index = 1 << (d);
				int right_index = 1 << (d + 1);
				if (index % right_index == 0) {
					int temp = A[index + left_index - 1];
					A[index + left_index - 1] = A[index + right_index - 1];
					A[index + right_index - 1] += temp;
				}
			}

		}

		__global__ void merge(int n, int* A, int* intermediate) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);

			if (index >= n) {
				return;
			}
			if(blockIdx.x > 0)
				A[index] = A[index] + intermediate[blockIdx.x];
		}

		/**
		 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
		 */
		void scan(int n, int *odata, const int *idata, int blockSize) {
			int* dev_odata;
			int* dev_intermediate;
			int* dev_intermediate_scan;
			
			int padded_size = 1 << (ilog2ceil(n));
			int numberOfBlocks = (padded_size + blockSize - 1) / blockSize;

			cudaMalloc((void**)&padded_idata_shared, padded_size * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc padded_idata_shared failed!");

			cudaMalloc((void**)&dev_odata, padded_size * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc padded_idata_shared failed!");

			cudaMalloc((void**)&dev_intermediate, numberOfBlocks * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc padded_idata_shared failed!");

			cudaMalloc((void**)&dev_intermediate_scan, numberOfBlocks * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc padded_idata_shared failed!");


			cudaMemset(padded_idata_shared, 0, padded_size * sizeof(int));
			cudaMemcpy(padded_idata_shared, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

			bool caught = false;
			try {
				timer().startGpuTimer();
			}
			catch (const std::exception& e) {
				caught = true;
			}

			dim3 fullBlocksPerGrid(numberOfBlocks);

			scan_array << <fullBlocksPerGrid, blockSize >> > (padded_size, padded_idata_shared, dev_odata, dev_intermediate);

			int* inter = (int*)malloc(numberOfBlocks * sizeof(int));
			/*printf("Number: %d", numberOfBlocks);
			printf("\nINTERMEDIATE\n");
			cudaMemcpy(inter, dev_intermediate, sizeof(int) * numberOfBlocks, cudaMemcpyDeviceToHost);
			printArray(numberOfBlocks, inter, true);*/

			//Scan the intermediate array (without shared memory in GPU)
			StreamCompaction::Efficient::scan_device(numberOfBlocks, dev_intermediate_scan, dev_intermediate, blockSize);

			

			int* inter2 = (int*)malloc(numberOfBlocks * sizeof(int));
			/*printf("INTERMEDIATE SCAN\n");
			cudaMemcpy(inter2, dev_intermediate_scan, sizeof(int) * numberOfBlocks, cudaMemcpyDeviceToHost);
			printArray(numberOfBlocks, inter2, true);*/


			//Add the elements of dev_intermediate to dev_odata
			merge << <fullBlocksPerGrid, blockSize >> > (padded_size, dev_odata, dev_intermediate_scan);

			//scan_array_old << <fullBlocksPerGrid, blockSize >> > (padded_size, padded_idata_shared);

			if (!caught) {
				timer().endGpuTimer();
			}

			cudaMemcpy(odata, dev_odata, sizeof(int) * padded_size, cudaMemcpyDeviceToHost);

			/*cudaFree(padded_idata_shared);
			cudaFree(dev_odata);
			cudaFree(dev_intermediate);
			cudaFree(dev_intermediate_scan);*/
		}

		//void scan_device(int n, int *odata, const int *idata, int blockSize) {
		//	int padded_size = 1 << (ilog2ceil(n));

		//	cudaMalloc((void**)&padded_idata_shared, padded_size * sizeof(int));
		//	checkCUDAErrorWithLine("cudaMalloc padded_idata_shared failed!");

		//	cudaMemset(padded_idata_shared, 0, padded_size * sizeof(int));
		//	cudaMemcpy(padded_idata_shared, idata, sizeof(int) * n, cudaMemcpyDeviceToDevice);

		//	bool caught = false;
		//	try {
		//		timer().startGpuTimer();
		//	}
		//	catch (const std::exception& e) {
		//		caught = true;
		//	}



		//	int iterations = ilog2(padded_size);
		//	dim3 fullBlocksPerGrid((padded_size + blockSize - 1) / blockSize);

		//	bool optimized = false;
		//	//Up-Sweep
		//	if (optimized) {
		//		int number_of_threads = padded_size;
		//		for (int d = 0; d < iterations; d++) {
		//			number_of_threads /= 2;
		//			dim3 fullBlocksPerGridUpSweep((number_of_threads + blockSize - 1) / blockSize);
		//			upSweepOptimized << <fullBlocksPerGridUpSweep, blockSize >> > (padded_size, d, padded_idata_shared);
		//		}
		//	}
		//	else {
		//		for (int d = 0; d < iterations; d++) {
		//			dim3 fullBlocksPerGrid((padded_size + blockSize - 1) / blockSize);
		//			upSweep << <fullBlocksPerGrid, blockSize >> > (padded_size, d, padded_idata_shared);
		//		}
		//	}


		//	//Down-Sweep
		//	cudaMemset(padded_idata_shared + (padded_size - 1), 0, sizeof(int));


		//	if (optimized) {
		//		int number_of_threads = 1;
		//		for (int d = iterations - 1; d >= 0; d--) {
		//			dim3 fullBlocksPerGridDownSweep((number_of_threads + blockSize - 1) / blockSize);
		//			downSweepOptimized << <fullBlocksPerGridDownSweep, blockSize >> > (padded_size, d, padded_idata_shared);
		//			number_of_threads *= 2;
		//		}
		//	}
		//	else {
		//		for (int d = iterations - 1; d >= 0; d--) {
		//			dim3 fullBlocksPerGrid((padded_size + blockSize - 1) / blockSize);
		//			downSweep << <fullBlocksPerGrid, blockSize >> > (padded_size, d, padded_idata_shared);
		//		}
		//	}
		//	if (!caught) {
		//		timer().endGpuTimer();
		//	}

		//	cudaMemcpy(odata, padded_idata_shared, sizeof(int) * n, cudaMemcpyDeviceToDevice);
		//}

		/**
		 * Performs stream compaction on idata, storing the result into odata.
		 * All zeroes are discarded.
		 *
		 * @param n      The number of elements in idata.
		 * @param odata  The array into which to store elements.
		 * @param idata  The array of elements to compact.
		 * @returns      The number of elements remaining after compaction.
		 */
		int compact(int n, int *odata, const int *idata, int blockSize) {


			cudaMalloc((void**)&dev_idata_shared, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_idata_shared failed!");

			cudaMemcpy(dev_idata_shared, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

			int* dev_bools;
			cudaMalloc((void**)&dev_bools, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_bools failed!");
			int* bools;
			bools = (int*)malloc(n * sizeof(int));
			int* indices;
			indices = (int*)malloc(n * sizeof(int));
			int *dev_indices;
			cudaMalloc((void**)&dev_indices, n * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_bools failed!");

			timer().startGpuTimer();
			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

			StreamCompaction::Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> > (n, dev_bools, dev_idata_shared);


			cudaMemcpy(bools, dev_bools, sizeof(int) * n, cudaMemcpyDeviceToHost);


			scan(n, indices, bools, blockSize);

			int output_length = indices[n - 1] + bools[n - 1];


			cudaMemcpy(dev_indices, indices, sizeof(int) * n, cudaMemcpyHostToDevice);

			int *dev_odata;
			cudaMalloc((void**)&dev_odata, output_length * sizeof(int));
			checkCUDAErrorWithLine("cudaMalloc dev_odata failed!");

			StreamCompaction::Common::kernScatter << <fullBlocksPerGrid, blockSize >> > (n, dev_odata, dev_idata_shared, dev_bools, dev_indices);

			cudaMemcpy(odata, dev_odata, sizeof(int) * output_length, cudaMemcpyDeviceToHost);

			timer().endGpuTimer();
			return output_length;
		}
	}
}
