#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "radix.h"

namespace StreamCompaction {
    namespace Radix {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer() {
            static PerformanceTimer timer;
            return timer;
        }

		__global__ void Init(int n, int *data, int value) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) {
				return;
			}
			data[index] = 0;
		}

		//generate b_array
		__global__ void kernGetBandEArray(int n, int bit, const int* idata, int *b_arr, int* e_arr) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;//threadindex
			if (index >= n) {
				return;
			}
			if ((idata[index]>>bit) & 1) {
				b_arr[index] = 1;
				e_arr[index] = 0;
			} else {
				b_arr[index] = 0;
				e_arr[index] = 1;
			}
		}

		__global__ void kernGetTArray(int n, const int *f_arr, int total, int* t_arr) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) {
				return;
			}
			t_arr[index] = index - f_arr[index] + total;
		}

		__global__ void kernGetDArray(int n, const int *b_arr, const int *t_arr, const int *f_arr, int* d_arr) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) {
				return;
			}
			//if b is 1 -> t
			d_arr[index] = b_arr[index] ? t_arr[index] : f_arr[index];
		}

		__global__ void kernRearrange(int n, int *d_arr, int *data) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) {
				return;
			}
			data[d_arr[index]] = data[index];
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void radix_sort(int n, int bits_num, int *odata, const int *idata) {
			int* dev_data;
			int* b_array;
			int* e_array;
			int* f_array;
			int *t_array;
			int *d_array;

			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

			//malloc memory initial b and e with -1 and f with 0(for kern)
			cudaMalloc((void**)&dev_data, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_data failed!");
			cudaMalloc((void**)&b_array, n * sizeof(int));
			checkCUDAError("cudaMalloc b_array failed!");
			Init << < fullBlocksPerGrid, blockSize >> > (n, b_array, -1);
			cudaMalloc((void**)&e_array, n * sizeof(int));
			checkCUDAError("cudaMalloc e_array failed!");
			Init << < fullBlocksPerGrid, blockSize >> > (n, e_array, -1);
			cudaMalloc((void**)&f_array, n * sizeof(int));
			checkCUDAError("cudaMalloc f_array failed!");
			Init<< < fullBlocksPerGrid, blockSize >> >(n, f_array, 0);
			cudaMalloc((void**)&t_array, n * sizeof(int));
			checkCUDAError("cudaMalloc t_array failed!");
			Init << < fullBlocksPerGrid, blockSize >> > (n, t_array, -1);
			cudaMalloc((void**)&d_array, n * sizeof(int));
			checkCUDAError("cudaMalloc d_array failed!");
			Init << < fullBlocksPerGrid, blockSize >> > (n, d_array, -1);
			cudaDeviceSynchronize();

			//mempy
			cudaMemcpy(dev_data, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy dev_data failed!");

			//start time 
			timer().startGpuTimer();
			//execute sort bits_num times
			int totalFlases = 0;
			for (int i = 0; i < bits_num; i++) {
				//generate b and e arr
				kernGetBandEArray << <fullBlocksPerGrid, blockSize>> >(n, i, dev_data, b_array, e_array);
				//generate f arr
				StreamCompaction::Efficient::scan(n, f_array, e_array);
				int *temp = new int[n];
				cudaMemcpy(temp, f_array, sizeof(int) * n, cudaMemcpyDeviceToHost);
				/*std::cout << "f_array" << std::endl;
				for (int i = 0; i < n; i++) {
					std::cout << temp[i] << std::endl;
				}*/
				int *temp2 = new int[n];
				cudaMemcpy(temp2, e_array, sizeof(int) * n, cudaMemcpyDeviceToHost);
				/*std::cout << "e_array" << std::endl;
				for (int i = 0; i < n; i++) {
					std::cout << temp2[i] << std::endl;
				}*/
				totalFlases = temp[n - 1] + temp2[n - 1];
				//std::cout <<"total: "<< totalFlases << std::endl;
				//get t arr
				kernGetTArray << <fullBlocksPerGrid, blockSize >> >(n, f_array, totalFlases, t_array);
				//get d array
				kernGetDArray << <fullBlocksPerGrid, blockSize >> >(n, b_array, t_array, f_array, d_array);
				//scatter
				kernRearrange << <fullBlocksPerGrid, blockSize >> >(n, d_array, dev_data);
			}
			timer().endGpuTimer();
			//end gpu time

			cudaMemcpy(odata, dev_data, sizeof(int) * n, cudaMemcpyDeviceToHost);//get the result
			checkCUDAError("get odata failed!\n");

			//free
			cudaFree(dev_data);
			cudaFree(b_array);
			cudaFree(e_array);
			cudaFree(f_array);
			cudaFree(t_array);
			cudaFree(d_array);
        }
    }
}
