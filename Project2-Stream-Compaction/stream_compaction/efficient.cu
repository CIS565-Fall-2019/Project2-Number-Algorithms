#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        using StreamCompaction::Common::kernMapToBoolean;
        using StreamCompaction::Common::kernScatter;
        //intermediate arrays
        int* temp_out;
        int* temp_bool;
        int* temp_scattered;
        int* temp_in;

        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        __global__ void kernComputePartialUpSweep(int n, const int pow_d_plus_one, int* idata)
        {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n)
            {
                return;
            }
            //no need to store the idata to temp_in, it is already there
            int pow_d = pow_d_plus_one / 2;
            //by 2^(d+1) means a stride of two
            if (index % pow_d_plus_one == 0)  idata[index + pow_d_plus_one - 1] += idata[index + pow_d - 1];
        }

        __global__ void kernComputePartialDownSweep(int n, const int pow_d_plus_one, int* odata)
        {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n)
            {
                return;
            }
            //no need to store the idata to temp_in, it is already there
            int pow_d = pow_d_plus_one / 2;
            if (index % pow_d_plus_one == 0)
            {
                int temp = odata[index + pow_d - 1];
                odata[index + pow_d - 1] = odata[index + pow_d_plus_one - 1];
                odata[index + pow_d_plus_one - 1] += temp;
            }
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            //init two new memory
            // TODO
            //padding to the nearest 2^d
            int logn = ilog2ceil(n);
            int powd = std::pow(2, logn);
            //init temp_out
            cudaMalloc((void**)&temp_out, powd * sizeof(int));
            checkCUDAError("cudaMalloc temp_out failed!");            
            //assign idata value to temp_out
            cudaMemcpy(temp_out, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
            dim3 blocksPerGrid(gridSize);
            dim3 threadsPerBlock(BLOCK_SIZE);
            ////first compute the new velocity
            int ceil = logn - 1;
            timer().startGpuTimer();
            for (int offset = 0; offset <= ceil; ++offset) {
                const int pow_d_plus_one = std::pow(2, offset + 1);
                //set last element to zero in temp_out in kernel
                kernComputePartialUpSweep << < blocksPerGrid, threadsPerBlock >> > (n, pow_d_plus_one, temp_out);
            }

            //assign 0 to root element
            int last_value = 0;
            cudaMemset(temp_out + powd - 1, last_value, sizeof(int));
            checkCUDAError("cudaMemSet temp_out last value to 0 failed!");
            //debug note --- If you have wrong operation in kernel, this will mess up your device memory, and next time you want to initalize them, it will fail -- so it is not problem in initialization, it is the operation having problem
            for (int offset = ceil; offset >= 0; --offset) {
                const int pow_d_plus_one = std::pow(2, offset + 1);
                kernComputePartialDownSweep << < blocksPerGrid, threadsPerBlock >> > (n, pow_d_plus_one, temp_out);
            }
            timer().endGpuTimer();

            cudaMemcpy(odata, temp_out, n * sizeof(int), cudaMemcpyDeviceToHost); //Using the wrong tag to copy will cause the next time initalization fail

            cudaFree(temp_out);
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
            int logn = ilog2ceil(n);
            int powd = std::pow(2, logn);
            //initialize intermediate arrays
            cudaMalloc((void**)&temp_bool, powd * sizeof(int));
            checkCUDAError("cudaMalloc temp_bool failed!");
            cudaMalloc((void**)&temp_scattered, powd * sizeof(int));
            checkCUDAError("cudaMalloc temp_scattered failed!");
            cudaMalloc((void**)&temp_in, powd * sizeof(int));
            checkCUDAError("cudaMalloc temp_in failed!");
            cudaMalloc((void**)&temp_out, powd * sizeof(int));
            checkCUDAError("cudaMalloc temp_out failed!");
            cudaMemcpy(temp_in, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy idata to temp_in failed!");
            int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
            dim3 blocksPerGrid(gridSize);
            dim3 threadsPerBlock(BLOCK_SIZE);

            //map values to bool array
            Common::kernMapToBoolean << < blocksPerGrid, threadsPerBlock >> > (n, temp_bool, temp_in);
            
            //assign bool value to temp_scattered
            cudaMemcpy(temp_scattered, temp_bool, n * sizeof(int), cudaMemcpyDeviceToDevice);
            checkCUDAError("cudaMemcpy temp_scattered failed!");
            //apply scan on bool array
            timer().startGpuTimer();
            int ceil = logn - 1;
            for (int offset = 0; offset <= ceil; ++offset) {
                const int pow_d_plus_one = std::pow(2, offset + 1);
                //set last element to zero in temp_out in kernel
                kernComputePartialUpSweep << < blocksPerGrid, threadsPerBlock >> > (n, pow_d_plus_one, temp_scattered);
            }

            //assign 0 to root element
            int last_value = 0;
            cudaMemset(temp_scattered + powd - 1, last_value, sizeof(int));
            checkCUDAError("cudaMemSet temp_scattered last value to 0 failed!");
            for (int offset = ceil; offset >= 0; --offset) {
                const int pow_d_plus_one = std::pow(2, offset + 1);
                kernComputePartialDownSweep << < blocksPerGrid, threadsPerBlock >> > (n, pow_d_plus_one, temp_scattered);
            }

            //got the correct indices of non-zero element in the array
            //map to odata
            Common::kernScatter << < blocksPerGrid, threadsPerBlock >> > (n, temp_out, temp_in, temp_bool, temp_scattered);

            timer().endGpuTimer();
            //copy from temp_out to odata
            cudaMemcpy(odata, temp_out, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy temp_out to odata failed!");

            int last_scattered = 0;
            int last_bool = 0;
            cudaMemcpy(&last_scattered, &temp_scattered[n - 1], sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&last_bool, &temp_bool[n - 1], sizeof(int), cudaMemcpyDeviceToHost);
            int odata_size = last_scattered + last_bool;

            cudaFree(temp_bool);
            cudaFree(temp_scattered);
            return odata_size;
        }
    }
}
