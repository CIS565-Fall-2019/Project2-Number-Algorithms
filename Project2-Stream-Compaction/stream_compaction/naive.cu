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
        // TODO: __global__

        __global__ void kernComputePartialNaive(int n,  int pow_d, int* temp_in, int* temp_out)
        {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n)
            {
                return;
            }
            //no need to store the idata to temp_in, it is already there
            if (index >= pow_d)  temp_out[index] = temp_in[index - pow_d] + temp_in[index];
            else temp_out[index] = temp_in[index];
        }

        __global__ void kernShiftTempOut(int n, int* temp_in, int* temp_out)
        {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n)
            {
                return;
            }

            if (index == 0)
            {
                temp_out[index] = 0;
            }
            else
            {
                temp_out[index] = temp_in[index - 1];
            }
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int* temp_in;
            int* temp_out;
            // TODO
            //init two new memory
            cudaMalloc((void**)&temp_in, n * sizeof(int));
            checkCUDAError("cudaMalloc temp_in failed!");

            cudaMalloc((void**)&temp_out, n * sizeof(int));
            checkCUDAError("cudaMalloc temp_out failed!");
            int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
            dim3 blocksPerGrid(gridSize);
            dim3 threadsPerBlock(BLOCK_SIZE);

            //copy idata to temp_in
            timer().startGpuTimer();
            cudaMemcpy(temp_in, idata, n * sizeof(int), cudaMemcpyHostToDevice); //idata is in global memory
            checkCUDAError("cudaMemcpy idata to temp_in failed!");
            //temp pointer

            //first compute the new velocity
            int ceil = ilog2ceil(n);
            for (int offset = 1; offset <= ceil; ++offset) {
                const int pow_d_minus_one = std::pow(2, offset - 1);
                kernComputePartialNaive << < blocksPerGrid, threadsPerBlock >> > (n, pow_d_minus_one, temp_in, temp_out);

                int* temp = temp_in;
                temp_in = temp_out;
                temp_out = temp;

            }
            timer().endGpuTimer();

            //shift temp_out by one to get exclusive scan from reduction
            kernShiftTempOut << < blocksPerGrid, threadsPerBlock >> > (n, temp_in,temp_out);
            //pass the result from temp array to result
            cudaMemcpy(odata, temp_out, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy temp_out to odata failed!");
            cudaFree(temp_in);
            cudaFree(temp_out);
        }
    }
}
