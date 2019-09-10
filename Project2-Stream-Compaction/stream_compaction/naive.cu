#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define blockSize 128

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        
        // out put and two input buffers to ping pong off of
        // 
        __global__ void kernel_scan( int buff_length, int array_pos, int* odata, int* idata )
        {
            tid = (blockIdx.x * blockDim.x) + threadIdx.x;
            if( tid >= buff_length )
            {
                __syncthreads();
                return;
            }
            
            if( tid <= array_pos ) // already been computed
            {
                __syncthreads(); // need this or will lock ): 
                odata[thid] = (tid == 0) ?  0 :  idata[tid];
                return;
            }
            
            //the original way ... but this wont work because we have threads manipulating the neighbors
            //odata[tid] = idata[tid-1] + odata[tid-1];
            int prev = odata[tid-1];
            // read your neighbors and wait
            __syncthreads();
            // now we can write as before.
            odata[tid] = idata[tid-1] + prev;
            return;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            int* dev_temp_in;
            int* dev_input;
            int fullBlocksPerGrid = ((n + blockSize - 1) / blockSize);
            //create cuda buffers and copy data over
            cudaMalloc((void**)&dev_temp_in, n * sizeof(int));
            checkCUDAErrorWithLine("malloc temp in failed!");
            cudaMalloc((void**)&dev_input, n * sizeof(int));
            checkCUDAErrorWithLine("malloc devinput failed!");
            // copy data to device n or n*size? check
            cudaCopy( dev_input, idata, n, cudaMemcpyHostToDevice );
            checkCUDAErrorWithLine("copy failed!");
            
            // think this itr count needs to be changed
            for(int i = 0; i < n; i++)
            {
                kernel_scan<<< fullBlocksPerGrid, blockSize >>>(n,i,dev_temp_in,dev_input);
                checkCUDAErrorWithLine("scan failed!");
                std::swap(dev_temp_in,dev_input);
                checkCUDAErrorWithLine("swap failed!");
            }
            
            cudaCopy( odata, dev_input, n, cudaMemcpyDeviceToHost );
            checkCUDAErrorWithLine("copy out failed!");
            cudaFree(dev_input);
            checkCUDAErrorWithLine("free input failed!");
            cudaFree(dev_temp_in);
            checkCUDAErrorWithLine("free temp failed!");
            
            timer().endGpuTimer();
        }
    }
}
