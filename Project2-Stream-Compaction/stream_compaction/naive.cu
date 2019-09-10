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
        
        // out put and two input buffers to ping pong off of
        // 
        __global__ void kernel_scan( int n, int* odata, int* idata )
        {
            if( tid <= n ) // already been computed
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
            //create cuda buffers and copy data over
            cudaMalloc((void**)&dev_temp_in, n * sizeof(int));
            cudaMalloc((void**)&dev_input, n * sizeof(int));
            // copy data to device n or n*size? check
            cudaCopy( dev_input, idata, n, cudaMemcpyHostToDevice );
          
            
            // think this itr count needs to be changed
            for(int i = 0; i < n; i++)
            {
                kernel_scan(n-i,dev_temp_in,dev_input);
                std::swap(dev_temp_in,dev_input);
            }
            
            cudaCopy( odata, dev_input, n, cudaMemcpyDeviceToHost );
                
            cudaFree(dev_input);
            cudaFree(dev_temp_in);
            
            timer().endGpuTimer();
        }
    }
}
