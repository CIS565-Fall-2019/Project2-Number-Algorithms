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

		__global__ void kernel_inclusive_to_exclusive(int buff_length, int* odata, int* idata)
		{
			int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (tid >= buff_length)
			{
				return;
			}
			
			if (tid == 0)
			{
				odata[tid] = 0;
			}
			else
			{
				odata[tid] = idata[tid - 1];
			}
			

		}
        
        // out put and two input buffers to ping pong off of
        // 
        __global__ void kernel_scan( int pow, int buff_length, int depth, int* odata, int* idata )
        {
            int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
            
			// get rid of any stray threads
			if( tid >= buff_length )
            {
               // __syncthreads();
                return;
            }
            
			// exclusive scan will always lead to a race condition with the 0th element
			// lets try making inclusive and Then exclusive.
            if( tid < pow ) // already been computed
            {
             //   __syncthreads(); // need this or will lock ): 
                odata[tid] = idata[tid];
                return;
            }
            //the original way ... but this wont work because we have threads manipulating the neighbors
            //odata[tid] = idata[tid-1] + odata[tid-1];
            int prev = idata[tid-pow];
            // read your neighbors and wait
           // __syncthreads();
            // now we can write as before.
            odata[tid] = idata[tid] + prev;
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
            dim3 fullBlocksPerGrid = ((n + blockSize - 1) / blockSize);
            //create cuda buffers and copy data over
            cudaMalloc((void**)&dev_temp_in, n * sizeof(int));
			checkCUDAErrorFn("malloc temp in failed!");
            cudaMalloc((void**)&dev_input, n * sizeof(int));
			checkCUDAErrorFn("malloc devinput failed!");
            // copy data to device n or n*size? check
            cudaMemcpy( dev_input, idata, n * sizeof(int), cudaMemcpyHostToDevice );
			checkCUDAErrorFn("copy failed!");
			cudaMemcpy( dev_temp_in, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAErrorFn("copy failed!");


			int depth = ilog2ceil(n);
            
            // think this itr count needs to be changed
            for(int i = 1; i <= depth; i++)
            {
				int pow2 = (1 << (i - 1));
				printf("i %d  -> depth %d \n ", i, pow2);
				kernel_scan<<< fullBlocksPerGrid, blockSize >>>(pow2,n,i,dev_temp_in,dev_input);
				checkCUDAErrorFn("scan failed!");
                std::swap(dev_temp_in,dev_input);
				checkCUDAErrorFn("swap failed!");
            }

			cudaMemcpy(odata, dev_temp_in, n * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAErrorFn("copy out failed!");

			kernel_inclusive_to_exclusive<<< fullBlocksPerGrid, blockSize >> > (n,dev_temp_in, dev_input);
			
            cudaMemcpy( odata, dev_temp_in, n * sizeof(int), cudaMemcpyDeviceToHost );
			checkCUDAErrorFn("copy out failed!");

            cudaFree(dev_input);
			checkCUDAErrorFn("free input failed!");
            cudaFree(dev_temp_in);
			checkCUDAErrorFn("free temp failed!");
            
            timer().endGpuTimer();
        }
    }
}
