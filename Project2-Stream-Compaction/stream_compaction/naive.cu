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
        __global__ void kernel_scan( int n, int* odata, int* idata )
        {
            if( tid >= n ) // already been computed so just copy over
            {
                odata = idata;
            }
            
            idata[
            
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            
            //create cuda buffers and copy data over
            cudaalloc(temp_in);
            cudaalloc(input);
            cudacopy(idata,input);
            
            o_data[0] = 0;
            
            
            timer().endGpuTimer();
        }
    }
}
