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
        __global__ void kernNaiveScan(int N, int lowbound, int *odata, int *idata) {
          int thid = threadIdx.x + (blockIdx.x * blockDim.x);
          if (thid >= lowbound) {
            odata[thid] = idata[thid] + idata[thid - lowbound];
          }
          else {
            odata[thid] = idata[thid];
          }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // TODO
            // Declare two device buffer
            int *dev_data[2];
              
            // Allocate device memory
            cudaMalloc((void**)&dev_data[0], n * sizeof(int));
            checkCUDAErrorFn("cudaMalloc dev_idata failed!");

            cudaMalloc((void**)&dev_data[1], n * sizeof(int));
            checkCUDAErrorFn("cudaMalloc dev_odata failed!");

            // Copy data to the first device buffer
            cudaMemcpy(dev_data[0], idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAErrorFn("cudaMemcpy (host to device) failed!");
            
            timer().startGpuTimer();
            dim3 fullBlocksPerGrid((n + BLOCKSIZE - 1) / BLOCKSIZE);
            for (int d = 1; d <= ilog2ceil(n); d++) {
              kernNaiveScan << <fullBlocksPerGrid, dim3(BLOCKSIZE) >> > (n, 1 << (d - 1), dev_data[d % 2], dev_data[(d + 1) % 2]);
            }
            timer().endGpuTimer();

            // Convert inclusive scan to exclusive scan, shift right and insert identity.
            cudaMemcpy(odata + 1, dev_data[ilog2ceil(n) % 2], (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAErrorFn("cudaMemcpy (device to host) failed!");
            odata[0] = 0;
            
            // Free device memory
            cudaFree(dev_data[0]);
            cudaFree(dev_data[1]);
        }
    }
}
