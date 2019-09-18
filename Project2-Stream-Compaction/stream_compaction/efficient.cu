#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpSweep(int n, int pow2d, int *idata) {
          int k = threadIdx.x + (blockIdx.x * blockDim.x);
          if (k >= n) {
            return;
          }
          int pow2d2 = pow2d * 2;
          idata[k * pow2d2 + pow2d2 - 1] += idata[k * pow2d2 + pow2d - 1];
        }

        __global__ void kernDownSweep(int n, int pow2d, int *idata) {
          int k = threadIdx.x + (blockIdx.x * blockDim.x);
          if (k >= n) {
            return;
          }
          int pow2d2 = pow2d * 2;
          int t = idata[k * pow2d2 + pow2d - 1];
          idata[k * pow2d2 + pow2d - 1] = idata[k * pow2d2 + pow2d2 - 1];
          idata[k * pow2d2 + pow2d2 - 1] += t;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // TODO
            int size = pow(2, ilog2ceil(n));
            int memsize = size * sizeof(int);
            int *dev_in;
            cudaMalloc((void**)&dev_in, memsize);
            cudaMemcpy(dev_in, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            dim3 blockSize(BLOCKSIZE);
            dim3 gridSize((size + BLOCKSIZE - 1) / BLOCKSIZE);
            int dmax = ilog2ceil(n);

            for (int d = 0; d < dmax; d++) {
              int n_threads = 1 << (dmax - 1 - d);
              gridSize = dim3((n_threads + BLOCKSIZE - 1) / BLOCKSIZE);
              kernUpSweep << <gridSize, blockSize >> > (n_threads, 1 << d, dev_in);
            }

            cudaMemset((void*)&(dev_in[size - 1]), 0, sizeof(int));

            for (int d = dmax - 1; d >= 0; d--) {
              int n_threads = 1 << (dmax - 1 - d);
              gridSize = dim3((n_threads + BLOCKSIZE - 1) / BLOCKSIZE);
              kernDownSweep << <gridSize, blockSize >> > (n_threads, 1 << d, dev_in);
            }

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_in, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_in);
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
            // TODO
            size_t size = n * sizeof(int);
            int n2 = pow(2, ilog2ceil(n));
            size_t size2 = n2 * sizeof(int);

            // allocate device memory
            int *dev_odata, *dev_idata, *dev_bools, *dev_indices;
            cudaMalloc((void**)&dev_idata, size);
            cudaMalloc((void**)&dev_odata, size);
            cudaMalloc((void**)&dev_bools, size);
            cudaMalloc((void**)&dev_indices, size2);

            // allocate host memory
            int *bools, *indices;
            bools = (int*)malloc(size);
            indices = (int*)malloc(size);
            
            // calculate block size and grid size
            int gridSize = (n + BLOCKSIZE - 1) / BLOCKSIZE;
            int gridSize2 = (n2 + BLOCKSIZE - 1) / BLOCKSIZE;
            
            // copy memory to device
            cudaMemcpy(dev_idata, idata, size, cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            
            // identify non-zero elements
            Common::kernMapToBoolean << <gridSize, BLOCKSIZE >> > (n, dev_bools, dev_idata);

            // exclusive scan
            cudaMemcpy(dev_indices, dev_bools, size, cudaMemcpyDeviceToDevice);

            int dmax = ilog2ceil(n2);
            int n_threads;
            for (int d = 0; d < dmax; d++) {
              int n_threads = 1 << (dmax - 1 - d);
              gridSize2 = (n_threads + BLOCKSIZE - 1) / BLOCKSIZE;
              kernUpSweep << <gridSize2, BLOCKSIZE >> > (n_threads, 1 << d, dev_indices);
            }

            cudaMemset((void*)&(dev_indices[n2 - 1]), 0, sizeof(int));

            for (int d = dmax - 1; d >= 0; d--) {
              int n_threads = 1 << (dmax - 1 - d);
              gridSize2 = (n_threads + BLOCKSIZE - 1) / BLOCKSIZE;
              kernDownSweep << <gridSize2, BLOCKSIZE >> > (n_threads, 1 << d, dev_indices);
            }
            
            // stream compaction
            Common::kernScatter << <gridSize, BLOCKSIZE >> > (n, dev_odata, dev_idata, dev_bools, dev_indices);

            timer().endGpuTimer();

            // copy memory to host
            cudaMemcpy(indices, dev_indices, size, cudaMemcpyDeviceToHost);
            cudaMemcpy(bools, dev_bools, size, cudaMemcpyDeviceToHost);
            cudaMemcpy(odata, dev_odata, size, cudaMemcpyDeviceToHost);
            int n_compact = bools[n - 1] ? indices[n - 1] + 1 : indices[n - 1];
             
            // free device memory
            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_bools);
            cudaFree(dev_indices);

            // free host memory
            free(bools);
            free(indices);

            return n_compact;
        }
    }
}
