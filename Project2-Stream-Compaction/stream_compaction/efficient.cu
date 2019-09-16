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
          if (k < n && k % (pow2d * 2) == 0) {
            idata[k + pow2d * 2 - 1] += idata[k + pow2d - 1];
          }
        }

        __global__ void kernDownSweep(int n, int pow2d, int *idata) {
          int k = threadIdx.x + (blockIdx.x * blockDim.x);
          if (k < n && k % (pow2d * 2) == 0) {
            int t = idata[k + pow2d - 1];
            idata[k + pow2d - 1] = idata[k + pow2d * 2 - 1];
            idata[k + pow2d * 2 - 1] += t;
          }
        }

        __global__ void kernIsNonZero(int n, int *isnonzero, int *idata) {
          int idx = threadIdx.x + (blockIdx.x * blockDim.x);
          if (idx < n) {
            isnonzero[idx] = (idata[idx] == 0) ? 0 : 1;
          }
        }

        __global__ void kernCompact(int n, int *odata, int *idata, int* isnonzero, int* indices) {
          int idx = threadIdx.x + (blockIdx.x * blockDim.x);
          if (idx < n && isnonzero[idx]) {
            odata[indices[idx]] = idata[idx];
          }
        }

        void scan_implementation(int n, int *odata, const int *idata) {
          int size = pow(2, ilog2ceil(n));
          int memsize = size * sizeof(int);
          int *dev_in, *dev_out, *dev_mid;
          cudaMalloc((void**)&dev_in, memsize);
          cudaMemcpy(dev_in, idata, n * sizeof(int), cudaMemcpyHostToDevice);

          dim3 blockSize(128);
          dim3 gridSize((size + 127) / 128);
          for (int d = 0; d < ilog2ceil(n); d++) {
            kernUpSweep << <gridSize, blockSize >> > (size, pow(2, d), dev_in);
          }

          cudaMemset((void*)&(dev_in[size - 1]), 0, sizeof(int));

          for (int d = ilog2ceil(n) - 1; d >= 0; d--) {
            kernDownSweep << <gridSize, blockSize >> > (size, pow(2, d), dev_in);
          }

          cudaMemcpy(odata, dev_in, n * sizeof(int), cudaMemcpyDeviceToHost);
          cudaFree(dev_in);
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            scan_implementation(n, odata, idata);
            timer().endGpuTimer();
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
            timer().startGpuTimer();
            // TODO
            size_t size = n * sizeof(int);

            // allocate device memory
            int *dev_odata, *dev_idata, *dev_isnonzero, *dev_indices;
            cudaMalloc((void**)&dev_idata, size);
            cudaMalloc((void**)&dev_odata, size);
            cudaMalloc((void**)&dev_isnonzero, size);
            cudaMalloc((void**)&dev_indices, size);

            // allocate host memory
            int *isnonzero, *indices;
            isnonzero = (int*)malloc(size);
            indices = (int*)malloc(size);
            
            // calculate block size and grid size
            int blockSize = 128;
            int gridSize = (n + blockSize - 1) / blockSize;
            
            // identify non-zero elements
            cudaMemcpy(dev_idata, idata, size, cudaMemcpyHostToDevice);
            kernIsNonZero<<<gridSize, blockSize>>>(n, dev_isnonzero, dev_idata);
            cudaMemcpy(isnonzero, dev_isnonzero, size, cudaMemcpyDeviceToHost);

            // exclusive scan
            scan_implementation(n, indices, isnonzero);
            int n_compact = isnonzero[n - 1] ? indices[n - 1] + 1 : indices[n - 1];
            cudaMemcpy(dev_indices, indices, size, cudaMemcpyHostToDevice);

            // stream compaction
            kernCompact<<<gridSize, blockSize>>>(n, dev_odata, dev_idata, dev_isnonzero, dev_indices);
            cudaMemcpy(odata, dev_odata, size, cudaMemcpyDeviceToHost);
             
            // free device memory
            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_isnonzero);
            cudaFree(dev_indices);

            // free host memory
            free(isnonzero);
            free(indices);

            timer().endGpuTimer();
            return n_compact;
        }
    }
}
