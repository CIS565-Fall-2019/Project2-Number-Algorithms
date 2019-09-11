#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

/*! Block size used for CUDA kernel launch. */
#define blockSize 128

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
		__global__ void kernScan(int N, int p, int *odata, int *idata) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= N) {
				return;
			}
			if (index >= p) {
				odata[index] = idata[index - p] + idata[index];
			}
			else {
				odata[index] = idata[index];
			}

		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			int *dev_idata;
			int *dev_odata;
			int *temp;

			cudaMalloc((void**)&dev_idata, n * sizeof(int));
			//checkCUDAErrorWithLine("cudaMalloc dev_idata failed!");

			cudaMalloc((void**)&dev_odata, n * sizeof(int));
			//checkCUDAErrorWithLine("cudaMalloc dev_odata failed!");

			cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

			dim3 threadsPerBlock(blockSize);
			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

			timer().startGpuTimer();
            // TODO
			for (int d = 1; d <= ilog2ceil(n); d++) {
				//int p = 1 << (d - 1);
				int p = pow(2,d-1);
				kernScan << <fullBlocksPerGrid, threadsPerBlock >> > (n, p, dev_odata, dev_idata);
				cudaThreadSynchronize();
				temp = dev_idata;
				dev_idata = dev_odata;
				dev_odata = temp;
			}
            timer().endGpuTimer();

			cudaMemcpy(odata + 1, dev_idata, sizeof(int) * n - 1, cudaMemcpyDeviceToHost);
			odata[0] = 0;

			cudaFree(dev_idata);
			cudaFree(dev_odata);
        }
    }
}
