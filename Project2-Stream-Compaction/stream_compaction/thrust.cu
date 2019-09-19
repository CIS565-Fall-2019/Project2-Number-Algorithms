#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
    namespace Thrust {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			int *dev_in;
			cudaMalloc((void**)&dev_in, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_in failed!");

			// copy input to device buffer
			cudaMemcpy(dev_in, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy dev_in idata failed!");

			thrust::device_vector<int> dev_thrust_in(dev_in, dev_in + n);
			thrust::device_vector<int> dev_thrust_out(n);

            timer().startGpuTimer();
            thrust::exclusive_scan(dev_thrust_in.begin(), dev_thrust_in.end(), dev_thrust_out.begin());
			checkCUDAError("thrust::exclusive_scan failed!");
            timer().endGpuTimer();

			int *dev_out = thrust::raw_pointer_cast(dev_thrust_out.data());
			cudaMemcpy(odata, dev_out, n * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy odata dev_out failed!");

			cudaFree(dev_in);
			checkCUDAError("cudaFree dev_in failed!");
        }
    }
}
