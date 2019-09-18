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
        void scan(unsigned long long int n, long long *odata, const long long *idata) {
			long long *dev_odata, *dev_idata;
			cudaMalloc((void**)&dev_odata, n * sizeof(long long));
			cudaMalloc((void**)&dev_idata, n * sizeof(long long));
			cudaMemcpy(dev_idata, idata, n * sizeof(long long), cudaMemcpyHostToDevice);
			timer().startGpuTimer();
			thrust::device_ptr<long long> dv_in(dev_idata);
			thrust::device_ptr<long long> dv_out(dev_odata);
            thrust::exclusive_scan(dv_in, dv_in + n, dv_out);
			timer().endGpuTimer();
			cudaMemcpy(odata, dev_odata, sizeof(long long) * n, cudaMemcpyDeviceToHost);
			cudaFree(dev_odata);
			cudaFree(dev_idata);
        }
    }
}
