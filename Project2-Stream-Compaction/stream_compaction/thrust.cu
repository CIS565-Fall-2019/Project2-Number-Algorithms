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
        //intermediate arrays
        int* temp_in;
        int* temp_out;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {

            // TODO use `thrust::exclusive_scan`
            cudaMalloc((void**)&temp_in, n * sizeof(int));
            checkCUDAError("cudaMalloc temp_in failed!");
            cudaMalloc((void**)&temp_out, n * sizeof(int));
            checkCUDAError("cudaMalloc temp_out failed!");

            cudaMemcpy(temp_in, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            timer().startGpuTimer();
            //warp to a device ptr
            thrust::device_ptr<int> dev_in(temp_in);
            thrust::device_ptr<int> dev_out(temp_out);
            // example: for device_vectors dv_in and dv_out:
            // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());
            thrust::exclusive_scan(dev_in, dev_in + n, dev_out);
            timer().endGpuTimer();
            cudaMemcpy(odata, temp_out, n * sizeof(int), cudaMemcpyDeviceToHost);

        }
    }
}
