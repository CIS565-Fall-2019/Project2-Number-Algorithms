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
			int bufferLength = n;
			int *dev_inputArray;
			int *dev_outputArray;

			cudaMalloc((void**)&dev_inputArray, bufferLength * sizeof(int));
			checkCUDAError("cudaMalloc dev_inputArray failed!");

			cudaMalloc((void**)&dev_outputArray, bufferLength * sizeof(int));
			checkCUDAError("cudaMalloc dev_outputArray failed!");

			cudaMemset(dev_inputArray, 0, bufferLength * sizeof(int));
			cudaMemset(dev_outputArray, 0, bufferLength * sizeof(int));

			cudaMemcpy(dev_inputArray, idata, n * sizeof(int), cudaMemcpyHostToDevice);

			thrust::device_ptr<int> dev_thrust_inputArray = thrust::device_pointer_cast<int>(dev_inputArray);
			thrust::device_ptr<int> dev_thrust_outputArray = thrust::device_pointer_cast<int>(dev_outputArray);

            timer().startGpuTimer();
            // TODO use `thrust::exclusive_scan`
            // example: for device_vectors dv_in and dv_out:
            // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());
			thrust::exclusive_scan(dev_thrust_inputArray, dev_thrust_inputArray + bufferLength, dev_thrust_outputArray);

			cudaMemcpy(odata, dev_outputArray, n * sizeof(int), cudaMemcpyDeviceToHost);

            timer().endGpuTimer();
        }
    }
}
