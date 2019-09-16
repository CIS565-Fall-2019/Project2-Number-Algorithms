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
		//https://thrust.github.io/doc/group__prefixsums_ga7be5451c96d8f649c8c43208fcebb8c3.html
        void scan(int n, int *odata, const int *idata) {
			int *dev_idata;
			int *dev_odata;
			//malloc memory
			cudaMalloc((void**)&dev_idata, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_idata failed!");
			cudaMalloc((void**)&dev_odata, n * sizeof(int));
			checkCUDAError("cudaMalloc dev_odata failed!");
			//mempy
			cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);//host to device
			checkCUDAError("cudaMemcpy dev_idata failed!");
			cudaMemcpy(dev_odata, odata, sizeof(int) * n, cudaMemcpyHostToDevice);//host to device
			checkCUDAError("cudaMemcpy dev_odata failed!");

			thrust::device_ptr<int> dev_in(dev_idata);
			thrust::device_ptr<int> dev_out(dev_odata);

            timer().startGpuTimer();
            // TODO use `thrust::exclusive_scan`
            // example: for device_vectors dv_in and dv_out:
            //thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());
			thrust::exclusive_scan(dev_in, dev_in + n, dev_out);
            timer().endGpuTimer();

			cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);//get the result
			checkCUDAError("get odata failed!");

			cudaFree(dev_idata);
			cudaFree(dev_odata);
        }
    }
}
