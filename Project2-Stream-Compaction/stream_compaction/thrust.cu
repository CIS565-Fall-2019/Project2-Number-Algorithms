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

			thrust::device_vector<int> in(idata, idata+n);
			thrust::device_vector<int> out(n);

			//thrust::exclusive_scan(in.begin(), in.end(), out.begin());
            // TODO use `thrust::exclusive_scan`
            // example: for device_vectors dv_in and dv_out:
			timer().startGpuTimer();
            thrust::exclusive_scan(in.begin(),in.end(),out.begin());
			timer().endGpuTimer();
			thrust::copy(out.begin(), out.end(), odata);
        }
    }
}
