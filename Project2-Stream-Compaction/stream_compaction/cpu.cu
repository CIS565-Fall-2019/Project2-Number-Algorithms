#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
	        static PerformanceTimer timer;
	        return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
			odata[0] = 0;
			for (int j = 1; j < n; j++) {
				odata[j] = odata[j - 1] + idata[j - 1];
			}
	        timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
			int count = 0;
			for (int j = 0; j < n; j++) {
				if (idata[j] != 0) {
					odata[count] = idata[j];
					count++;
				}
			}
	        timer().endCpuTimer();
            return count;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
			int *valid = new int[n];
			for (int i = 0; i < n; i++) {
				if (idata[i] == 0) {
					valid[i] = 0;
				}
				else {
					valid[i] = 1;
				}
			}
			int *index = new int[n];
			index[0] = 0;
			for (int j = 1; j < n; j++) {
				index[j] = index[j - 1] + valid[j - 1];
			}
			for (int i = 0; i < n; i++) {
				if (valid[i] == 1) {
					odata[index[i]] = idata[i];
				}
			}
			int count = index[n - 1];
			delete valid, index;
	        timer().endCpuTimer();
            return count;
        }
    }
}
