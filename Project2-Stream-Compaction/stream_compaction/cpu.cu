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
            // TODO

			if (n == 0) {
				timer().endCpuTimer();
				return;
			}
			// odata[0] = idata[0];  // Inclusive Scan
			odata[0] = 0;	// Exclusive Scan
			for (int i = 1; i < n; i++) {
				odata[i] = odata[i - 1] + idata[i-1];
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
            // TODO
			int num = 0;
			for (int i = 0; i < n; i++) {
				if (idata[i] == 0) continue;
				odata[num++] = idata[i];
			}
	        timer().endCpuTimer();
            return num;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
	        // TODO
			int* scanResult = (int*)malloc(n * sizeof(int));

			// Scan
			scanResult[0] = 0;
			for (int i = 1; i < n; i++) {
				scanResult[i] = scanResult[i - 1] + (idata[i-1]?1:0);
			}
			
			int num = 0;
			for (int i = 0; i < n; i++) {
				// Only write on element if idata has a 1
				if (idata[i]) {
					odata[scanResult[i]] = idata[i];
					num++;
				}
			}
			free(scanResult);
	        timer().endCpuTimer();
            return num;
        }
    }
}
