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
			bool newTimer = true;
			if (timer().getCpuTimerStarted()) {
				newTimer = false;
			}
			if (newTimer) {
				timer().startCpuTimer();
			}
            // TODO
			if (n > 0) {
				odata[0] = 0;
				for (int k = 1; k < n; ++k) {
					odata[k] = odata[k - 1] + idata[k - 1];
				}
			}
			if (newTimer) {
				timer().endCpuTimer();
			}
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
            // TODO
			int counter = 0;
			for (int k = 0; k < n; ++k) {
				int currVal = idata[k];
				if (currVal != 0) {
					odata[counter] = currVal;
					counter++;
				}
			}
	        timer().endCpuTimer();
            return counter;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
	        // TODO
			int *tempArray = new int[n];
			for (int k = 0; k < n; ++k) {
				tempArray[k] = (int) idata[k] != 0;
			}
			int counter = 0;
			int *scanResult = new int[n];
			scan(n, scanResult, tempArray);
			for (int k = 0; k < n; ++k) {
				if (tempArray[k]) {
					int index = scanResult[k];
					odata[index] = idata[k];
					counter++;
				}
			}


			delete[] scanResult;
			delete[] tempArray;
	        timer().endCpuTimer();
            return counter;
        }
    }
}
