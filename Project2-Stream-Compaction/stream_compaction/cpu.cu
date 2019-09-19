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
			for (int i = 1; i < n; i++) {
				odata[i] = idata[i - 1] + odata[i - 1];
			}

	        timer().endCpuTimer();
        }

		/**
		 * CPU scan (prefix sum) with no timers
		 */
		void scanNoTimer(int n, int *odata, const int *idata) {
			odata[0] = 0;
			for (int i = 1; i < n; i++) {
				odata[i] = idata[i - 1] + odata[i - 1];
			}
		}

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();

			// copy over non-zero values to odata
			int currIndex = 0;
			for (int i = 0; i < n; i++) {
				if (idata[i] != 0) {
					odata[currIndex] = idata[i];
					currIndex++;
				}
			}

	        timer().endCpuTimer();
            return currIndex;
        }

		/**
		 * CPU scatter
		 *
		 * @returns the number of elements remaining after compaction.
		 */
		int scatter(int n, int *odata, const int *idata, const int *bools, const int *scanOutput) {
			for (int i = 0; i < n; i++) {
				if (bools[i] == 1) {
					odata[scanOutput[i]] = idata[i];
				}
			}
			return scanOutput[n - 1];
		}

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();

			int *bools = new int[n];
			int *scanResult = new int[n];

			// map input to binary array 
			for (int i = 0; i < n; i++) {
				bools[i] = idata[i] == 0 ? 0 : 1;
			}

			// scan binary array
			scanNoTimer(n, scanResult, bools);

			// scatter
			int outSize = scatter(n, odata, idata, bools, scanResult);
			delete(bools);
			delete(scanResult);

	        timer().endCpuTimer();

            return outSize;
        }
    }
}
