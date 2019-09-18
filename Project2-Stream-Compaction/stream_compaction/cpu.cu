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
			for (int i = 0; i < n; i++) {
				int sum = 0;
				for (int j = 0; j < i; j++) {
					sum += idata[j];
				}
				odata[i] = sum;
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
			int j = 0;
			for (int i = 0; i < n; i++) {
				int toPut = idata[i];
				if (toPut != 0) {
					odata[j] = toPut;
					j++;
				}
			}
	        timer().endCpuTimer();
            return j;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
			int *mappedArray = new int[n];
			for (int i = 0; i < n; i++) {
				mappedArray[i] = 0;
			}
			for (int i = 0; i < n; i++) {
				if (idata[i] == 0) mappedArray[i] = 0;
				else mappedArray[i] = 1;
			}
			int *scannedArray = new int[n];
			for (int i = 0; i < n; i++) {
				scannedArray[i] = 0;
			}
			scanNoTimer(n, scannedArray, mappedArray);
			int count = 0;
			for (int i = 0; i < n; i++) {
				int toPut = idata[i];
				int indexToPut = scannedArray[i];
				if (toPut != 0) {
					odata[indexToPut] = toPut;
					count++;
				}
			}
	        timer().endCpuTimer();
			delete[] mappedArray;
			delete[] scannedArray;
            return count;
        }

		void scanNoTimer(int n, int *odata, const int *idata) {
			for (int i = 0; i < n; i++) {
				int sum = 0;
				for (int j = 0; j < i; j++) {
					sum += idata[j];
				}
				odata[i] = sum;
			}
		}
    }
}
