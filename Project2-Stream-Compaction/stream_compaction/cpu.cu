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
			// idata: orig int array, odata: output int array, n is len(int array)
	        timer().startCpuTimer();
			odata[0] = 0;
			for (int i = 1; i < n; i++) {
				odata[i] = odata[i - 1] + idata[i-1]; //n-1 adds
			}
	        timer().endCpuTimer();
        }

        void scan_notimer(int n, int *odata, const int *idata) {
			// idata: orig int array, odata: output int array, n is len(int array)
			odata[0] = 0;
			for (int i = 1; i < n; i++) {
				odata[i] = odata[i - 1] + idata[i-1]; //n-1 adds
			}
        }


        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
			// idata: orig int array, odata: output int array, n is len(int array)
	        timer().startCpuTimer();
			int num_nonzeros = 0;
			for (int i = 0; i < n; i++) {
				int elt_i = idata[i];
				if (elt_i != 0) {
					odata[num_nonzeros] = elt_i;
					++num_nonzeros;
				}
			}
	        timer().endCpuTimer();
            return num_nonzeros;
        }

		void computeTemporaryArray(int n, int *tempArray, const int *idata) {
			//Temporary array copies zeros & sets nonzeros to 1
			for (int i = 0; i < n; i++) {
				if (idata[i] != 0) {
					tempArray[i] = 1;
				}
				else {
					tempArray[i] = 0;
				}
			}
		}

		int scatter(int n, int *odata, const int *idata, const int *tempArray) {
			//odata now contains the scan result
			int elt_i, shouldInclude, newIdx;
			int count = 0;
			for (int i = 0; i < n; i++) {
				shouldInclude = tempArray[i];
				elt_i = idata[i];
				if (shouldInclude) {
					newIdx = odata[i];
					odata[newIdx] = elt_i;
					++count;
				}
			}
			return count;
		}

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
			// idata: orig int array, odata: output int array, n is len(int array)
	        timer().startCpuTimer();

			//1: Malloc & Compute Temporary Array
			int *tempArray = new int[n];
			computeTemporaryArray(n, tempArray, idata);

			//2: Exclusive Scan on tempArray
			scan_notimer(n, odata, tempArray);

			//3: Scatter
			int newlen = scatter(n, odata, idata, tempArray);
	        timer().endCpuTimer();
            return newlen;
        }
    }
}