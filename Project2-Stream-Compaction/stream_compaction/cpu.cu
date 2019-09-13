#include <cstdio>
#include "cpu.h"
#include <iostream>
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
        void scan(int n, int *odata, const int *idata, const bool time/* = true*/) {
	        (time == true)?timer().startCpuTimer():0;
			int running_total = 0;
			for (int i = 0; i < n-1; i++) { // n-1 because we want exclusive not inclusive so last element isn't added
				running_total += idata[i]; // n-1 adds for n size 
				odata[i+1] = running_total;
			}
			(time == true) ? timer().endCpuTimer() : 0;
			
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
			int count = 0;
			int odata_pos = 0;
			for (int i = 0; i < n; i++)
				if (idata[i] != 0) {
					count++;
					odata[++odata_pos] = idata[i];
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
	        // Step 1, mark each cell with 1/0 if it has a element or not (super easy to parallelize)
			int* scan_data = new int[n]();
			int* mask = new int[n]();
			for (int i = 0; i < n; i++) {
				if (idata[i])
					mask[i] = 1;
				else
					mask[i] = 0;
			}
			// scan the mask array (can be done in parallel by using a balanced binary tree)
			scan(n, scan_data, mask, false);
			// Scatter array (go to each position and copy the value) (super easy to parallelize)
			for (int i = 0; i < n - 1; i++) {
				if (idata[i])
					odata[scan_data[i] + 1] = idata[i]; 
			}
			int res = scan_data[n - 1];
			delete[] scan_data;
			delete[] mask;
	        timer().endCpuTimer();
            return res;
        }
    }
}
