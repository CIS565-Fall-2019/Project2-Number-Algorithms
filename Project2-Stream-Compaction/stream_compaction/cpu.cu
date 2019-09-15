#include <cstdio>
#include "cpu.h"
#include <iostream>
#include "common.h"
// function (in this file) assume zeroed memory
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
        void scan(unsigned long long int n, long long *odata, const long long *idata, const bool time/* = true*/) {
	        (time == true)?timer().startCpuTimer():0;
			unsigned long long int running_total = 0; // assumes only positive values
			for (unsigned long long i = 0; i < n-1; i++) { // n-1 because we want exclusive not inclusive so last element isn't added
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
		unsigned long long int compactWithoutScan(unsigned long long int n, long long *odata, const long long *idata) {
	        timer().startCpuTimer();
			unsigned long long int count = 0;
			unsigned long long int odata_pos = 0;
			for (unsigned long long i = 0; i < n; i++)
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
		unsigned long long int compactWithScan(unsigned long long int n, long long *odata, const long long *idata) {
	        timer().startCpuTimer();
	        // Step 1, mark each cell with 1/0 if it has a element or not (super easy to parallelize)
			long long* scan_data = new long long[n]();
			long long* mask = new long long[n]();
			for (unsigned long long i = 0; i < n; i++) {
				if (idata[i])
					mask[i] = 1;
				else
					mask[i] = 0;
			}
			// scan the mask array (can be done in parallel by using a balanced binary tree)
			scan(n, scan_data, mask, false);
			// Scatter array (go to each position and copy the value) (super easy to parallelize)
			for (unsigned long long i = 0; i < n - 1; i++) {
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
