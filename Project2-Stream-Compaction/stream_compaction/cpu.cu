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
			if(n > 0){
				int sum = 0;
			   for (int i = 0; i < n; i++) {
				   odata[i] = sum;
				   sum += idata[i];
			    }
			}
	        timer().endCpuTimer();
        }

		void scan_notimer(int n, int *odata, const int *idata) {
			if(n > 0){
				int sum = 0;
			   for (int i = 0; i < n; i++) {
				   odata[i] = sum;
				   sum += idata[i];
			    }
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
			int count = 0;
			for (int i = 0; i < n; i++) {
				if (idata[i] != 0) {
					odata[count++] = idata[i];
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
			if (n < 1) {
				return 0;
			}
			int* tmp = new int [n];
			 
			for (int i = 0; i < n; i++) {
				tmp[i] = (idata[i] == 0) ? 0 : 1;
			}
			int * scan_o = new int[n];

			scan_notimer(n, scan_o, tmp);
			int scan_counter = 0;
			for (int i = 1; i < n; i++) {
				if (scan_o[i] != scan_o[i - 1]) {
					odata[scan_counter++] = idata[i - 1];
				}
			}

	        timer().endCpuTimer();
			delete[] scan_o;
			delete[] tmp;
			return scan_counter;
        }
    }
}
