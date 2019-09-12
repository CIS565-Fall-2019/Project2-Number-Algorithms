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
			bool started_timer = true;
			try {
				timer().startCpuTimer();
			} catch (const std::exception& e) {
				started_timer = false;
			}

	        // TODO
			odata[0] = 0;
			for (int i = 1; i < n; i++)
				odata[i] = odata[i - 1] + idata[i-1];
			if(started_timer)
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
			int j = 0;
			for (int i = 0; i < n; i++)
				if (idata[i] != 0)
					odata[j++] = idata[i];
	        timer().endCpuTimer();
            return j;
        }

		void printArray(int n, int *a, bool abridged = false) {
			printf("    [ ");
			for (int i = 0; i < n; i++) {
				if (abridged && i + 2 == 15 && n > 16) {
					i = n - 2;
					printf("... ");
				}
				printf("%3d ", a[i]);
			}
			printf("]\n");
		}

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
	        // TODO
			int *binary_idata = new int[n];
			int *final_index = new int[n];
			for (int i = 0; i < n; i++)
				binary_idata[i] = idata[i] > 0 ? 1 : 0;
			scan(n, final_index, binary_idata);
			for (int i = 0; i < n; i++)
				if (binary_idata[i] == 1)
					odata[final_index[i]] = idata[i];
	        timer().endCpuTimer();
            return final_index[n-1] + (idata[n-1] > 0);
        }
    }
	
}
