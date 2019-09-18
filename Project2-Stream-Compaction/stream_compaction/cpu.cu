#include <cstdio>
#include<iostream>
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
			int timer_flag = 0;
			try {
				timer().startCpuTimer();
			}
			catch (...) {
				timer_flag = 1;
			}
			std::cout << "Timer had already started" << std::endl;
			odata[0] = 0;
			for (int i = 1; i < n; i++) {
				odata[i] = odata[i - 1] + idata[i - 1];
			}
			if(timer_flag == 0)
				timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
			int index_out = 0;
			for (int i = 0; i < n; i++) {
				if (idata[i] != 0)
					odata[index_out++] = idata[i];
			}
	        timer().endCpuTimer();
            return index_out;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
			int *mask = new int[n];
			int *mask_scan = new int[n];
			memset(mask, 0, sizeof(mask));
			memset(mask_scan, 0, sizeof(mask_scan));
			for (int i = 0; i < n; i++) {
				if (idata[i] == 0)
					mask[i] = 0;
				else
					mask[i] = 1;
			}
			scan(n, mask_scan, mask);
			int max_index = 0;
			for (int i = 0; i < n; i++) {
				if (mask[i] == 1) {
					odata[mask_scan[i]] = idata[i];
					max_index = mask_scan[i];
				}
			}
			timer().endCpuTimer();
            return max_index+1;
        }
    }
}
