#include <cstdio>
#include "cpu.h"

#include "common.h"
#include <iostream>

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
            for (int idx = 0; idx < n; ++idx)
            {
                if (idx == 0) odata[idx] = 0;
                else odata[idx] = odata[idx - 1] + idata[idx - 1];
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
            int counter = 0;
            for (int idx = 0; idx < n; ++idx)
            {
                int curr_val = idata[idx];
                if (curr_val != 0)
                {
                 odata[counter] = curr_val;
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
            //do we need to allocate a new array?
            //how to allocate in cpu
            int *bool_array = new int[n]();
            int *scattered_array = new int[n]();
            for (int idx = 0; idx < n; ++idx)
            {
                if (idata[idx] == 0) bool_array[idx] = 0;
                else bool_array[idx] = 1;
            }
            //scan -- if directly call scan will cause cpu timer error
            for (int idx = 0; idx < n; ++idx)
            {
                if (idx == 0) scattered_array[idx] = 0;
                else scattered_array[idx] = scattered_array[idx - 1] + bool_array[idx - 1];
            }

            for (int idx = 0; idx < n; ++idx)
            {
                int curr_odata_idx = scattered_array[idx];
                if (bool_array[idx] != 0)
                {
                    odata[curr_odata_idx] = idata[idx];
                }
            
            }
	        // TODO
	        timer().endCpuTimer();
            int returned_val = scattered_array[n - 1] + bool_array[n-1]; //index start from 0
            free(bool_array);
            free(scattered_array);
            return returned_val;
        }
    }
}
