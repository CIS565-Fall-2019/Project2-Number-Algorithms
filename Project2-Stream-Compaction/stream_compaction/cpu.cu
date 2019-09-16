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

        void scan_implementation(int n, int *odata, const int *idata) {
          odata[0] = 0;
          for (int i = 1; i < n; i++) {
            odata[i] = odata[i - 1] + idata[i - 1];
          }
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
          // TODO
          scan_implementation(n, odata, idata);
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
          int cnt = 0;
          for (int i = 0; i < n; i++) {
            if (idata[i] != 0) {
              odata[cnt++] = idata[i];
            }
          }
	        timer().endCpuTimer();
          return cnt;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
	        // TODO
          int* isnonzero = (int*)malloc(n * sizeof(int));
          for (int i = 0; i < n; i++) {
            isnonzero[i] = (idata[i] != 0) ? 1 : 0;
          }

          // compute indices with an exclusive scan
          int* indices = (int*)malloc(n * sizeof(int));
          scan_implementation(n, indices, isnonzero);
          int n_compact = isnonzero[n - 1] ? indices[n - 1] + 1: indices[n - 1];

          // scatter
          for (int i = 0; i < n; i++) {
            if (isnonzero[i]) {
              odata[indices[i]] = idata[i];
            }
          }

          // free memory
          free(isnonzero);
          free(indices);
 
	        timer().endCpuTimer();
          return n_compact;
        }
    }
}
