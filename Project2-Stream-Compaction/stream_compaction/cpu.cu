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
			bool exception = true;
			try {
				timer().startCpuTimer();
				exception = false;
			}
			catch (const std::exception& e) {
				exception = true;
			}

            // TODO
			odata[0] = 0;
			for (int i = 1; i < n; i++) {
				odata[i] = odata[i - 1] + idata[i-1];
			}

			try {
				if (exception == false) {
					timer().endCpuTimer();
				}
			}
			catch (const std::exception& e) {

			}
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {

			bool exception = true;
			try {
				timer().startCpuTimer();
				exception = false;
			}
			catch (const std::exception& e) {
				exception = true;
			}

            // TODO
			int compactedIndex = 0;
			for (int i = 0; i < n; i++) {
				if (idata[i] != 0) {
					odata[compactedIndex++] = idata[i];
				}
			}

			try {
				if (exception == false) {
					timer().endCpuTimer();
				}
			}
			catch (const std::exception& e) {

			}
            return compactedIndex;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
			bool exception = true;
			try {
				timer().startCpuTimer();
				exception = false;
			}
			catch (const std::exception& e) {
				exception = true;
			}

			int *binaryMap = new int[n];
			int *scannedBinaryArray = new int[n];
			int numOfElements = 0;
	        // TODO
			for (int i = 0; i < n; i++) {
				binaryMap[i] = idata[i] == 0 ? 0 : 1;
			}

			scan(n, scannedBinaryArray, binaryMap);

			for (int i = 0; i < n; i++) {
				if (binaryMap[i] == 1) {
					odata[scannedBinaryArray[i]] = idata[i];
					numOfElements++;
				}
			}

			try {
				if (exception == false) {
					timer().endCpuTimer();
				}
			}
			catch (const std::exception& e) {

			}
            return numOfElements;
        }
    }
}
