#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
	    StreamCompaction::Common::PerformanceTimer& timer();

        void scan(unsigned long int n, int *odata, const int *idata, const bool time = true);

		unsigned long int compactWithoutScan(unsigned long int n, int *odata, const int *idata);

		unsigned long int compactWithScan(unsigned long int n, int *odata, const int *idata);
    }
}
