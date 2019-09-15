#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
	    StreamCompaction::Common::PerformanceTimer& timer();

        void scan(unsigned long long int n, int *odata, const int *idata, const bool time = true);

		unsigned long long int compactWithoutScan(unsigned long long int n, int *odata, const int *idata);

		unsigned long long int compactWithScan(unsigned long long int n, int *odata, const int *idata);
    }
}
