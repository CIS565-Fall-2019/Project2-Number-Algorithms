#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
	    StreamCompaction::Common::PerformanceTimer& timer();

        void scan(unsigned long long int n, long long *odata, const long long *idata, const bool time = true);

		unsigned long long int compactWithoutScan(unsigned long long int n, long long *odata, const long long *idata);

		unsigned long long int compactWithScan(unsigned long long int n, long long *odata, const long long *idata);
    }
}
