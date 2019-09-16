#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(unsigned long long int n, long long *odata, const long long *idata);

		unsigned long long int compact(unsigned long long int n, long long *odata, const long long *idata);
    }
	namespace SharedMemory {
		StreamCompaction::Common::PerformanceTimer& timer();
		void scan(unsigned long long int n, long long *odata, long long *idata);
		void dev_scan(unsigned long long int n, long long *odata, long long *idata);
		unsigned long long int compact(unsigned long long int n, long long *odata, const long long *idata);
	}
}
