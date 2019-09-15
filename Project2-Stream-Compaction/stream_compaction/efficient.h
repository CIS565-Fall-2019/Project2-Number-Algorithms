#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(unsigned long long int n, int *odata, const int *idata);

		unsigned long long int compact(unsigned long long int n, int *odata, const int *idata);
    }
	namespace SharedMemory {
		StreamCompaction::Common::PerformanceTimer& timer();

		void scan(unsigned long long int n, int *odata, int *idata);
		void dev_scan(unsigned long long int n, int *odata, int *idata);
	}
}
