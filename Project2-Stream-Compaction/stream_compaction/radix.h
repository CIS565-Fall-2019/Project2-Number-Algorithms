#pragma once

#include "common.h"

namespace Sorting {
	namespace Radix {
		StreamCompaction::Common::PerformanceTimer& timer();
		void sort(unsigned long long int n, long long *odata, long long *idata, int max_value);
	}
}
