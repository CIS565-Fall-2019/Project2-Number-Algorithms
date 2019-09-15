#pragma once

#include "common.h"

namespace Sorting {
	namespace Radix {
		StreamCompaction::Common::PerformanceTimer& timer();
		void sort(unsigned long long int n, int *odata, int *idata, int max_value);
	}
}
