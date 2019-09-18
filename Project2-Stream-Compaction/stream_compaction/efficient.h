#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

		void scan(int n, int *odata, const int *idata);

		void scanForRadix(int n, int *odata, const int *idata, int radixBlockSize);

        int compact(int n, int *odata, const int *idata);
    }
}
