#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Naive {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(unsigned long int n, int *odata, const int *idata);
    }
}
