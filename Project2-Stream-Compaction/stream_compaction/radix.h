#pragma once

#include "common.h"
#include <stream_compaction/efficient.h>

namespace StreamCompaction {
    namespace Radix {
        StreamCompaction::Common::PerformanceTimer& timer();

        void radix_sort(int n, int bits_num, int *odata, const int *idata);
    }
}
