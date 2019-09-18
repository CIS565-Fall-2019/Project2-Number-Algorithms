#pragma once

#include "common.h"

namespace StreamCompaction {
	namespace RadixSort {
		void sort(int n, int *odata, const int *idata, int blockSize);
	}
}
