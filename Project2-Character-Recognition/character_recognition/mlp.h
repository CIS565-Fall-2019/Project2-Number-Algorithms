#pragma once

#include "common.h"

namespace CharacterRecognition {
    Common::PerformanceTimer& timer();
	// TODO: implement required elements for MLP sections 1 and 2 here

	// MLP section 1 and 2 Character Reader
	void initMLP(int N, int P, int iDim, int hDim, int oDim);
	void readData(int N, int P, int iDim, int hDim, int oDim);
	void trainMLP(int N, int P, int iDim, int hDim, int oDim);
	void testMLP(int N, int P, int iDim, int hDim, int oDim);

}

