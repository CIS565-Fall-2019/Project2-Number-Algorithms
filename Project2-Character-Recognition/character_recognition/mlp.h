#pragma once

#include "common.h"
#include <cublas_v2.h>

namespace CharacterRecognition {
    Common::PerformanceTimer& timer();

    // TODO: implement required elements for MLP sections 1 and 2 here
	/*void forward();
	void backward();
	void train();
	void test();*/
	void testMatrixMultiply();
	void train(float *X, float *y, int sizeData = 10205, const int hiddenNodes = 256, const int numLabels = 52, const int numData = 52);
	//void mult(cublasHandle_t* handle, const float* A, const float* B, float* C, int m, int k, int n);
}
