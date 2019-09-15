#pragma once

#include "common.h"
#include<cublas_v2.h>

namespace CharacterRecognition {
    Common::PerformanceTimer& timer();
	class NeuralNet {
		float *inp, *theta1, *bias1, *theta2, *bias2, *theta3, *bias3, *g1, *a1, *g2, *a2, *g3, *output; cublasHandle_t handle;
		float *dtheta1, *dtheta2, *dtheta3, *dbias1, *dbias2, *dbias3;
		float *dyhatg3, *da2g2, *da1g1;

	public:
		NeuralNet();
		float* forward(float *input);
		void mmul(const float* A, const float* B, float* C, const int m, const int k, const int n);
		~NeuralNet();
	};

    // TODO: implement required elements for MLP sections 1 and 2 here
}
