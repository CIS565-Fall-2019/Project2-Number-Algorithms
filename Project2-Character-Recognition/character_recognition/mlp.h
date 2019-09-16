#pragma once

#include "common.h"
#include<cublas_v2.h>
#include<vector>
using namespace std;

namespace CharacterRecognition {
    Common::PerformanceTimer& timer();
	class NeuralNet {
		//The weight matrices
		vector<float*>w;
		//The bias vectors
		vector<float*>b;
		// z^[l] = W^[l]a^[l-1] + b^[l]
		vector<float*>z;
		//a^[l] = g^[l](z^[l])
		// a^[0] is the input
		vector<float*>a;
		// The derivatives of weight matrices
		vector<float*>dw;
		// The derivatives of bias matrices
		vector<float*>db;
		// The derivatives of z
		vector<float*>dz;
		// The derivatives of a
		vector<float*>da;
		// The layer sizes
		vector<int>layer_sizes;
	public:
		//The constructor takes in the input size and the number of output classes
		NeuralNet(int input_size, int classes, vector<int>layers);
		// The forward function will do the forward propogation
		float* forward(float *input);
		// The backward function is responsible for calculating gradients
		// and applying the update formula
		// w^[l] = w^[l] - alpha*dw^[l]
		//void backward(float* y);
		// Calculates the loss given predicted value of y and actual value of y and stores it in loss;
		//void calculateLoss(float *ypred, float* y, float* loss);
		// The descrutor will free up the memory;
		~NeuralNet();
	};

}
