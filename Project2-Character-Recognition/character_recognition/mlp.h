#pragma once

#include "common.h"
#include<cublas_v2.h>
#include<vector>
using namespace std;

namespace CharacterRecognition {
    Common::PerformanceTimer& timer();
	class NeuralNet {
		//The weight matrices
		vector<double*>w;
		//The bias vectors
		vector<double*>b;
		// z^[l] = W^[l]a^[l-1] + b^[l]
		vector<double*>z;
		//a^[l] = g^[l](z^[l])
		// a^[0] is the input
		vector<double*>a;
		// The derivatives of weight matrices
		vector<double*>dw;
		// The derivatives of bias matrices
		vector<double*>db;
		// The derivatives of z
		vector<double*>dz;
		// The derivatives of a
		vector<double*>da;
		// The layer sizes
		vector<int>layer_sizes;
		// The activation differentiation vectors
		vector<double*>ghat;
	public:
		//The constructor takes in the input size and the number of output classes
		NeuralNet(int input_size, int classes, vector<int>layers);
		// The forward function will do the forward propogation
		double* forward(double *input);
		// The backward function is responsible for calculating gradients
		// and applying the update formula
		// w^[l] = w^[l] - alpha*dw^[l]
		void backward(double* y);
		// Calculates the loss given predicted value of y and actual value of y and stores it in loss;
		double calculateLoss(double *ypred, double* y, int classes);
		// The descrutor will free up the memory;
		~NeuralNet();
	};

}
