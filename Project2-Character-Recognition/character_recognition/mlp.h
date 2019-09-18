#pragma once

#include "common.h"
#include <vector>
#include <math.h>

namespace CharacterRecognition {
    Common::PerformanceTimer& timer();
	class FullyConnectedLayer {
		float *weight = NULL;
		float *inputs = NULL;
		int inputDim;
		int batchDim;
		int outputDim;
		bool lastLayer;

	public:
		FullyConnectedLayer(int inputDim, int outputDim, int batchDim, bool lastLayer);
		void forward(float *input, float *output, bool test = false);
		void backward(float learningRate, float *incomingGradient, float *outgoingGradient);
		int getInputDim();
		int getOutputDim();
	};


	class MultiLayerPerceptron {

		std::vector<FullyConnectedLayer*> layers;
		int batchDim;
	public :
		MultiLayerPerceptron(int inputDim, int numHiddenLayers, int *hiddenDim, int outputDim, int batchDim);
		void forward(float *input, float *output, bool test = false);
		void backward(float *output, float *predicted, float learningRate);
		float loss(float *label, float *predicted);
	};
}
