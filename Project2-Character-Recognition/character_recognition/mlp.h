#pragma once

#include <vector>
#include "common.h"

namespace CharacterRecognition {
    Common::PerformanceTimer& timer();
	class AffineLayer{
		float *dev_in;
		float *dev_out;
		float *W;
		float *b;
		int numSamples;
		int inputDim, outputDim;
		bool sigmoid;
		bool eval;
		bool doneFwd;
	public:
		AffineLayer(int idim, int odim, int ns);
		float* forward(float *in, int num_samples);
		float* backward(float *dout, float lr);
		void setEval(bool state);
		void setSigmoid(bool state);
		float softmax_loss(float *pred, float *target, float *dout, int numSamples, int outputDim);
		void cpu_softmax(float *pred, int numSamples, int outputDim);
		float cpu_crossEntropy(float *pred, float *target, int numSamples, int outputDim, float* dout);
		char* getType();
	};

	class FCN {
		std::vector<AffineLayer*> layers;
	public:
		FCN(int inputDim, int outputDim, int numHiddenLayers, int *hiddenDims);
		void forward(float *input, float *ouput, bool eval);
	};
    // TODO: implement required elements for MLP sections 1 and 2 here
	void XORTest();
}
