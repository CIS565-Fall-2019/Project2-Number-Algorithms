#pragma once

#include <vector>
#include "common.h"

namespace CharacterRecognition {
    Common::PerformanceTimer& timer();
	class AffineLayer{
		float *dev_in;
		float *dev_out;
		float *dev_dout;
		float *dev_doutLinear;
		float *dev_din;
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
