#pragma once

#include <vector>
#include "common.h"

namespace CharacterRecognition {
    Common::PerformanceTimer& timer();
	class AffineLayer{
		float *W;
		float *b;
		int inputDim, outputDim;
		bool sigmoid;
		bool eval;
	public:
		AffineLayer(int idim, int odim);
		void forward(float *in, float *out, int num_samples);
		void backward(float *dout, float *dw, float *dx, float *db);
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
}
