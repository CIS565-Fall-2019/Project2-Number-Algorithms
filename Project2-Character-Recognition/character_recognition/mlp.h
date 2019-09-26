#pragma once
#include "common.h"
#include <cublas_v2.h>


namespace CharacterRecognition {
	Common::PerformanceTimer& timer();

	class mlp {
	public:

		mlp(int n_input, int n_hidden, int n_output);
		~mlp();

		void initWeights(float *wkj, float *wji);
		void initRandom();
		void train(float *x, float *y, int n, int epoch);
		void predict(float *x, float *y, int n);
		float getError() const {
			return error;
		}

	private:
		void forward();
		void backProp();
		void loss();
		void update();

		int input_size;
		int hidden_size;
		int output_size;

		float *wkj, *wji;
		float *gwkj, *gwji;
		int num;
		float* dev_x;
		float* dev_hidden;
		float* dev_hidden_sm;
		float* dev_y;
		float* dev_y_sm;
		float* dev_target;
		float error;

		cublasHandle_t handle;
	};
}
