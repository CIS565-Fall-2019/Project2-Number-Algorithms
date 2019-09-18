#pragma once

#include "common.h"

namespace CharacterRecognition {
		Common::PerformanceTimer& timer();

		// TODO: implement required elements for MLP sections 1 and 2 here

		void build_network(int data_count, int num_feature, int num_class,
			int hid_size, float ler, float loss_thre);

		void train(float* input, float* real, int train_time);

		void test(float* test_input, int test_times, int type);
}
