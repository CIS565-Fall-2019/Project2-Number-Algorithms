#pragma once

#include "common.h"

namespace CharacterRecognition {
    Common::PerformanceTimer& timer();

    // TODO: implement required elements for MLP sections 1 and 2 here
    void initSimulation(int num_object, int hidden_num, int output_num, std::vector<float*> inputs, std::vector<float*> target_outputs);
    float sigmoid(float z);
    float sigmoid_prime(float z);
    void feed_forward(int n, int hidden_num, int output_num, int idata_idx, float* odata, float* input_weight_matrix, float* hidden_weight_matrix, float* hidden_bias_vec, float* output_bias_vec);
    void argmax(int output_num, float* curr_output);
    void SGD(int training_round, int hidden_size, float eta, float* input_weight_matrix, float* hidden_weight_matrix, float* hidden_bias_vec, float* output_bias_vec);
    void back_prop(float* cost, float* input_data, float* target_output_data, float* input_weight_matrix, float* hidden_weight_matrix, float* hidden_bias_vec, float* output_bias_vec, float* updated_input_weight, float* updated_hidden_weight, float* updated_hidden_bias, float* updated_output_bias);
    void Evaluate(int n, int hidden_num, int output_num, float* test_data, float* target_output, float* input_weight_matrix, float* hidden_weight_matrix, float* hidden_bias_vec, float* output_bias_vec);
    float compute_cost(int size, float* target_output_data, float* output_layer);
}
