/**
 * @file      main.cpp
 * @brief     Stream compaction test program
 * @authors   Kai Ninomiya
 * @date      2015
 * @copyright University of Pennsylvania
 */

#include <curand.h>
#include <conio.h>
#include <iostream>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <string.h>
#include <algorithm>
#include <character_recognition/mlp.h>
#include <character_recognition/common.h>
#include "testing_helpers.hpp"

#define LAYER 2
#define INPUT_NUM 52

const int INPUT_SIZE = 10201; // feel free to change the size of array
const int OUTPUT_SIZE = 52;
const int HIDDEN_SIZE = 30;
//float* input = new float[INPUT_SIZE];
//float* output = new float[OUTPUT_SIZE];
std::vector<float*> array_inputs;
std::vector<float*> array_target_outputs;

void Evaluate(int n, int hidden_num, int output_num, float* test_data, float* target_output, float* input_weight_matrix, float* hidden_weight_matrix, float* hidden_bias_vec, float* output_bias_vec);

void read_in_inputs()
{
    std::string input_file_postfix = "info.txt";
    std::string input_file_folder = "..\\data-set\\";
    for (int i = 1; i <= OUTPUT_SIZE; ++i)
    {
        //init the buffer we want to store the input data
        float* input = new float[INPUT_SIZE];
        float* output = new float[OUTPUT_SIZE];

        std::string input_file_prefix = i < 10 ? "0" + std::to_string(i) : std::to_string(i);
        std::string input_file = input_file_folder + input_file_prefix + input_file_postfix;

        std::ifstream myfile(input_file);
        if (!myfile)
        {
            std::cout << "Error opening input_file" << std::endl;
            std::cout << input_file << std::endl;
            system("pause");
            return;
        }
        //std::cout << input_file_prefix << "." << std::endl;
        int count = 0;
        while (!myfile.eof())
        {
            std::string intermediate_array;
            std::vector <std::string> tokens;
            std::getline(myfile, intermediate_array, '\n');
            if (count == 0)
            {
                //std::cout << intermediate_array << std::endl;
                int output_result = std::stoi(intermediate_array);
                for (int i = 1; i <= OUTPUT_SIZE; ++i)
                {
                    if (i == output_result) output[i-1] = 1;
                    else output[i-1] = 0;
                }
            }
            else if (count == 2)
            {
                std::stringstream tokens(intermediate_array);
                std::string token;
                // Tokenizing w.r.t. space ' ' 
                int array_count = 0;
                //the first is a space, clean it
                std::getline(tokens, token, ' ');
                while (std::getline(tokens, token, ' '))
                {
                    input[array_count] = std::stoi(token) / 255; //normalize
                    array_count++;
                }
                if (array_count != INPUT_SIZE)
                {
                    std::cout << "array_count != INPUT_SIZE, something wrong" << std::endl;
                }
            }
            count++;
        }
        //store the input array into vector
        array_inputs.push_back(input);
        array_target_outputs.push_back(output);
    }
}

//first do in cpu version

//utility

//float sigmoid(float z)
//{
//    return 1 / (1 + std::pow(exp(1.0), -z));
//}
//
//float sigmoid_prime(float z)
//{
//    return sigmoid(z) * (1 - sigmoid(z));
//}
//
//void cost_derivative(int n, float* target_output, float* actual_output, float* error_vec)
//{
//    for (int i = 0; i < n; ++i)
//    {
//        error_vec[i] = actual_output[i] - target_output[i];
//    }
//}
//
//void argmax(int output_num, float* curr_output)
//{
//    int max_idx = -1;
//    float max = 0;
//    for (int i = 0; i < output_num; ++i)
//    {
//        if (curr_output[i] >= max)
//        {
//            max_idx = i;
//            max = curr_output[i];
//        }
//    }
//
//    for (int i = 0; i < output_num; ++i)
//    {
//        if (i == max_idx) {
//            curr_output[i] = 1;
//        }
//        else curr_output[i] = 0;
//    }
//}
//
//float compute_cost(int size, float* target_output_data, float* output_layer)
//{
//    float sum = 0;
//    for (int i = 0; i < size; ++i)
//    {
//        sum += std::pow(target_output_data[i] - output_layer[i], 2);
//    }
//
//    sum /= 2;
//    //std::cout << "The cost of current data is: " << sum << std::endl;
//    return sum;
//}
//
//void feed_forward(int n, int hidden_num, int output_num, int idata_idx, float* odata, float* input_weight_matrix, float* hidden_weight_matrix, float* hidden_bias_vec, float* output_bias_vec)
//{
//    float* temp_hidden = new float[hidden_num];
//    float* idata = array_inputs[idata_idx];
//    //input to hidden
//    for (int row = 0; row < hidden_num; ++row)
//    {
//        float sum = 0;
//        for (int col = 0; col < n; ++col)
//        {
//            int idx = row * n + col;
//            float w = input_weight_matrix[idx];
//            float input = idata[col];
//            sum += w * input;
//
//        }
//        temp_hidden[row] = sigmoid(sum + hidden_bias_vec[row]);
//    }
//
//    //test
//    //std::cout << "the 2064 of element in this file is: " << idata[2064] << std::endl;
//    //printFloatArray(hidden_num, temp_hidden, false);
//    //from hidden to output
//        //input to hidden
//    for (int row = 0; row < output_num; ++row)
//    {
//        float sum = 0;
//        for (int col = 0; col < hidden_num; ++col)
//        {
//            int idx = row * hidden_num + col;
//            float w = hidden_weight_matrix[idx];
//            float input = temp_hidden[col];
//            sum += w * input;
//
//        }
//        odata[row] = sigmoid(sum + output_bias_vec[row]);
//    }
//
//    delete[] temp_hidden;
//}
//
//void back_prop(int hidden_num, float* cost, float* input_data, float* target_output_data, float* input_weight_matrix, float* hidden_weight_matrix, float* hidden_bias_vec, float* output_bias_vec, float* updated_input_weight, float* updated_hidden_weight, float* updated_hidden_bias, float* updated_output_bias)
//{
//    //generate intermediate hidden layer
//    float* hidden_layer = new float[hidden_num];
//    float* output_layer = new float[OUTPUT_SIZE];
//    float* hidden_weighted_input = new float[hidden_num];
//    float* output_weighted_input = new float[OUTPUT_SIZE];
//    float* output_cost_error = new float[OUTPUT_SIZE];
//    float* hidden_cost_error = new float[hidden_num];
//
//    //initialize
//    std::fill(hidden_layer, hidden_layer + hidden_num, 0);
//    std::fill(output_layer, output_layer + OUTPUT_SIZE, 0);
//    std::fill(hidden_weighted_input, hidden_weighted_input + hidden_num, 0);
//    std::fill(output_weighted_input, output_weighted_input + OUTPUT_SIZE, 0);
//    std::fill(hidden_cost_error, hidden_cost_error + hidden_num, 0);
//    std::fill(output_cost_error, output_cost_error + OUTPUT_SIZE, 0);
//
//    //feedfoward
//    for (int row = 0; row < hidden_num; ++row)
//    {
//        float sum = 0;
//        for (int col = 0; col < INPUT_SIZE; ++col)
//        {
//            int idx = row * INPUT_SIZE + col;
//            float w = input_weight_matrix[idx];
//            float input = input_data[col];
//            sum += w * input;
//
//        }
//        hidden_weighted_input[row] = sum + hidden_bias_vec[row];
//        hidden_layer[row] = sigmoid(sum + hidden_bias_vec[row]);
//    }
//
//    for (int row = 0; row < OUTPUT_SIZE; ++row)
//    {
//        float sum = 0;
//        for (int col = 0; col < hidden_num; ++col)
//        {
//            int idx = row * hidden_num + col;
//            float w = hidden_weight_matrix[idx];
//            float input = hidden_layer[col];
//            sum += w * input;
//
//        }
//        output_weighted_input[row] = sum + output_bias_vec[row];
//        output_layer[row] = sigmoid(sum + output_bias_vec[row]);
//    }
//	
//    //output cost here
//    *cost += compute_cost(OUTPUT_SIZE, target_output_data, output_layer);
//
//    //Get the cost derivative from the output layer result and target output data
//    cost_derivative(OUTPUT_SIZE, target_output_data, output_layer, output_cost_error);
//    //add the sigmoid prime to it
//    for (int row = 0; row < OUTPUT_SIZE; ++row)
//    {
//        output_cost_error[row] *= sigmoid_prime(output_weighted_input[row]);
//        //assign to updated weights and bias for output
//        updated_output_bias[row] += output_cost_error[row];
//        for (int col = 0; col < hidden_num; ++col)
//        {
//            int mat_idx = row * hidden_num + col;
//            updated_hidden_weight[mat_idx] += hidden_layer[col] * output_cost_error[row];
//        }
//    }
//
//    //compute the hidden_cost_error by output_cost_error
//    //use transpose index
//    for (int row = 0; row < hidden_num; ++row)
//    {
//        for (int col = 0; col < OUTPUT_SIZE; ++col)
//        {
//            int mat_idx = row * OUTPUT_SIZE + col;
//            hidden_cost_error[row] += hidden_weight_matrix[mat_idx] * output_cost_error[col];
//        }
//
//        //apply sigmoid prime after we compute the derivative
//        hidden_cost_error[row] *= sigmoid_prime(hidden_weighted_input[row]);
//        //assign to updated matrix and vec
//        updated_hidden_bias[row] += hidden_cost_error[row];
//        for (int mat_col = 0; mat_col < INPUT_SIZE; ++mat_col)
//        {
//            int mat_idx = row * INPUT_SIZE + mat_col;
//            updated_input_weight[mat_idx] += input_data[mat_col] * hidden_cost_error[row];
//        }
//    }
//}
//
////SGD
////we use training data as array_input and array_target_output, same as test data -- probably all size information should be inputs
//void SGD(int training_round, int hidden_size, float eta, float* input_weight_matrix, float* hidden_weight_matrix, float* hidden_bias_vec, float* output_bias_vec)
//{
//    //init necessary buffers
//    float* updated_input_weight = new float[INPUT_SIZE * hidden_size];
//    float* updated_hidden_weight = new float[OUTPUT_SIZE * hidden_size];
//    float* updated_hidden_bias = new float[hidden_size];
//    float* updated_output_bias = new float[OUTPUT_SIZE];
//
//    for (int round = 0; round < training_round; ++round)
//    {
//        //zero out all components for new round of change
//        std::fill(updated_input_weight, updated_input_weight + INPUT_SIZE * hidden_size, 0);
//        std::fill(updated_hidden_weight, updated_hidden_weight + OUTPUT_SIZE * hidden_size, 0);
//        std::fill(updated_hidden_bias, updated_hidden_bias + hidden_size, 0);
//        std::fill(updated_output_bias, updated_output_bias + OUTPUT_SIZE, 0);
//        //memcpy(updated_hidden_bias, hidden_bias_vec, HIDDEN_SIZE * sizeof(float));
//
//        std::cout << "Round " << round << ":" << std::endl;
//        float cost = 0;
//        for (int input_index = 0; input_index < OUTPUT_SIZE; ++input_index)
//        {
//            //update each element in the buffer arrays directly in back_prop
//            back_prop(hidden_size,&cost, array_inputs[input_index], array_target_outputs[input_index], input_weight_matrix, hidden_weight_matrix, hidden_bias_vec, output_bias_vec, updated_input_weight, updated_hidden_weight, updated_hidden_bias, updated_output_bias);
//        }
//
//        //update the weights and bias after each round
//        for (int input_weight_index = 0; input_weight_index < INPUT_SIZE * hidden_size; ++input_weight_index)
//        {
//            input_weight_matrix[input_weight_index] -= (eta / OUTPUT_SIZE) * updated_input_weight[input_weight_index];
//        }
//
//        for (int hidden_weight_index = 0; hidden_weight_index < OUTPUT_SIZE * hidden_size; ++hidden_weight_index)
//        {
//            hidden_weight_matrix[hidden_weight_index] -= (eta / OUTPUT_SIZE) * updated_hidden_weight[hidden_weight_index];
//        }
//
//        for (int hidden_bias_index = 0; hidden_bias_index < hidden_size; ++hidden_bias_index)
//        {
//            hidden_bias_vec[hidden_bias_index] -= (eta / OUTPUT_SIZE) * updated_hidden_bias[hidden_bias_index];
//        }
//
//        for (int output_bias_index = 0; output_bias_index < OUTPUT_SIZE; ++output_bias_index)
//        {
//            output_bias_vec[output_bias_index] -= (eta / OUTPUT_SIZE) * updated_output_bias[output_bias_index];
//        }
//
//        //output cost
//        cost /= OUTPUT_SIZE;
//        std::cout << "The cost of current data is: " << cost << std::endl;
//    }
//
//    std::cout << "we train " << training_round << " rounds" << std::endl;
//    Evaluate(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, nullptr, nullptr, input_weight_matrix, hidden_weight_matrix, hidden_bias_vec, output_bias_vec);
//
//    delete[] updated_input_weight;
//    delete[] updated_hidden_weight;
//    delete[] updated_hidden_bias;
//    delete[] updated_output_bias;
//}
//
bool check_data(int* file_idx, int* array_idx)
{
    for (int i = 0; i < OUTPUT_SIZE - 1; ++i)
    {
        for (int j = 0; j < INPUT_SIZE; ++j)
        {
            if (array_inputs[i][j] != array_inputs[i + 1][j])
            {
                *file_idx = i;
                *array_idx = j;
                return true;
            }
        }
    }
    return false;
}
//
//void Evaluate(int n, int hidden_num, int output_num, float* test_data, float* target_output, float* input_weight_matrix, float* hidden_weight_matrix, float* hidden_bias_vec, float* output_bias_vec)
//{
//    int sum = 0;
//    float* curr_output = new float[output_num];
//    //unused (test_data, target_output) --> will use later
//    for (int i = 0; i < output_num; ++i)
//    {
//        //zero out output
//        std::fill(curr_output, curr_output + output_num, 0);
//        //for (int j = 0; j < output_num; ++j)
//        //{
//        //    std::cout << "curr_output[" << j << "] is " << curr_output[j] << std::endl;
//        //}
//
//        feed_forward(n, hidden_num, output_num, i, curr_output, input_weight_matrix, hidden_weight_matrix, hidden_bias_vec, output_bias_vec);
//
//        argmax(output_num, curr_output);
//        bool same = true;
//        for (int j = 0; j < output_num; ++j)
//        {
//            if (curr_output[j] != array_target_outputs[i][j])
//            {
//                same = false;
//            }
//        }
//
//        for (int j = 0; j < output_num; ++j)
//        {
//            if (curr_output[j] == 1)
//            {
//                std::cout << "curr_output[" << j << "] is " << curr_output[j] << std::endl;
//            }
//        }
//        
//        if (same) sum++;
//
//    }
//
//
//    std::cout << "Result: " << sum << " / " << output_num << std::endl;
//
//    delete[] curr_output;
//}



int main(int argc, char* argv[]) {

    read_in_inputs();
    //the data will be stored in array_inputs and array_outputs.
    int file_idx = -1;
    int array_idx = -1;
    if (check_data(&file_idx, &array_idx))
    {
        std::cout << "the data has different in " << file_idx << " file " << ", array index is " << array_idx << std::endl;
    }

    //init input_weights and hidden weights
    float *input_weight_matrix = new float[INPUT_SIZE * HIDDEN_SIZE];
    float *hidden_weight_matrix = new float[OUTPUT_SIZE * HIDDEN_SIZE];
    init_weight_matrix(INPUT_SIZE, HIDDEN_SIZE, input_weight_matrix, -0.1f , 0.1f);  // Leave a 0 at the end to test that edge case
    init_weight_matrix(OUTPUT_SIZE, HIDDEN_SIZE, hidden_weight_matrix, -0.1f, 0.1f);
    float *hidden_bias_vec = new float[HIDDEN_SIZE];
    float *output_bias_vec = new float[OUTPUT_SIZE];
    std::fill(hidden_bias_vec, hidden_bias_vec + HIDDEN_SIZE, 0);
    std::fill(output_bias_vec, output_bias_vec + OUTPUT_SIZE, 0);

    CharacterRecognition::initSimulation(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, array_inputs, array_target_outputs);
    CharacterRecognition::SGD(50, HIDDEN_SIZE, 0.1f, input_weight_matrix, hidden_weight_matrix, hidden_bias_vec, output_bias_vec);

    //assume we get the correct output, we should calculate the error and apply bp algorithm back to weight

    //understand bp algorithm
    //how to make it keep running? -- while loop within a certain threshold
    delete[] input_weight_matrix;
    delete[] hidden_weight_matrix;
    delete[] hidden_bias_vec;
    delete[] output_bias_vec;

    //delete the whole array
    array_inputs.clear();
    array_target_outputs.clear();
}
