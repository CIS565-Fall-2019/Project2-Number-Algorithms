/**
 * @file      main.cpp
 * @brief     Stream compaction test program
 * @authors   Kai Ninomiya
 * @date      2015
 * @copyright University of Pennsylvania
 */

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
const int HIDDEN_SIZE = 10;
//float* input = new float[INPUT_SIZE];
//float* output = new float[OUTPUT_SIZE];
std::vector<float*> array_inputs;
std::vector<float*> array_target_outputs;

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
        std::cout << input_file_prefix << "." << std::endl;
        int count = 0;
        while (!myfile.eof())
        {
            std::string intermediate_array;
            std::vector <std::string> tokens;
            std::getline(myfile, intermediate_array, '\n');
            if (count == 0)
            {
                std::cout << intermediate_array << std::endl;
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
                int count = 0;
                //the first is a space, clean it
                std::getline(tokens, token, ' ');
                while (std::getline(tokens, token, ' '))
                {
                    input[count] = std::stoi(token);
                    count++;
                }
                if (count != INPUT_SIZE)
                {
                    std::cout << "count != INPUT_SIZE, something wrong" << std::endl;
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
//def evaluate(self, test_data) :
//    """Return the number of test inputs for which the neural
//    network outputs the correct result.Note that the neural
//    network's output is assumed to be the index of whichever
//    neuron in the final layer has the highest activation."""
//    test_results = [(np.argmax(self.feedforward(x)), y)
//    for (x, y) in test_data]
//    return sum(int(x == y) for (x, y) in test_results)

//def cost_derivative(self, output_activations, y) :
//"""Return the vector of partial derivatives \partial C_x /
//\partial a for the output activations."""
 //return (output_activations - y)


float sigmoid(float z)
{
    return 1 / (1 + std::pow(exp(1.0), -z));
}

float sigmoid_prime(float z)
{
    return sigmoid(z) * (1 - sigmoid(z));
}

void cost_derivative(int n, float* target_output, float* actual_output, float* error_vec)
{
    for (int i = 0; i < n; ++i)
    {
        error_vec[i] = actual_output[i] - target_output[i];
    }
}

void argmax(int output_num, float* curr_output)
{
    int max_idx = -1;
    float max = 0;
    for (int i = 0; i < output_num; ++i)
    {
        if (curr_output[output_num] >= max)
        {
            max_idx = i;
            max = curr_output[output_num];
        }
    }
}


void feed_forward(int n, int hidden_num, int output_num, float* idata, float* odata, float* input_weight_matrix, float* hidden_weight_matrix, float* input_bias_vec, float* hidden_bias_vec)
{
    float* temp_hidden = new float[hidden_num];

    //input to hidden
    for (int row = 0; row < hidden_num; ++row)
    {
        float sum = 0;
        for (int col = 0; col < n; ++col)
        {
            int idx = row * n + col;
            float w = input_weight_matrix[idx];
            float input = idata[col];
            sum += w * input + input_bias_vec[col];

        }
        temp_hidden[row] = sigmoid(sum);
    }

    //from hidden to output
        //input to hidden
    for (int row = 0; row < output_num; ++row)
    {
        float sum = 0;
        for (int col = 0; col < hidden_num; ++col)
        {
            int idx = row * n + col;
            float w = hidden_weight_matrix[idx];
            float input = temp_hidden[col];
            sum += w * input + hidden_bias_vec[col];

        }
        odata[row] = sigmoid(sum);
    }
}

//SGD
//we use training data as array_input and array_target_output, same as test data -- probably all size information should be inputs
void SGD(int training_round, int hidden_size, float eta, float* input_weight_matrix, float* hidden_weight_matrix, float* input_bias_vec, float* hidden_bias_vec)
{
    //init necessary buffers
    float* updated_input_weight = new float[INPUT_SIZE * hidden_size];
    float* updated_hidden_weight = new float[OUTPUT_SIZE * hidden_size];
    float* updated_input_bias = new float[INPUT_SIZE];
    float* updated_hidden_bias = new float[hidden_size];

    for (int round = 0; round < training_round; ++round)
    {
        //zero out all components for new round of change
        std::fill(updated_input_weight, updated_input_weight + INPUT_SIZE * hidden_size, 0);
        std::fill(updated_hidden_weight, updated_hidden_weight + OUTPUT_SIZE * hidden_size, 0);
        std::fill(updated_input_bias, updated_input_bias + INPUT_SIZE, 0);
        std::fill(updated_hidden_bias, updated_hidden_bias + hidden_size, 0);
        //memcpy(updated_hidden_bias, hidden_bias_vec, HIDDEN_SIZE * sizeof(float));

        for (int input_index = 0; input_index < OUTPUT_SIZE; ++input_index)
        {
            //update each element in the buffer arrays directly in back_prop
            back_prop(array_inputs[input_index], array_target_outputs[input_index], updated_input_weight, updated_hidden_weight, updated_input_bias, updated_hidden_bias, hidden_size);
        }

        //update the weights and bias after each round
        for (int input_weight_index = 0; input_weight_index < INPUT_SIZE * hidden_size; ++input_weight_index)
        {
            input_weight_matrix[input_weight_index] -= (eta / OUTPUT_SIZE) * updated_input_weight[input_weight_index];
        }

        for (int hidden_weight_index = 0; hidden_weight_index < OUTPUT_SIZE * hidden_size; ++hidden_weight_index)
        {
            hidden_weight_matrix[hidden_weight_index] -= (eta / OUTPUT_SIZE) * updated_hidden_weight[hidden_weight_index];
        }

        for (int input_bias_index = 0; input_bias_index < INPUT_SIZE; ++input_bias_index)
        {
            input_bias_vec[input_bias_index] -= (eta / OUTPUT_SIZE) * updated_input_bias[input_bias_index];
        }

        for (int hidden_bias_index = 0; hidden_bias_index < hidden_size; ++hidden_bias_index)
        {
            hidden_bias_vec[hidden_bias_index] -= (eta / OUTPUT_SIZE) * updated_hidden_bias[hidden_bias_index];
        }
    }

    free(updated_input_weight);
    free(updated_hidden_weight);
    free(updated_input_bias);
    free(updated_hidden_bias);
}

void back_prop(float* input_data, float* target_output_data, float* updated_input_weight, float* updated_hidden_weight, float* updated_input_bias, float* updated_hidden_bias, int hidden_size)
{

}


void Evaluate(int n, int hidden_num, int output_num, float* test_data, float* target_output, float* input_weight_matrix, float* hidden_weight_matrix, float* input_bias_vec, float* hidden_bias_vec)
{
    int sum = 0;
    float* curr_output = new float[output_num];
    //unused (test_data, target_output) --> will use later
    for (int i = 0; i < output_num; ++i)
    {
        //zero out output
        std::fill(curr_output, curr_output + output_num, 0);
        feed_forward(n, hidden_num, output_num, array_inputs[i], curr_output, input_weight_matrix, hidden_weight_matrix, input_bias_vec, hidden_bias_vec);

        argmax(curr_output);
        if (curr_output[i] == array_target_outputs[i][i])
        {
            //hack, we know that that element in target_output should be 1, so if both are one, it is correct
            sum++;
        }
    }
}



int main(int argc, char* argv[]) {

    read_in_inputs();
    //the data will be stored in array_inputs and array_outputs.

    //init input_weights and hidden weights
    float *input_weight_matrix = new float[INPUT_SIZE * HIDDEN_SIZE];
    float *hidden_weight_matrix = new float[OUTPUT_SIZE * HIDDEN_SIZE];
    init_weight_matrix(INPUT_SIZE, HIDDEN_SIZE, input_weight_matrix);  // Leave a 0 at the end to test that edge case
    init_weight_matrix(OUTPUT_SIZE, HIDDEN_SIZE, hidden_weight_matrix);
    float *input_bias_vec = new float[INPUT_SIZE];
    float *hidden_bias_vec = new float[HIDDEN_SIZE];
    std::fill(input_bias_vec, input_bias_vec + INPUT_SIZE, 0);
    std::fill(hidden_bias_vec, hidden_bias_vec + HIDDEN_SIZE, 0);

    float target_error = 0.0003;
    float total_error = 0.2; //will be modified
    float target_val = 1; //tested on XOR example -- should this be a 
    //first generate the weight matrix to store all weight values for each element, in this homework, it should be a n * 2 matrix, such that n is the number of input

    for (int i = 0; i < OUTPUT_SIZE; ++i)
    {
        printFloatArray(INPUT_SIZE, array_inputs[i], true);
        printFloatArray(OUTPUT_SIZE, array_target_outputs[i], true);
        std::cout << std::endl;
    }
    //printFloatArray(INPUT_SIZE * HIDDEN_SIZE, weight_matrix, false);

    //get input from files?
    //CharacterRecognition::MLP_calculation(INPUT_SIZE, OUTPUT_SIZE, input, weight_matrix, output);
    //printElapsedTime(CharacterRecognition::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    //printFloatArray(OUTPUT_SIZE, output, true);

    //assume we get the correct output, we should calculate the error and apply bp algorithm back to weight

    //how do we get the target value? -- from file?

    //understand bp algorithm
    //how to make it keep running? -- while loop within a certain threshold
    delete[] weight_matrix;
 //   // Scan tests

 //   printf("\n");
 //   printf("****************\n");
 //   printf("** SCAN TESTS **\n");
 //   printf("****************\n");

 //   genArray(SIZE - 1, a, 50);  // Leave a 0 at the end to test that edge case
 //   a[SIZE - 1] = 0;
 //   printArray(SIZE, a, true);

 //   // initialize b using StreamCompaction::CPU::scan you implement
 //   // We use b for further comparison. Make sure your StreamCompaction::CPU::scan is correct.
 //   // At first all cases passed because b && c are all zeroes.
 //   zeroArray(SIZE, b);
 //   printDesc("cpu scan, power-of-two");
 //   StreamCompaction::CPU::scan(SIZE, b, a);
 //   printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
 //   printArray(SIZE, b, true);

 //   zeroArray(SIZE, c);
 //   printDesc("cpu scan, non-power-of-two");
 //   StreamCompaction::CPU::scan(NPOT, c, a);
 //   printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
 //   printArray(NPOT, b, true);
 //   printCmpResult(NPOT, b, c);

 //   zeroArray(SIZE, c);
 //   printDesc("naive scan, power-of-two");
 //   StreamCompaction::Naive::scan(SIZE, c, a);
 //   printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
 //   //printArray(SIZE, c, true);
 //   printCmpResult(SIZE, b, c);

	///* For bug-finding only: Array of 1s to help find bugs in stream compaction or scan
	//onesArray(SIZE, c);
	//printDesc("1s array for finding bugs");
	//StreamCompaction::Naive::scan(SIZE, c, a);
	//printArray(SIZE, c, true); */

 //   zeroArray(SIZE, c);
 //   printDesc("naive scan, non-power-of-two");
 //   StreamCompaction::Naive::scan(NPOT, c, a);
 //   printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
 //   //printArray(SIZE, c, true);
 //   printCmpResult(NPOT, b, c);

 //   zeroArray(SIZE, c);
 //   printDesc("work-efficient scan, power-of-two");
 //   StreamCompaction::Efficient::scan(SIZE, c, a);
 //   printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
 //   //printArray(SIZE, c, true);
 //   printCmpResult(SIZE, b, c);

 //   zeroArray(SIZE, c);
 //   printDesc("work-efficient scan, non-power-of-two");
 //   StreamCompaction::Efficient::scan(NPOT, c, a);
 //   printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
 //   //printArray(NPOT, c, true);
 //   printCmpResult(NPOT, b, c);

 //   zeroArray(SIZE, c);
 //   printDesc("thrust scan, power-of-two");
 //   StreamCompaction::Thrust::scan(SIZE, c, a);
 //   printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
 //   //printArray(SIZE, c, true);
 //   printCmpResult(SIZE, b, c);

 //   zeroArray(SIZE, c);
 //   printDesc("thrust scan, non-power-of-two");
 //   StreamCompaction::Thrust::scan(NPOT, c, a);
 //   printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
 //   //printArray(NPOT, c, true);
 //   printCmpResult(NPOT, b, c);

 //   printf("\n");
 //   printf("*****************************\n");
 //   printf("** STREAM COMPACTION TESTS **\n");
 //   printf("*****************************\n");

 //   // Compaction tests

 //   genArray(SIZE - 1, a, 4);  // Leave a 0 at the end to test that edge case
 //   a[SIZE - 1] = 0;
 //   printArray(SIZE, a, true);

 //   int count, expectedCount, expectedNPOT;

 //   // initialize b using StreamCompaction::CPU::compactWithoutScan you implement
 //   // We use b for further comparison. Make sure your StreamCompaction::CPU::compactWithoutScan is correct.
 //   zeroArray(SIZE, b);
 //   printDesc("cpu compact without scan, power-of-two");
 //   count = StreamCompaction::CPU::compactWithoutScan(SIZE, b, a);
 //   printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
 //   expectedCount = count;
 //   printArray(count, b, true);
 //   printCmpLenResult(count, expectedCount, b, b);

 //   zeroArray(SIZE, c);
 //   printDesc("cpu compact without scan, non-power-of-two");
 //   count = StreamCompaction::CPU::compactWithoutScan(NPOT, c, a);
 //   printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
 //   expectedNPOT = count;
 //   printArray(count, c, true);
 //   printCmpLenResult(count, expectedNPOT, b, c);

 //   zeroArray(SIZE, c);
 //   printDesc("cpu compact with scan");
 //   count = StreamCompaction::CPU::compactWithScan(SIZE, c, a);
 //   printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
 //   printArray(count, c, true);
 //   printCmpLenResult(count, expectedCount, b, c);

 //   zeroArray(SIZE, c);
 //   printDesc("work-efficient compact, power-of-two");
 //   count = StreamCompaction::Efficient::compact(SIZE, c, a);
 //   printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
 //   //printArray(count, c, true);
 //   printCmpLenResult(count, expectedCount, b, c);

 //   zeroArray(SIZE, c);
 //   printDesc("work-efficient compact, non-power-of-two");
 //   count = StreamCompaction::Efficient::compact(NPOT, c, a);
 //   printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
 //   //printArray(count, c, true);
 //   printCmpLenResult(count, expectedNPOT, b, c);

 //   system("pause"); // stop Win32 console from closing on exit
	//delete[] a;
	//delete[] b;
	//delete[] c;
}
