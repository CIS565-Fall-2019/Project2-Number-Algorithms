#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/complex.h>
#include "common.h"
#include "mlp.h"

namespace CharacterRecognition {
    using Common::PerformanceTimer;
    PerformanceTimer& timer()
    {
        static PerformanceTimer timer;
        return timer;
    }

    //three buffers
    float* device_input;
    float* device_weight_matrix;
    float* device_hidden;
    float* device_ji_buffer;
    float* device_output;

    //directly used in class function
    int INPUT_SIZE = 0;
    int HIDDEN_SIZE = 0;
    int OUTPUT_SIZE = 0;

    std::vector<float*> array_inputs;
    std::vector<float*> array_target_outputs;
        
    void initSimulation(int num_object, int hidden_num, int output_num, std::vector<float*> inputs, std::vector<float*> target_outputs)
    {
        CharacterRecognition::INPUT_SIZE = num_object;
        CharacterRecognition::HIDDEN_SIZE = hidden_num;
        CharacterRecognition::OUTPUT_SIZE = output_num;

        array_inputs.assign(inputs.begin(), inputs.end());
        array_target_outputs.assign(target_outputs.begin(), target_outputs.end());
    }

    float sigmoid(float z)
    {
        return 1 / (1 + std::pow(exp(1.0), -z));
    }

    float sigmoid_prime(float z)
    {
        return sigmoid(z) * (1 - sigmoid(z));
    }

    //kernel function
    void cost_derivative(int n, float* target_output, float* actual_output, float* error_vec)
    {
        for (int i = 0; i < n; ++i)
        {
            error_vec[i] = actual_output[i] - target_output[i];
        }
    }

    __global__ void kernCostDerivative(int n, float* target_output, float* actual_output, float* error_vec)
    {
        int index = threadIdx.x + (blockIdx.x * blockDim.x);
        if (index >= n)
        {
            return;
        }

        error_vec[index] = actual_output[index] - target_output[index];
    }

    // TODO: __global__
    __global__ void kernInputMultWeight(int max_bound, int n, float* idata, float* weight_matrix, float* odata)
    {
        int index = threadIdx.x + (blockIdx.x * blockDim.x);
        if (index >= max_bound)
        {
            return;
        }
        int row = index;

        float sum = 0;
        for (int col = 0; col < n; ++col)
        {
            int idx = row * n + col;
            float w = weight_matrix[idx];
            float input = idata[col];
            sum += w * input; //weight's row is fixed, but col different, similarly ,the weight corresponds to what element in idata
        }

        odata[row] = sum;
    }
    //kernel?
    void argmax(int output_num, float* curr_output)
    {
        int max_idx = -1;
        float max = 0;
        for (int i = 0; i < output_num; ++i)
        {
            if (curr_output[i] >= max)
            {
                max_idx = i;
                max = curr_output[i];
            }
        }

        for (int i = 0; i < output_num; ++i)
        {
            if (i == max_idx) {
                curr_output[i] = 1;
            }
            else curr_output[i] = 0;
        }
    }

    void feed_forward(int n, int hidden_num, int output_num, int idata_idx, float* odata, float* input_weight_matrix, float* hidden_weight_matrix, float* hidden_bias_vec, float* output_bias_vec)
    {
        float* temp_hidden = new float[hidden_num];
        float* idata = array_inputs[idata_idx];
        //input to hidden
        for (int row = 0; row < hidden_num; ++row)
        {
            float sum = 0;
            for (int col = 0; col < n; ++col)
            {
                int idx = row * n + col;
                float w = input_weight_matrix[idx];
                float input = idata[col];
                sum += w * input;

            }
            temp_hidden[row] = sigmoid(sum + hidden_bias_vec[row]);
        }

        //test
        //std::cout << "the 2064 of element in this file is: " << idata[2064] << std::endl;
        //printFloatArray(hidden_num, temp_hidden, false);
        //from hidden to output
            //input to hidden
        for (int row = 0; row < output_num; ++row)
        {
            float sum = 0;
            for (int col = 0; col < hidden_num; ++col)
            {
                int idx = row * hidden_num + col;
                float w = hidden_weight_matrix[idx];
                float input = temp_hidden[col];
                sum += w * input;

            }
            odata[row] = sigmoid(sum + output_bias_vec[row]);
        }

        delete[] temp_hidden;
    }

    void back_prop(float* cost, float* input_data, float* target_output_data, float* input_weight_matrix, float* hidden_weight_matrix, float* hidden_bias_vec, float* output_bias_vec, float* updated_input_weight, float* updated_hidden_weight, float* updated_hidden_bias, float* updated_output_bias)
    {
        //generate intermediate hidden layer
        float* hidden_layer = new float[HIDDEN_SIZE];
        float* output_layer = new float[OUTPUT_SIZE];
        float* hidden_weighted_input = new float[HIDDEN_SIZE];
        float* output_weighted_input = new float[OUTPUT_SIZE];
        float* hidden_cost_error = new float[HIDDEN_SIZE];
        float* output_cost_error = new float[OUTPUT_SIZE];


        float* device_hidden_layer;
        float* device_output_layer;
        float* device_hidden_weighted_input;
        float* device_output_weighted_input;
        float* device_hidden_cost_error;
        float* device_output_cost_error;
        float* device_input_data;
        float* device_target_output_data;
        float* device_input_weight_mat;
        float* device_hidden_weight_mat;


        cudaMalloc((void**)&device_hidden_layer, HIDDEN_SIZE * sizeof(float));
        checkCUDAError("cudaMalloc device_hidden_layer failed!");
        cudaMalloc((void**)&device_output_layer, OUTPUT_SIZE * sizeof(float));
        checkCUDAError("cudaMalloc device_output_layer failed!");
        cudaMalloc((void**)&device_hidden_weighted_input, HIDDEN_SIZE * sizeof(float));
        checkCUDAError("cudaMalloc device_hidden_weighted_input failed!");
        cudaMalloc((void**)&device_output_weighted_input, OUTPUT_SIZE  * sizeof(float));
        checkCUDAError("cudaMalloc device_output_weighted_input failed!");
        cudaMalloc((void**)&device_hidden_cost_error, HIDDEN_SIZE * sizeof(float));
        checkCUDAError("cudaMalloc device_hidden_cost_error failed!");
        cudaMalloc((void**)&device_output_cost_error, OUTPUT_SIZE * sizeof(float));
        checkCUDAError("cudaMalloc device_output_cost_error failed!");
        cudaMalloc((void**)&device_input_data, INPUT_SIZE * sizeof(float));
        checkCUDAError("cudaMalloc device_input_data failed!");
        cudaMalloc((void**)&device_target_output_data, OUTPUT_SIZE * sizeof(float));
        checkCUDAError("cudaMalloc device_target_output_data failed!");
        cudaMalloc((void**)&device_input_weight_mat, HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
        checkCUDAError("cudaMalloc device_hidden_weight_mat failed!");
        cudaMalloc((void**)&device_hidden_weight_mat, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
        checkCUDAError("cudaMalloc device_output_weight_mat failed!");

        //memcpy input and target output
        cudaMemcpy(device_input_data, input_data, INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(device_target_output_data, target_output_data, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(device_input_weight_mat, input_weight_matrix, HIDDEN_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(device_hidden_weight_mat, hidden_weight_matrix, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);

        cudaMemset(device_hidden_layer, 0, HIDDEN_SIZE * sizeof(float));
        cudaMemset(device_output_layer, 0, OUTPUT_SIZE * sizeof(float));
        cudaMemset(device_hidden_weighted_input, 0, HIDDEN_SIZE * sizeof(float));
        cudaMemset(device_output_weighted_input, 0, OUTPUT_SIZE * sizeof(float));
        cudaMemset(device_hidden_cost_error, 0, HIDDEN_SIZE * sizeof(float));
        cudaMemset(device_output_cost_error, 0, OUTPUT_SIZE * sizeof(float));

        int gridSize = (INPUT_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
        dim3 normalBlocksPerGrid(gridSize);
        int hiddenBufferSize = (HIDDEN_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
        dim3 hiddenBufferBlocksPerGrid(hiddenBufferSize);
        //int jiBufferSize = (output_num * n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        //dim3 jiBufferBlocksPerGrid(jiBufferSize);
        int outputSize = (OUTPUT_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
        dim3 outputBlocksPerGrid(outputSize);
        dim3 threadsPerBlock(BLOCK_SIZE);

        //initialize
        std::fill(hidden_layer, hidden_layer + HIDDEN_SIZE, 0);
        std::fill(output_layer, output_layer + OUTPUT_SIZE, 0);
        std::fill(hidden_weighted_input, hidden_weighted_input + HIDDEN_SIZE, 0);
        std::fill(output_weighted_input, output_weighted_input + OUTPUT_SIZE, 0);
        std::fill(hidden_cost_error, hidden_cost_error + HIDDEN_SIZE, 0);
        std::fill(output_cost_error, output_cost_error + OUTPUT_SIZE, 0);

        timer().startGpuTimer();

        kernInputMultWeight << < hiddenBufferBlocksPerGrid, threadsPerBlock >> > (HIDDEN_SIZE, INPUT_SIZE, device_input_data, device_input_weight_mat, device_hidden_layer);

        timer().endGpuTimer();
        //need to apply the activate function on each element, currently each element is only the sum.
        //copy to a temp array and copmute sequentially for now
        cudaMemcpy(hidden_layer, device_hidden_layer, HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
        checkCUDAError("cudaMemcpy device_hidden_layer to hidden_layer failed!");

        //compute activation function
        for (int i = 0; i < HIDDEN_SIZE; ++i)
        {
            hidden_weighted_input[i] = hidden_layer[i] + hidden_bias_vec[i];
            hidden_layer[i] = sigmoid(hidden_weighted_input[i]);
        }

        //copy back to hidden
        cudaMemcpy(device_hidden_layer, hidden_layer, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
        checkCUDAError("cudaMemcpy hidden_layer to device_hidden_layer failed!");

        timer().startGpuTimer();
        //do we use the same weight? -- in XOR example, it is not
        kernInputMultWeight << < outputBlocksPerGrid, threadsPerBlock >> > (OUTPUT_SIZE, HIDDEN_SIZE, device_hidden_layer, device_hidden_weight_mat, device_output_layer);
        timer().endGpuTimer();

        //how to calculate the error and affect the next weights?  -- how to get expcted result? -- read in?
        cudaMemcpy(output_layer, device_output_layer, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
        checkCUDAError("cudaMemcpy device_output to odata failed!");

        for (int i = 0; i < OUTPUT_SIZE; ++i)
        {
            output_weighted_input[i] = output_layer[i] + output_bias_vec[i];
            output_layer[i] = sigmoid(output_weighted_input[i]);
        }

        //output cost here
        *cost += compute_cost(OUTPUT_SIZE, target_output_data, output_layer);

        //Get the cost derivative from the output layer result and target output data
        //test
        cudaMemcpy(device_output_layer, output_layer, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
        kernCostDerivative <<< outputBlocksPerGrid, threadsPerBlock >> > (OUTPUT_SIZE, device_target_output_data, device_output_layer, device_output_cost_error);
        cudaMemcpy(output_cost_error, device_output_cost_error, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
        //add the sigmoid prime to it
        for (int row = 0; row < OUTPUT_SIZE; ++row)
        {
            output_cost_error[row] *= sigmoid_prime(output_weighted_input[row]);
            //assign to updated weights and bias for output
            updated_output_bias[row] += output_cost_error[row];
            for (int col = 0; col < HIDDEN_SIZE; ++col)
            {
                int mat_idx = row * HIDDEN_SIZE + col;
                updated_hidden_weight[mat_idx] += hidden_layer[col] * output_cost_error[row];
            }
        }

        //compute the hidden_cost_error by output_cost_error
        //use transpose index
        for (int row = 0; row < HIDDEN_SIZE; ++row)
        {
            for (int col = 0; col < OUTPUT_SIZE; ++col)
            {
                int mat_idx = row * OUTPUT_SIZE + col;
                hidden_cost_error[row] += hidden_weight_matrix[mat_idx] * output_cost_error[col];
            }

            //apply sigmoid prime after we compute the derivative
            hidden_cost_error[row] *= sigmoid_prime(hidden_weighted_input[row]);
            //assign to updated matrix and vec
            updated_hidden_bias[row] += hidden_cost_error[row];
            for (int mat_col = 0; mat_col < INPUT_SIZE; ++mat_col)
            {
                int mat_idx = row * INPUT_SIZE + mat_col;
                updated_input_weight[mat_idx] += input_data[mat_col] * hidden_cost_error[row];
            }
        }

        cudaFree(device_hidden_layer);
        cudaFree(device_output_layer);
        cudaFree(device_hidden_weighted_input);
        cudaFree(device_output_weighted_input);
        cudaFree(device_hidden_cost_error);
        cudaFree(device_output_cost_error);
        cudaFree(device_input_data);
        cudaFree(device_target_output_data);

        delete[] hidden_layer;
        delete[] output_layer;
        delete[] hidden_weighted_input;
        delete[] output_weighted_input;
        delete[] hidden_cost_error;
        delete[] output_cost_error;
    }

    //SGD
    //we use training data as array_input and array_target_output, same as test data -- probably all size information should be inputs
    void SGD(int training_round, int hidden_size, float eta, float* input_weight_matrix, float* hidden_weight_matrix, float* hidden_bias_vec, float* output_bias_vec)
    {
        //init necessary buffers
        float* updated_input_weight = new float[INPUT_SIZE * hidden_size];
        float* updated_hidden_weight = new float[OUTPUT_SIZE * hidden_size];
        float* updated_hidden_bias = new float[hidden_size];
        float* updated_output_bias = new float[OUTPUT_SIZE];

        //need cuda malloc to init

        for (int round = 0; round < training_round; ++round)
        {
            //zero out all components for new round of change
            std::fill(updated_input_weight, updated_input_weight + INPUT_SIZE * hidden_size, 0);
            std::fill(updated_hidden_weight, updated_hidden_weight + OUTPUT_SIZE * hidden_size, 0);
            std::fill(updated_hidden_bias, updated_hidden_bias + hidden_size, 0);
            std::fill(updated_output_bias, updated_output_bias + OUTPUT_SIZE, 0);
            //memcpy(updated_hidden_bias, hidden_bias_vec, HIDDEN_SIZE * sizeof(float));

            std::cout << "Round " << round << ":" << std::endl;
            float cost = 0;
            for (int input_index = 0; input_index < OUTPUT_SIZE; ++input_index)
            {
                //update each element in the buffer arrays directly in back_prop
                back_prop(&cost, array_inputs[input_index], array_target_outputs[input_index], input_weight_matrix, hidden_weight_matrix, hidden_bias_vec, output_bias_vec, updated_input_weight, updated_hidden_weight, updated_hidden_bias, updated_output_bias);
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

            for (int hidden_bias_index = 0; hidden_bias_index < hidden_size; ++hidden_bias_index)
            {
                hidden_bias_vec[hidden_bias_index] -= (eta / OUTPUT_SIZE) * updated_hidden_bias[hidden_bias_index];
            }

            for (int output_bias_index = 0; output_bias_index < OUTPUT_SIZE; ++output_bias_index)
            {
                output_bias_vec[output_bias_index] -= (eta / OUTPUT_SIZE) * updated_output_bias[output_bias_index];
            }

            //output cost
            cost /= OUTPUT_SIZE;
            std::cout << "The cost of current data is: " << cost << std::endl;
        }

        std::cout << "we train " << training_round << " rounds" << std::endl;
        Evaluate(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, nullptr, nullptr, input_weight_matrix, hidden_weight_matrix, hidden_bias_vec, output_bias_vec);

        delete[] updated_input_weight;
        delete[] updated_hidden_weight;
        delete[] updated_hidden_bias;
        delete[] updated_output_bias;
    }

    float compute_cost(int size, float* target_output_data, float* output_layer)
    {
        float sum = 0;
        for (int i = 0; i < size; ++i)
        {
            sum += std::pow(target_output_data[i] - output_layer[i], 2);
        }

        sum /= 2;
        //std::cout << "The cost of current data is: " << sum << std::endl;
        return sum;
    }

    void Evaluate(int n, int hidden_num, int output_num, float* test_data, float* target_output, float* input_weight_matrix, float* hidden_weight_matrix, float* hidden_bias_vec, float* output_bias_vec)
    {
        int sum = 0;
        float* curr_output = new float[output_num];
        //unused (test_data, target_output) --> will use later
        for (int i = 0; i < output_num; ++i)
        {
            //zero out output
            std::fill(curr_output, curr_output + output_num, 0);
            //for (int j = 0; j < output_num; ++j)
            //{
            //    std::cout << "curr_output[" << j << "] is " << curr_output[j] << std::endl;
            //}

            feed_forward(n, hidden_num, output_num, i, curr_output, input_weight_matrix, hidden_weight_matrix, hidden_bias_vec, output_bias_vec);

            argmax(output_num, curr_output);
            bool same = true;
            for (int j = 0; j < output_num; ++j)
            {
                if (curr_output[j] != array_target_outputs[i][j])
                {
                    same = false;
                }
            }

            for (int j = 0; j < output_num; ++j)
            {
                if (curr_output[j] == 1)
                {
                    std::cout << "curr_output[" << j << "] is " << curr_output[j] << std::endl;
                }
            }

            if (same) sum++;

        }


        std::cout << "Result: " << sum << " / " << output_num << std::endl;

        delete[] curr_output;
    }

    //the same as inputMultWeight, except the max bound is different, here it is constrained by output_num
    __global__ void kernHiddenMultWeight(int max_bound, int n, float* idata, float* weight_matrix, float* odata)
    {
        int index = threadIdx.x + (blockIdx.x * blockDim.x);
        if (index >= max_bound)
        {
            return;
        }

        int row = index;

        float sum = 0;
        for (int col = 0; col < n; ++col)
        {
            int idx = row * n + col;
            sum += weight_matrix[idx] * idata[col]; //weight's row is fixed, but col different, similarly ,the weight corresponds to what element in idata
        }

        odata[row] = sum;
    }

}
