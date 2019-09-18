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
        
    // TODO: __global__
    __global__ void kernInputMultWeight(int n, float* idata, float* weight_matrix, float* odata)
    {
        int index = threadIdx.x + (blockIdx.x * blockDim.x);
        if (index >= n)
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
        //https://stackoverflow.com/questions/10375680/using-stdvector-in-cuda-device-code -- have to use thrust library, but don't know how
        //float returned_val = 1 / (1 + thrust::pow((float)exp(1.0), -sum));
        //odata[row] = returned_val;
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

    //better to use something like thrust library and calculate the reduction  -- very slow
    //__global__ void kernKJBufferToHidden(int n, float* idata, float* odata)
    //{
    //    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    //    if (index >= n)
    //    {
    //        return;
    //    }

    //    float sum = 0;
    //    for (int idx = index * n; idx < (index + 1) * n; ++idx)
    //    {
    //        sum += idata[idx];
    //    }

    //    float returned_val = 1 / (1 + std::pow(exp(1.0), -sum));
    //    odata[index] = returned_val;
    //}

    ////similar to kj one, should have a better naming
    //__global__ void kernJIBufferToOutput(int n, float* idata, float* odata)
    //{
    //    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    //    if (index >= n)
    //    {
    //        return;
    //    }

    //    float sum = 0;
    //    for (int idx = index * n; idx < (index + 1) * n; ++idx)
    //    {
    //        sum += idata[idx];
    //    }

    //    float returned_val = 1 / (1 + std::pow(exp(1.0), -sum));
    //    odata[index] = returned_val;
    //}
    /**
        * Example of use case (follow how you did it in stream compaction)
        */
    /*void scan(int n, int *odata, const int *idata) {
        timer().startGpuTimer();
        // TODO
        timer().endGpuTimer();
    }
    */
    //void CharacterRecognition::initSimulation(int n, int output_num)
    //{
    //    cudaMalloc((void**)&device_input, n * sizeof(float));
    //    checkCUDAError("cudaMalloc device_input failed!");
    //    cudaMalloc((void**)&device_kj_buffer, n * n * sizeof(float));
    //    checkCUDAError("cudaMalloc device_kj_buffer failed!");
    //    cudaMalloc((void**)&device_hidden, n * sizeof(float));
    //    checkCUDAError("cudaMalloc device_hidden failed!");
    //    cudaMalloc((void**)&device_ji_buffer, output_num * n * sizeof(float));
    //    checkCUDAError("cudaMalloc device_ji_buffer failed!");
    //    cudaMalloc((void**)&device_output, output_num * sizeof(float));
    //    checkCUDAError("cudaMalloc device_output failed!");
    //}
	// TODO: implement required elements for MLP sections 1 and 2 here

    //feedforward functionality here, Sigmoid graduate descedent only need to add the number of group of training data and for loop them
    void MLP_calculation(int n, int output_num, float* idata, float* weight_matrix, float* odata)
    {
        //init buffer
        cudaMalloc((void**)&device_input, n * sizeof(float));
        checkCUDAError("cudaMalloc device_input failed!");
        cudaMalloc((void**)&device_weight_matrix, n * n * sizeof(float));
        checkCUDAError("cudaMalloc device_weight_matrix failed!");
        cudaMalloc((void**)&device_hidden, n * sizeof(float));
        checkCUDAError("cudaMalloc device_hidden failed!");
        cudaMalloc((void**)&device_ji_buffer, output_num * n * sizeof(float));
        checkCUDAError("cudaMalloc device_ji_buffer failed!");
        cudaMalloc((void**)&device_output, output_num * sizeof(float));
        checkCUDAError("cudaMalloc device_output failed!");

        //two temp main memory buffer
        float* temp_hidden = new float[n];

        cudaMemcpy(device_input, idata, n * sizeof(float), cudaMemcpyHostToDevice);
        checkCUDAError("cudaMemcpy idata to device_input failed!");

        cudaMemcpy(device_weight_matrix, weight_matrix, n * n * sizeof(float), cudaMemcpyHostToDevice);
        checkCUDAError("cudaMemcpy idata to device_weight_matrix failed!");

        int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        dim3 normalBlocksPerGrid(gridSize);
        //int kjBufferSize = (n * n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        //dim3 kjBufferBlocksPerGrid(kjBufferSize);
        //int jiBufferSize = (output_num * n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        //dim3 jiBufferBlocksPerGrid(jiBufferSize);
        int outputSize = (output_num + BLOCK_SIZE - 1) / BLOCK_SIZE;
        dim3 outputBlocksPerGrid(outputSize);
        dim3 threadsPerBlock(BLOCK_SIZE);


        //we first calcualte all the results into a matrix, and then read and sum each row in activate function and write in the hidden layer
        timer().startGpuTimer();

        kernInputMultWeight << < normalBlocksPerGrid, threadsPerBlock >> > (n, device_input, device_weight_matrix, device_hidden);

        timer().endGpuTimer();
        //need to apply the activate function on each element, currently each element is only the sum.
        //copy to a temp array and copmute sequentially for now
        cudaMemcpy(temp_hidden, device_hidden, n * sizeof(float), cudaMemcpyDeviceToHost);
        checkCUDAError("cudaMemcpy device_hidden to temp_hidden failed!");

        //compute activation function
        for (int i = 0; i < n; ++i)
        {
            temp_hidden[i] = 1 / (1 + std::pow(exp(1.0), -temp_hidden[i]));
        }

        //copy back to hidden
        cudaMemcpy(device_hidden, temp_hidden, n * sizeof(float), cudaMemcpyHostToDevice);
        checkCUDAError("cudaMemcpy temp_hidden to device_hidden failed!");


        //legacy
        //kernKJBufferToHidden << < normalBlocksPerGrid, threadsPerBlock >> > (n, device_kj_buffer, device_hidden);
        //kernHiddenMultWeight << < jiBufferBlocksPerGrid, threadsPerBlock >> > (output_num*n, n, device_hidden, weight_matrix, device_ji_buffer);
        //kernJIBufferToOutput << < outputBlocksPerGrid, threadsPerBlock >> > (n, device_ji_buffer, device_output);
        

        timer().startGpuTimer();
        //do we use the same weight? -- in XOR example, it is not
        kernHiddenMultWeight << < outputBlocksPerGrid, threadsPerBlock >> > (output_num, n, device_hidden, device_weight_matrix, device_output);
        timer().endGpuTimer();

        //how to calculate the error and affect the next weights?  -- how to get expcted result? -- read in?
        cudaMemcpy(odata, device_output, output_num * sizeof(float), cudaMemcpyDeviceToHost);
        checkCUDAError("cudaMemcpy device_output to odata failed!");

        //do the activaition functoin here
        for (int i = 0; i < output_num; ++i)
        {
            odata[i] = 1 / (1 + std::pow(exp(1.0), -odata[i]));
        }

        cudaFree(device_input);
        cudaFree(device_weight_matrix);
        cudaFree(device_hidden);
        cudaFree(device_ji_buffer);
        cudaFree(device_output);

        delete[] temp_hidden;
    }
}
