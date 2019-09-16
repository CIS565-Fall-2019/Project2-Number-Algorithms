#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/random.h>
#include "common.h"
#include "mlp.h"

namespace CharacterRecognition {
    using Common::PerformanceTimer;
    PerformanceTimer& timer()
    {
        static PerformanceTimer timer;
        return timer;
    }
        
    // TODO: __global__

    /**
        * Example of use case (follow how you did it in stream compaction)
        */
    /*void scan(int n, int *odata, const int *idata) {
        timer().startGpuTimer();
        // TODO
        timer().endGpuTimer();
    }
    */

#define blockSize 128

    __host__ __device__ unsigned int hash(unsigned int a) {
        a = (a + 0x7ed55d16) + (a << 12);
        a = (a ^ 0xc761c23c) ^ (a >> 19);
        a = (a + 0x165667b1) + (a << 5);
        a = (a + 0xd3a2646c) ^ (a << 9);
        a = (a + 0xfd7046c5) + (a << 3);
        a = (a ^ 0xb55a4f09) ^ (a >> 16);
        return a;
    }

    __host__ __device__ float genRandom(int index) {
        thrust::default_random_engine rng(hash((int)(index)));
        thrust::uniform_real_distribution<float> unitDistrib(-1, 1);

        return (float)unitDistrib(rng);
    }

    __global__ void kernInitRandomWeights(int N, float* wtMat, float scale)
    {
        int index = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (index < N) {
            float rand = genRandom(index);
            wtMat[index] = scale * rand;
        }
    }

    __global__ void kernInitZero(int N, float* data)
    {
        int index = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (index < N) 
        { 
            data[index] = 0; 
        }
    }

    __global__ void kernSumWeights(int iDim, int oDim, float* wtMat, float* idata, float* odata)
    {
        int index = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (index >= oDim) { return; }

        int row = index * iDim;
        for (int idx = 0; idx < iDim; idx++)
        {
            int wtIdx = idx + row;
            odata[index] += wtMat[wtIdx] * idata[idx];
        }
    }

    __global__ void kernActivationFxn(int N, float* idata, float* odata)
    {
        int index = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (index >= N) { return; }

        float x = idata[index];
        odata[index] = 1.0f / (1.0f + exp(-x));
    }

    __global__ void kernCalcErrors(int N, float* target, float* output, float* odata)
    {
        int index = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (index >= N) { return; }

        odata[index] = target[index] - output[index];
    }

    __global__ void kernEditWeightsji(int N, int jDim, float lambda, float* hidden, float* errors, float* outputSums, 
        float* partialErr, float* wtMat)
    {
        // for hidden to output weights:
        // delta = lambda * value of hidden node * (target - output) * derivative of f(x) (where x is the sum before it went in f(x) or is just the output??)
        // derivative of f = f * (1-f)

        int index = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (index >= N) { return; }

        int i = index % jDim;
        int j = index - i;
        
        float x = outputSums[i];
        float fx = 1.0f / (1.0f + exp(-x));
        partialErr[i] = errors[i] * fx * (1 - fx);
        float deltaW = lambda * hidden[j] * partialErr[i];

        wtMat[index] += deltaW;
    }

    __global__ void kernEditWeightskj(int N, int kDim, int jDim, int iDim, float lambda, float* input, float* hiddenSums, 
        float* partialErr, float* wji,
        float* wtMat)
    {
        // for hidden to output weights:
        // delta = lambda * value of input node * derivative of f(x) * 
        // derivative of f = f * (1-f)

        int index = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (index >= N) { return; }

        int j = index % kDim;
        int k = index - j;

        float sumPropErrs = 0;
        for (int i = 0; i < iDim; i++)
        {
            sumPropErrs += partialErr[i] * wji[j + i * jDim];
        }

        float x = hiddenSums[j];
        float fx = 1.0f / (1.0f + exp(-x));
        float deltaW = lambda * input[k] * sumPropErrs * fx * (1 - fx);

        wtMat[index] += deltaW;
    }

	// TODO: implement required elements for MLP sections 1 and 2 here
    void mlpTrain(int i, int j, int k, float* odata, float* idata, float* wkj, float* wji, float* target)
    {
        float *dev_input, *dev_hidden, *dev_output;
        float *dev_hiddenSums, *dev_outputSums;
        float *dev_wkj, *dev_wji;
        float *dev_target, *dev_errors, *dev_partialErr, *dev_tempwji;

        cudaMalloc((void**)&dev_input, k * sizeof(float));
        cudaMalloc((void**)&dev_hidden, j * sizeof(float));
        cudaMalloc((void**)&dev_output, i * sizeof(float));
        cudaMemcpy(dev_input, idata, i * sizeof(float), cudaMemcpyHostToDevice);

        cudaMalloc((void**)&dev_hiddenSums, j * sizeof(float));
        cudaMalloc((void**)&dev_outputSums, i * sizeof(float));

        cudaMalloc((void**)&dev_wkj, k * j * sizeof(float));
        cudaMalloc((void**)&dev_wji, j * i * sizeof(float));

        cudaMalloc((void**)&dev_target, i * sizeof(float));
        cudaMalloc((void**)&dev_errors, i * sizeof(float));
        cudaMalloc((void**)&dev_partialErr, i * sizeof(float));
        cudaMalloc((void**)&dev_tempwji, i * j * sizeof(float));
        cudaMemcpy(dev_target, target, i * sizeof(float), cudaMemcpyHostToDevice);

        //cudaMemcpy(dev_wkj, wkj, k * j * sizeof(float), cudaMemcpyHostToDevice);
        //cudaMemcpy(dev_wji, wji, j * i * sizeof(float), cudaMemcpyHostToDevice);

        dim3 ifullBlocksPerGrid((i + blockSize - 1) / blockSize);
        dim3 jfullBlocksPerGrid((j + blockSize - 1) / blockSize);
        dim3 kfullBlocksPerGrid((k + blockSize - 1) / blockSize);
        dim3 wkjfullBlocksPerGrid((k*j + blockSize - 1) / blockSize);
        dim3 wjifullBlocksPerGrid((j*i + blockSize - 1) / blockSize);

        // initialize non input buffers to zeros and give weight matrices random values
        kernInitRandomWeights << <wkjfullBlocksPerGrid, blockSize >> > (k*j, dev_wkj, 100);
        kernInitRandomWeights << <wjifullBlocksPerGrid, blockSize >> > (j*i, dev_wji, 100);

        kernInitZero << <jfullBlocksPerGrid, blockSize >> > (j, dev_hidden);
        kernInitZero << <ifullBlocksPerGrid, blockSize >> > (i, dev_output);

        // input -> hidden
        kernSumWeights << <jfullBlocksPerGrid, blockSize >> > (k, j, dev_wkj, dev_input, dev_hiddenSums);
        kernActivationFxn << <jfullBlocksPerGrid, blockSize >> > (j, dev_hiddenSums, dev_hidden);

        // hidden -> output
        kernSumWeights << <ifullBlocksPerGrid, blockSize >> > (j, i, dev_wji, dev_hidden, dev_outputSums);
        kernActivationFxn << <ifullBlocksPerGrid, blockSize >> > (i, dev_outputSums, dev_output);

        // calculate error, lambda 
        kernCalcErrors << <ifullBlocksPerGrid, blockSize >> > (i, dev_target, dev_output, dev_errors);
        
        float* errs = new float[i];
        cudaMemcpy(errs, dev_errors, i * sizeof(float), cudaMemcpyDeviceToHost);
        float sumErr = 0;
        for (int e = 0; e < i; e++)
        {
            sumErr += (errs[e]*errs[e]);
        }
        float lambda = sumErr/2.0f;

        // update weights
        cudaMemcpy(dev_tempwji, dev_wji, j * i * sizeof(float), cudaMemcpyDeviceToDevice);
        kernEditWeightsji << <wjifullBlocksPerGrid, blockSize >> > (j*i, j, lambda, dev_hidden, dev_errors, dev_outputSums,
            dev_partialErr, dev_wji);
        kernEditWeightskj << <wjifullBlocksPerGrid, blockSize >> > (k*j, k, j, i, lambda, dev_input, dev_hiddenSums, dev_partialErr,
            dev_tempwji, dev_wkj);

        cudaMemcpy(odata, dev_output, i * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(wkj, dev_wkj, k * j * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(wji, dev_wji, j * i * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(dev_input);
        cudaFree(dev_hidden);
        cudaFree(dev_output);

        cudaFree(dev_hiddenSums);
        cudaFree(dev_outputSums);

        cudaFree(dev_wkj);
        cudaFree(dev_wji);
    }
}
