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

    __host__ __device__ float genRandom(float time, int index) {
        thrust::default_random_engine rng(hash((int)(index * time)));
        thrust::uniform_real_distribution<float> unitDistrib(-1, 1);

        return (float)unitDistrib(rng);
    }

    __global__ void kernInitRandomWeights(int N, float* wtMat, float scale)
    {
        int index = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (index < N) {
            float rand = genRandom(N, index);
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

        for (int idx = 0; idx < iDim; idx++)
        {
            int wtIdx = idx * oDim + index;
            odata[index] += wtMat[wtIdx] * idata[idx];
        }
    }

    __global__ void kernActivationFxn(int N, float* idata, float* odata)
    {
        int index = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (index >= N) { return; }

        float x = idata[index];
        float e = exp(-x);
        odata[index] = 1.0f / (1.0f + e);
    }

    __global__ void kernCalcErrors(int N, float* target, float* output, float* odata)
    {
        int index = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (index >= N) { return; }

        odata[index] = target[index] - output[index];
    }

    __global__ void kernEditWeightsji(int N, int iDim, float lambda, float* hidden, float* errors, float* outputSums, 
        float* partialErr, float* wtMat)
    {
        // for hidden to output weights:
        // delta = lambda * value of hidden node * (target - output) * derivative of f(x) (where x is the sum before it went in f(x) or is just the output??)
        // derivative of f = f * (1-f)

        int index = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (index >= N) { return; }

        int i = index % iDim;
        int j = index / iDim;
        
        float x = outputSums[i];
        float fx = 1.0f / (1.0f + exp(-x));
        partialErr[i] = errors[i] * fx * (1 - fx);
        float deltaW = lambda * hidden[j] * partialErr[i];

        wtMat[index] += deltaW;
    }

    __global__ void kernEditWeightskj(int N, int jDim, int iDim, float lambda, float* input, float* hiddenSums, 
        float* partialErr, float* wji,
        float* wtMat)
    {
        // for hidden to output weights:
        // delta = lambda * value of input node * derivative of f(x) * 
        // derivative of f = f * (1-f)

        int index = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (index >= N) { return; }

        int j = index % jDim;
        int k = index / jDim;

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

    void makeWeightMat(int n, float* data)
    {
        float* dev_data;
        cudaMalloc((void**)&dev_data, n * sizeof(float));

        kernInitRandomWeights << <n, blockSize >> > (n, dev_data, 30);

        cudaMemcpy(data, dev_data, n * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(dev_data);
    }

	// TODO: implement required elements for MLP sections 1 and 2 here
    float mlpTrain(int i, int j, int k, float* odata, float* idata, float* wkj, float* wji, float* target)
    {
        float *dev_input, *dev_hidden, *dev_output;
        float *dev_hiddenSums, *dev_outputSums;
        float *dev_wkj, *dev_wji;
        float *dev_target, *dev_errors, *dev_partialErr, *dev_tempwji;

        cudaMalloc((void**)&dev_input, k * sizeof(float));
        cudaMalloc((void**)&dev_hidden, j * sizeof(float));
        cudaMalloc((void**)&dev_output, i * sizeof(float));
        cudaMemcpy(dev_input, idata, k * sizeof(float), cudaMemcpyHostToDevice);

        cudaMalloc((void**)&dev_hiddenSums, j * sizeof(float));
        cudaMalloc((void**)&dev_outputSums, i * sizeof(float));

        cudaMalloc((void**)&dev_wkj, k * j * sizeof(float));
        cudaMalloc((void**)&dev_wji, j * i * sizeof(float));
        cudaMemcpy(dev_wkj, wkj, k * j * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_wji, wji, j * i * sizeof(float), cudaMemcpyHostToDevice);

        cudaMalloc((void**)&dev_target, i * sizeof(float));
        cudaMalloc((void**)&dev_errors, i * sizeof(float));
        cudaMalloc((void**)&dev_partialErr, i * sizeof(float));
        cudaMalloc((void**)&dev_tempwji, i * j * sizeof(float));
        cudaMemcpy(dev_target, target, i * sizeof(float), cudaMemcpyHostToDevice);

        // initialize non input buffers to zeros
        kernInitZero << <j, blockSize >> > (j, dev_hidden);
        kernInitZero << <i, blockSize >> > (i, dev_output);

        // input -> hidden
        kernSumWeights << <j, blockSize >> > (k, j, dev_wkj, dev_input, dev_hiddenSums);
        kernActivationFxn << <j, blockSize >> > (j, dev_hiddenSums, dev_hidden);

        // hidden -> output
        kernSumWeights << <i, blockSize >> > (j, i, dev_wji, dev_hidden, dev_outputSums);
        kernActivationFxn << <i, blockSize >> > (i, dev_outputSums, dev_output);

        // calculate error, lambda 
        kernCalcErrors << <i, blockSize >> > (i, dev_target, dev_output, dev_errors);
        
        float* errs = new float[i];
        cudaMemcpy(errs, dev_errors, i * sizeof(float), cudaMemcpyDeviceToHost);
        float sumErr = 0;
        for (int e = 0; e < i; e++)
        {
            sumErr += (errs[e]*errs[e]);
        }
        sumErr /= 2.0f;
        float lambda = sumErr;

        // update weights
        cudaMemcpy(dev_tempwji, dev_wji, j * i * sizeof(float), cudaMemcpyDeviceToDevice);
        kernEditWeightsji << <j*i, blockSize >> > (j*i, i, lambda, dev_hidden, dev_errors, dev_output,
            dev_partialErr, dev_wji);
        kernEditWeightskj << <k*j, blockSize >> > (k*j, j, i, lambda, dev_input, dev_hidden, dev_partialErr,
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

        cudaFree(dev_target);
        cudaFree(dev_errors);
        cudaFree(dev_partialErr);
        cudaFree(dev_tempwji);

        return sumErr;
    }

    void mlpRun(int i, int j, int k, float* odata, float* idata, float* wkj, float* wji)
    {
        float *dev_input, *dev_hidden, *dev_output;
        float *dev_hiddenSums, *dev_outputSums;
        float *dev_wkj, *dev_wji;

        cudaMalloc((void**)&dev_input, k * sizeof(float));
        cudaMalloc((void**)&dev_hidden, j * sizeof(float));
        cudaMalloc((void**)&dev_output, i * sizeof(float));
        cudaMemcpy(dev_input, idata, k * sizeof(float), cudaMemcpyHostToDevice);

        cudaMalloc((void**)&dev_hiddenSums, j * sizeof(float));
        cudaMalloc((void**)&dev_outputSums, i * sizeof(float));

        cudaMalloc((void**)&dev_wkj, k * j * sizeof(float));
        cudaMalloc((void**)&dev_wji, j * i * sizeof(float));
        cudaMemcpy(dev_wkj, wkj, k * j * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_wji, wji, j * i * sizeof(float), cudaMemcpyHostToDevice);

        // initialize non input buffers to zeros
        kernInitZero << <j, blockSize >> > (j, dev_hidden);
        kernInitZero << <i, blockSize >> > (i, dev_output);

        // input -> hidden
        kernSumWeights << <j, blockSize >> > (k, j, dev_wkj, dev_input, dev_hiddenSums);
        kernActivationFxn << <j, blockSize >> > (j, dev_hiddenSums, dev_hidden);

        // hidden -> output
        kernSumWeights << <i, blockSize >> > (j, i, dev_wji, dev_hidden, dev_outputSums);
        kernActivationFxn << <i, blockSize >> > (i, dev_outputSums, dev_output);

        cudaMemcpy(odata, dev_output, i * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(dev_input);
        cudaFree(dev_hidden);
        cudaFree(dev_output);

        cudaFree(dev_hiddenSums);
        cudaFree(dev_outputSums);

        cudaFree(dev_wkj);
        cudaFree(dev_wji);
    }
}
