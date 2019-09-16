#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "mlp.h"
#include <math.h>

#define NUM_LAYERS 2
#define INPUT_NODES 2
#define NUM_WEIGHTS 2

#define blockSize 128

namespace CharacterRecognition {
    using Common::PerformanceTimer;
    PerformanceTimer& timer()
    {
        static PerformanceTimer timer;
        return timer;
    }

	void memory_debug_float(int elements, float* cuda_mem, float* cpu_mem)
	{
		printf("elements %d\n ", elements);
		cudaMemcpy(cpu_mem, cuda_mem, elements * sizeof(float), cudaMemcpyDeviceToHost);
		checkCUDAErrorFn("debug failed!");
		printf("=============================\n");
		for (int i = 0; i < elements; i++)
		{
			printf("out[%d] %f \n", i, cpu_mem[i]);
		}
		printf("=============================\n");
	}

        
    // TODO: __global__

	__global__ void kernel_feed_forward(int n, float* dev_in, float* weights)
	{
		int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

		if (tid >= n)
		{
			return;
		}

		float data = dev_in[tid];

		dev_in[tid] = (data * weights[tid]) + (data * weights[tid+n]);
	}

	__global__ void kernel_activate(int n, float* dev_in)
	{
		int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

		if (tid >= n)
		{
			return;
		}

		float var = (dev_in[tid] * -1);
		float e = expf(var);
		dev_in[tid] = 1 / (1 + e);
	}

	void feed_forward(float* in, float* out, float* weights, int length, int w_length)
	{
		for (int i = 0; i < length; i++)
		{
			out[i] = 0;
			for (int j = 0; j < w_length; j++)
			{
				float weight = *((weights+i*w_length) + j);
				out[i] += (in[i] * weight);
				printf("in[%d] = %f * %f\n", i, in[i], weight);
			}
			printf("out[%d] = %f\n", i, out[i]);
		}
	}

	void activate_function(float* in, float* out, int length)
	{
		for (int i = 0; i < length; i++)
		{
			float var = (in[i] * -1);
			float e = exp(var);
			out[i] = 1 / (1 + e);
			printf("activate: %f\n", out[i]);
		}
	}

    /**
        * Example of use case (follow how you did it in stream compaction)
        */
    void train(int n, int *odata, const int *idata) {
        timer().startGpuTimer();
        
		float weights[6];
		float in_data[4] = { 0,0 };
		float out_data[4];
		float temp[4];
		weights[0] = 10.1;
		weights[1] = .9;
		weights[2] = 20;
		weights[3] = .87;
		weights[4] = 41;
		weights[5] = -54;
		int wt_idx = 0;

		float* dev_buff;
		float* dev_wts;

		float* host_buff = new float[4];

		int rounded_depth = ilog2ceil(NUM_LAYERS);
		int rounded_elements = 1 << rounded_depth;
		printf("rounded elements %d\n ", rounded_elements);
		dim3 fullBlocksPerGrid((rounded_elements + blockSize - 1) / blockSize);
		


		cudaMalloc((void**)&dev_buff, 2 * sizeof(float));
		checkCUDAErrorFn("malloc dev_boolbuff in failed!");
		cudaMemcpy(dev_buff, in_data, 2 * sizeof(float), cudaMemcpyHostToDevice);
		checkCUDAErrorFn("dev_in copy failed!");

		/*cudaMalloc((void**)&dev_out, n * sizeof(int));
		checkCUDAErrorFn("malloc dev_out in failed!");
		cudaMalloc((void**)&dev_in, rounded_elements * sizeof(int));
		checkCUDAErrorFn("malloc dev_in in failed!");*/


		// is there a way to place this in memory at compile time?
		cudaMalloc((void**)&dev_wts, 6 * sizeof(float));
		checkCUDAErrorFn("malloc dev_in in failed!");
		cudaMemcpy(dev_wts, weights, 6 * sizeof(float), cudaMemcpyHostToDevice);
		checkCUDAErrorFn("dev_weights copy failed!");
		
		// input nodes  
		// each wight has an impact on the other nodes
		for (int i = NUM_LAYERS; i > 0; i--)
		{
			//feed_forward(&in_data[0], &temp[0], (float*)&weights[wt_idx], i, INPUT_NODES);
			//activate_function(&temp[0], &out_data[0], i );
			kernel_feed_forward << < fullBlocksPerGrid, blockSize >> > (i, dev_buff, &dev_wts[wt_idx]);
			checkCUDAErrorFn("feed forward failed!");
			memory_debug_float(i, &dev_buff[0], &host_buff[0]);
			kernel_activate << < fullBlocksPerGrid, blockSize >> > (i, dev_buff);
			checkCUDAErrorFn("activate failed!");
			memory_debug_float(i, dev_buff, host_buff);
			//feed_forward(&out_data[0], &temp[0], (float*)&weights[wt_idx][0], 1,2);
			//activate_function(&temp[0], &out_data[0], 1);
			//std::swap(in_data, out_data);
			wt_idx += 4; // length of array? NUM_NODES* INPUT NODES?
		}

		//error = out_data[0]
        timer().endGpuTimer();
    }
    

	// TODO: implement required elements for MLP sections 1 and 2 here
}
