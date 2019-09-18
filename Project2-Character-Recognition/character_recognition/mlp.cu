#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "mlp.h"
#include <cuda_runtime_api.h>

#define num_examples 4
#define num_input_chanels 2
#define num_hidden_channels 2
#define num_out_channels 2
#define batch_size 1


namespace CharacterRecognition {
    using Common::PerformanceTimer;
    PerformanceTimer& timer()
    {
        static PerformanceTimer timer;
        return timer;
    }


	float *inp = new float[batch_size*num_input_chanels];
	float *dev_inp;

	float *target = new float[batch_size*num_out_channels];
	float *dev_target;


	float *w1 = new float[num_input_chanels*num_hidden_channels];
	float *dev_w1;

	float *dev_a1_B_relu;
	float *dev_a1_A_relu;

	float *w2 = new float[num_hidden_channels * num_out_channels];
	float *dev_w2;

	float *dev_out_B_soft;
	float *dev_out_A_soft;

	float *dev_dw1;
	float *dev_dw2;


	float initweight(int n, float *w) {

		for (int i = 0; i < n; i++) {
			w[i] = ((2* rand()/RAND_MAX)-1) * 0.002;
		}

		return *w;
	}
	
	//int in = num_hidden_channels
	w1 = initweight(num_input_chanels * num_hidden_channels, w1);
	w2 = initweight(num_hidden_channels * num_out_channels, w2);
	
	//cudaMemset(dataGPU, 0, 1000 * sizeof(float));

	//cudaMalloc((void**)&dev_odata, n * sizeof(int));
	//cudaMemcpy(dev_odata, odata, sizeof(int) * n, cudaMemcpyHostToDevice);

	void initialize() {

		cudaMalloc((void**)&dev_inp, batch_size * num_input_chanels * sizeof(float));
		cudaMemcpy(dev_inp, inp, batch_size * num_input_chanels * sizeof(float), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_target, batch_size * num_out_channels * sizeof(float));
		cudaMemcpy(dev_target, target, batch_size * num_out_channels * sizeof(float), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_w1, num_input_chanels*num_hidden_channels * sizeof(float));
		cudaMemcpy(dev_w1, w1, num_input_chanels*num_hidden_channels * sizeof(float), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_w2, num_hidden_channels * num_out_channels * sizeof(float));
		cudaMemcpy(dev_w2, w2, num_hidden_channels * num_out_channels * sizeof(float), cudaMemcpyHostToDevice);

		cudaMalloc((void**)&dev_a1_B_relu, batch_size * num_hidden_channels * sizeof(float));
		cudaMalloc((void**)&dev_a1_A_relu, batch_size * num_hidden_channels * sizeof(float));

		cudaMalloc((void**)&dev_out_B_soft, batch_size * num_out_channels * sizeof(float));
		cudaMalloc((void**)&dev_out_A_soft, batch_size * num_out_channels * sizeof(float));

		cudaMalloc((void**)&dev_dw1, num_input_chanels * num_hidden_channels * sizeof(float));
		cudaMalloc((void**)&dev_dw2, num_hidden_channels * num_out_channels * sizeof(float));
		// random initializes w1,w2 and the rest to zeros
	}
	
	
	
	__global__ void SumVectors(float *vec1, float *vec2, float *out) {

		int index = blockIdx.x * blockDim.x + threadIdx.x;
		out[index] = vec1[index] + vec2[index];
	}

	// help from http://luniak.io/cuda-neural-network-implementation-part
	__global__ void ForwardLayer(float* W, float* activation, float* Z, float* b, int W_x_dim, int W_y_dim, int Acti_x_dim, int Acti_y_dim) {

		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;

		int Z_x_dim = Acti_x_dim;
		int Z_y_dim = W_y_dim;

		float Z_value = 0;

		if (row < Z_y_dim && col < Z_x_dim) {
			for (int i = 0; i < W_x_dim; i++) {
				Z_value += W[row * W_x_dim + i] * activation[i * Acti_x_dim + col];
			}
			Z[row * Z_x_dim + col] = Z_value + b[row];
		}
	}

	__global__ void BackpropLayer(float* W, float* dZ, float *dActi, int W_x_dim, int W_y_dim, int dZ_x_dim, int dZ_y_dim) {

		int col = blockIdx.x * blockDim.x + threadIdx.x;
		int row = blockIdx.y * blockDim.y + threadIdx.y;

		int dActi_x_dim = dZ_x_dim;
		int dActi_y_dim = W_x_dim;

		float dA_value = 0.0f;

		if (row < dActi_y_dim && col < dActi_x_dim) {
			for (int i = 0; i < W_y_dim; i++) {
				dA_value += W[i * W_x_dim + row] * dZ[i * dZ_x_dim + col];
			}
			dActi[row * dActi_x_dim + col] = dA_value;
		}
	}


	__global__ void reluActivationForward(float *Z, float *activation, int n) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;

		if (index < n) {
			A[index] = fmaxf(Z[index], 0);
		}
	}

	__global__ void reluActivationBackprop(int n ,float *Z, float *dZ, float *dactivation) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;

		if (index < n) {
			if (Z[index] > 0) {
				dZ[index] = dactivation[index];
			}
			else {
				dZ[index] = 0;
			}
		}
	}

	__device__ float sigmoid(float x) {
		return 1.0f / (1 + exp(-x));
	}

	__global__ void sigmoidActivationForward(int n, float* Z, float* activation) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;

		if (index < n) {
			activation[index] = sigmoid(Z[index]);
		}
	}

	__global__ void sigmoidActivationBackprop(int n, float *Z, float *dZ, float *dactivation) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;

		if (index < n) {
			dZ[index] = dactivation[index] * sigmoid(Z[index]) * (1 - sigmoid(Z[index]));
		}
	}


	__global__ void BCELoss(int n,float *preds, float *target, float *loss) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;

		if (index < n) {
			float partial_cost = target[index] * logf(preds[index])
				+ (1.0f - target[index]) * logf(1.0f - preds[index]);
			
			atomicAdd(loss, -partial_cost / n);
		}
	}

	__global__ void dBCELoss(int n ,float *preds, float *target, float* dY) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;

		if (index < n) {
			dY[index] = -1.0 * (target[index] / preds[index] - (1 - target[index]) / (1 - preds[index]));
		}
	}

	__global__ void ElementWiseMultiplication(int n, float *input1, float *input2, float *out) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;

		if (index <n) {
			out[index] = input1[index] * input2[index];
		}
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

	// TODO: implement required elements for MLP sections 1 and 2 here
}
