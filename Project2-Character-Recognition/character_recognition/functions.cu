#include "functions.h"
#include "device_launch_parameters.h"

__device__ float sigmoid(float x) {
	return 1.0f / (1 + exp(-x));
}
namespace Functions {
	__global__ void sigmoidActivation(float* input, float* output, int x_dim, int y_dim) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index < x_dim * y_dim) {
			output[index] = sigmoid(input[index]);
		}
	}

	__global__ void reluActivation(float* input, float* output, int x_dim, int y_dim) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index < x_dim * y_dim) {
			output[index] = input[index] > 0 ? input[index] : 0;
		}
	}

	__global__ void ExponentialActivation(float* input, float* output, int x_dim, int y_dim)
	{
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index < x_dim * y_dim) {
			output[index] = exp(input[index]);
		}
	}

	__global__ void CrossEntropyLoss(float * target, float * predicted, float * output, int x_dim, int y_dim)
	{
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index < x_dim * y_dim) {
			output[index] = -1 * (target[index] * logf(predicted[index]));
		}
	}

	__global__ void Add(float * input, float * constant, int x_dim, int y_dim)
	{
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index < x_dim * y_dim) {
			input[index] += *constant;
		}
	}

	__global__ void Divide(float* input, float* constant, int x_dim, int y_dim)
	{
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index < x_dim * y_dim) {
			if (constant > 0)
				input[index] /= *constant;
		}
	}

	__global__ void Multiply(float* input, float* constant, int x_dim, int y_dim)
	{
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index < x_dim * y_dim) {
			input[index] *= *constant;
		}
	}

	__global__ void Subtract(float * input, float * constant, int x_dim, int y_dim)
	{
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index < x_dim * y_dim) {
			input[index] -= *constant;
		}
	}

	__global__ void ElementwiseMultiplication(float * input1, float * input2, float* output, int x_dim, int y_dim)
	{
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index < x_dim * y_dim) {
			output[index] = input1[index] * input2[index];
		}
	}

	__global__ void ElementwiseSubtraction(float * input1, float * input2, float* output, int x_dim, int y_dim)
	{
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index < x_dim * y_dim) {
			output[index] = input1[index] - input2[index];
		}
	}

	__global__ void KernelElementwiseMultiplySigmoid(float * input_output, float * input, int x_dim, int y_dim)
	{
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index < x_dim * y_dim) {
			input_output[index] *= input[index] * (1 - input[index]);
		}
	}

	__global__ void KernelElementwiseMultiplyRelu(float * input_output, float * input, int x_dim, int y_dim)
	{
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index < x_dim * y_dim) {
			input_output[index] = input[index] > 0 ? input_output[index] : 0;
		}
	}
	__global__ void normalize(float * array, int x_dim, int y_dim)
	{
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index < x_dim * y_dim) {
			array[index] = ((array[index] * 2) - 1) / 10;
		}
	}
}