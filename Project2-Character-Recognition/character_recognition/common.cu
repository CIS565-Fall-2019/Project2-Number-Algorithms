#include "common.h"
#include "device_launch_parameters.h"

void checkCUDAErrorFn(const char *msg, const char *file, int line) {
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess == err) {
		return;
	}

	fprintf(stderr, "CUDA error");
	if (file) {
		fprintf(stderr, " (%s:%d)", file, line);
	}
	fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
	exit(EXIT_FAILURE);
}

void printArray(int n, float *a, bool abridged = false) {
	printf("    [ ");
	for (int i = 0; i < n; i++) {
		if (abridged && i + 2 == 15 && n > 16) {
			i = n - 2;
			printf("... ");
		}
		printf("%f ", a[i]);
	}
	printf("]\n");
}

void printCudaArray(int size, float* data) {
	float *d_data = new float[size];
	cudaMemcpy(d_data, data, size * sizeof(float), cudaMemcpyDeviceToHost);
	printArray(size, d_data, true);
}

void printCuda2DArray(int height, int width, float* data) {
	float *d_data = new float[width*height];
	cudaMemcpy(d_data, data, width*height * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < height; i++)
		printArray(width, d_data + i * width, true);
}

__device__ float sigmoid(float x) {
	return 1.0f / (1 + exp(-x));
}

__global__ void Functions::sigmoidActivation(float* input, float* output, int x_dim, int y_dim) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < x_dim * y_dim) {
		output[index] = sigmoid(input[index]);
	}
}

__global__ void Functions::reluActivation(float* input, float* output, int x_dim, int y_dim) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < x_dim * y_dim) {
		output[index] = input[index] > 0 ? input[index] : 0;
	}
}

__global__ void Functions::ExponentialActivation(float* input, float* output, int x_dim, int y_dim)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < x_dim * y_dim) {
		output[index] = exp(input[index]);
	}
}

__global__ void Functions::Divide(float* input, float* constant, int x_dim, int y_dim)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < x_dim * y_dim) {
		if(constant > 0)
			input[index] /= *constant;
	}
}

__global__ void Functions::Multiply(float* input, float* constant, int x_dim, int y_dim)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < x_dim * y_dim) {
		input[index] *= *constant;
	}
}

__global__ void Functions::Subtract(float * input, float * constant, int x_dim, int y_dim)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < x_dim * y_dim) {
		input[index] -= *constant;
	}
}

__global__ void Functions::ElementwiseMultiplication(float * input1, float * input2, float* output, int x_dim, int y_dim)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < x_dim * y_dim) {
		output[index] = input1[index] * input2[index];
	}
}

__global__ void Functions::ElementwiseSubtraction(float * input1, float * input2, float* output, int x_dim, int y_dim)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < x_dim * y_dim) {
		output[index] = input1[index] - input2[index];
	}
}

__global__ void Functions::KernelElementwiseMultiplySigmoid(float * input_output, float * input, int x_dim, int y_dim)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < x_dim * y_dim) {
		input_output[index] *=  input[index] * (1 - input[index]);
	}
}

__global__ void Functions::KernelElementwiseMultiplyRelu(float * input_output, float * input, int x_dim, int y_dim)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < x_dim * y_dim) {
		input_output[index] = input[index] > 0 ? input_output[index] : 0;
	}
}
