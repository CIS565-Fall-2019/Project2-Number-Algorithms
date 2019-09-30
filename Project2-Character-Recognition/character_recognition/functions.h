#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

namespace Functions {
	// Non Linear Operations
	__global__ void sigmoidActivation(float * input, float * output, int x_dim, int y_dim);
	__global__ void reluActivation(float * input, float * output, int x_dim, int y_dim);
	__global__ void ExponentialActivation(float * input, float * output, int x_dim, int y_dim);
	__global__ void CrossEntropyLoss(float * target, float * predicted, float* output, int x_dim, int y_dim);

	// Operations between an array and a constant
	__global__ void Add(float* input, float* constant, int x_dim, int y_dim);
	__global__ void Divide(float* input, float* constant, int x_dim, int y_dim);
	__global__ void Multiply(float* input, float* constant, int x_dim, int y_dim);
	__global__ void Subtract(float* input, float* constant, int x_dim, int y_dim);
	
	// Element wise Operations between two arrays
	__global__ void ElementwiseMultiplication(float* input1, float* input2, float* output, int x_dim, int y_dim);
	__global__ void ElementwiseSubtraction(float* input1, float* input2, float* output, int x_dim, int y_dim);
	
	// Custom Kernel for gradient descent
	__global__ void KernelElementwiseMultiplySigmoid(float* input_output, float* input, int x_dim, int y_dim);
	__global__ void KernelElementwiseMultiplyRelu(float* input_output, float* input, int x_dim, int y_dim);

	__global__ void normalize(float* array, int x_dim, int y_dim);
}
