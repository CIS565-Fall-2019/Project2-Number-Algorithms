#pragma once

#include "common.h"
#include <cublas_v2.h>

namespace CharacterRecognition {
    Common::PerformanceTimer& timer();

	void kmallocBuffers();
	void kfreeBuffers();

	/**
	Fills the buffer with random values between the two provided values
	*/
	void gpuFillRand(float* A, int nr_rows_A, int nr_cols_A, float lo = 0.0, float hi = 1.0);

	/**
	Does the necessary model training so it can recognize characters, hopefully

	@param records Array full of records to be used as training data
	@param numIterations How many times to run the forward/back propagation
	*/
	void trainWeights(InputData_v records, int numIterations = 100);

	/**
	Forward-propagate in host memory space
	Returns an array with the error bars
	*/
	float_v forwardPropagate(InputData x, float* resultArray, cublasHandle_t* handle = NULL);

	/**
	Backward-propagate our error to update the relevant weights
	*/
	void backPropagate(cublasHandle_t* handle);

	/**
	For forward propagation, does the steps of convolving and max pooling
	Does not work backwards in back-propagation (filters remain static)
	*/
	int convolveStep(float* inputLayer, int inputLayerSize, float* outputPoolingLayer, float* outputLayer, int poolWidth);

	/**
	Gives the error value between the two data points
	*/
	float_v calcErrorSingle(InputData record, float* resultArray, float* kResultArray = NULL);

	/**
	Calculates the sum of all run error functions, squared
	*/
	float_v calcSumSquareErrors(float_vv errorVals);

	/**
	Transposes A "in place" in kernel space
	A goes from having dimensions mxn to dimensions nxm
	*/
	void transpose(float* A, float* Aswap, int m, int n);

	/**
	Wraps up the cublas matrix multiplication on arrays in kernel space
	Does so with the assumption that all of them are in row-major order
	Also handles the necessary transposing

	@param A		The first matrix to multiply, with dimensions mxk
	@param B		The second matrix to multiply, with dimensions kxn
	@param C		Location for the output matrix, with dimensionx mxn
	*/
	void matMul(cublasHandle_t* handle, const float* A, const float* B, float* C, int m, int k, int n);


}
