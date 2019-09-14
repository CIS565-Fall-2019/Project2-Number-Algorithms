#pragma once

#include "common.h"
#include <cublas_v2.h>

namespace CharacterRecognition {
    Common::PerformanceTimer& timer();

	void testMatrixMultiply();

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
	@param Cswap	Swap-space for the transpose on C; will allocate if none provided
	*/
	void matMul(cublasHandle_t* handle, const float* A, const float* B, float* C, int m, int k, int n, float* Cswap = NULL);

    // TODO: implement required elements for MLP sections 1 and 2 here
}
