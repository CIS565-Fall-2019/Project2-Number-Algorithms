#pragma once

#include "common.h"
#include <iostream>
#include <vector>

namespace CharacterRecognition {
	using namespace std;
    Common::PerformanceTimer& timer();
	struct Matrix {
		float* data;
		float* dev_data;
		int numRow;
		int numCol;
		size_t dataSize;

		Matrix(int rows, int cols) :
			numRow(rows), numCol(cols)
		{
			dataSize = rows * cols * sizeof(float);
			data = (float*)malloc(dataSize);
			cudaMalloc(&dev_data, dataSize);
		}

		Matrix(const vector<vector<float> >& dataArr ) {
			numRow = dataArr.size();
			if (!numRow) return;
			numCol = dataArr[0].size();
			dataSize = numRow * numCol * sizeof(float);
			data = (float*)malloc(dataSize);
			cudaMalloc(&dev_data, dataSize);

			for (size_t row = 0; row < numRow; row++) {
				for (size_t col = 0; col < numCol; col++) {
					data[row * numCol + col] = dataArr[row][col];
				}
			}
		}

		void copyToHost() {
			cudaMemcpy(data, dev_data, dataSize, cudaMemcpyDeviceToHost);
		}

		void copyToDevice() {
			cudaMemcpy(dev_data, data, dataSize, cudaMemcpyHostToDevice);
		}

		void initWithRandom();
		void initWithTest();
		void initWithZero();
		
		Matrix* dot(const Matrix* other) const;
		Matrix* transpose() const;
		Matrix* subtract(const Matrix* other) const;
		Matrix* diffSquare(const Matrix* other) const;

		void add(const Matrix* other);
		void sigmoid();
		void sigmoidePrime();
		void subtract_inplace(const Matrix* other);
		Matrix* multiply(const float constant);
		Matrix* multiply(const Matrix* other);


		void print() {
			std::cout << "-------- Matrix " << numRow << " X " << numCol << "-------- \n";
			for (int row = 0; row < numRow; row++) {
				for (int col = 0; col < numCol; col++) {
					printf("%f ", *(data + row*numCol + col));
				}
				std::cout << std::endl;
			}
		}


		~Matrix() {
			free(data);
			cudaFree(dev_data);
		}
	};
    // TODO: implement required elements for MLP sections 1 and 2 here
	void init(int inputNeuron, int hiddenNeuron, int outputNeuron, double rate);
	void GPU_fill_rand(float* A, int nr_rows_A, int nr_cols_A);
	void GPU_fill_rand(Matrix* p_matrix);
	void gpu_blas_mmul(const float* A, const float* B, float* C, const int m, const int k, const int n);
	Matrix* computeOutput(const vector<float>& input);
	float learn(const vector<float> expectedOutput);

	void unitTest();

}
