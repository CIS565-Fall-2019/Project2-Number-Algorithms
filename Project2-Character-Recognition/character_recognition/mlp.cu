#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <cublas_v2.h>
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

	struct MLPData {
		int input_len;
		int hidden_len;
		int out_len;

		int* input_layer;
		int* hidden_layer;
		int* output_layer;

		int* ih_weights;
		int* ho_weights;
	};

	void intializeMLP(int input_len, int hidden_len, int out_len) {
	}

	void loadInputMLP(struct CharacterRecognition::MLPData* mlp, int* idata, int len) {

	}

	void stepMLP(struct CharacterRecognition::MLPData* mlp) {
		// This MLP flows from input to output with no feedback
		// So we will work in steps
		// 1) (CPU) Read input data and copy to Device
		// 2) (GPU) Each Node of Hidden layer computes its value by sum(input*weight foreach input) and compares it to activation
		// 3) (GPU) Each Node of the Output layer computes its value by sum(hidden*weifght foreach hidden) and compares it to activation
		// 4) (CPU) Reads output nodes and uses lookup table to get result.
	}

	void matrixMultiplyExample() {
		int input_rows;
		int input_cols;
		int ih_weight_rows;
		int ih_weight_cols;
		int hidden_rows;
		int hidden_cols;

		// Allocate the matricies
		float *input_matrix = (float*)malloc(input_rows * input_cols * sizeof(float));
		float *weight_matrix = (float*)malloc(ih_weight_rows * ih_weight_cols * sizeof(float));
		float *hidden_matrix = (float*)malloc(hidden_rows * hidden_cols * sizeof(float));

		// Allocate the matricies on the GPU
		float* dev_input_matrix;
		float* dev_weight_matrix;
		float* dev_hidden_matrix;
		cudaMalloc(&dev_input_matrix, input_rows * input_cols * sizeof(float));
		cudaMalloc(&dev_weight_matrix, ih_weight_rows * ih_weight_cols * sizeof(float));
		cudaMalloc(&dev_hidden_matrix, hidden_rows * hidden_cols * sizeof(float));

		// Work work work

		// Free memory
		cudaFree(dev_input_matrix);
		cudaFree(dev_weight_matrix);
		cudaFree(dev_hidden_matrix);

		free(input_matrix);
		free(weight_matrix);
		free(hidden_matrix);

		return;
	}

	void matrixMul(const Matrix* A, const Matrix* B, Matrix* C) {
		const float alpha = 1.0f;
		const float beta = 0.0f;

		// Create
		cublasHandle_t ch;
		cublasCreate(&ch);

		// Do a Matrix Multiply
		cublasSgemm(
			ch, 
			CUBLAS_OP_N, 
			CUBLAS_OP_N, 
			A->rowcnt, 
			B->colcnt, 
			A->colcnt, 
			&alpha, 
			A->dev_data,
			A->rowcnt,
			B->dev_data,
			B->rowcnt, 
			&beta, 
			C->dev_data,
			A->rowcnt);

		// Destroy
		cublasDestroy(ch);
	}

	Matrix::Matrix(int colcnt, int rowcnt) : colcnt(colcnt), rowcnt(rowcnt)
	{
		this->cpuAlloc();
		for (int i = 0; i < this->getLen(); i++) {
			this->cpu_data[i] = 0;
		}

		this->devAlloc();

	}

	Matrix::~Matrix()
	{
		this->cpuFree();
		this->devFree();
	}

	void Matrix::cpuAlloc()
	{
		cpu_data = (float*)malloc(rowcnt * colcnt * sizeof(float));
		if (dev_data == NULL) {
			throw std::runtime_error("Failed to allocate cpu_data for Matrix!");
		}
	}

	void Matrix::devAlloc()
	{
		cudaMalloc(&dev_data, rowcnt * colcnt * sizeof(float));
		checkCUDAError("Failed to allocate dev_data for Matrix!");
	}

	void Matrix::cpuFree()
	{
		if (cpu_data) {
			free(cpu_data);
		}
	}

	void Matrix::devFree()
	{
		if (dev_data) {
			cudaFree(dev_data);
		}
	}

	void Matrix::copyCpuToDev()
	{
		cudaMemcpy(this->dev_data, this->cpu_data, this->getLen() * sizeof(float), ::cudaMemcpyHostToDevice);
	}

	void Matrix::copyDevToCpu()
	{
		cudaMemcpy(this->cpu_data, this->dev_data, this->getLen() * sizeof(float), ::cudaMemcpyDeviceToHost);
	}

	int Matrix::getLen()
	{
		return rowcnt * colcnt;
	}

	ImageFile::ImageFile(std::string filepath) : fd(0)
	{
		fd = std::fopen(filepath.c_str(), "r");
	}

	ImageFile::~ImageFile()
	{
		std::fclose(fd);
	}

	void ImageFile::readImage(Matrix* m)
	{
		// Format of Image File
		// filename\r\n
		// num_pixels\r\n
		//  pixels_0_255 ... pixels_0_255\r\n // Note leading space and list is space-delimited.

		int bytes_read = 0;

		bytes_read += std::fscanf(this->fd, "%i", &this->expected_number);
		bytes_read += std::fscanf(this->fd, "%i", &this->pixels);
		
		for (int i = 0; i < pixels; i++) {
			int tmp = 0;
			bytes_read += std::fscanf(this->fd, "%i", &tmp);
			m->cpu_data[i] = (float)(tmp / 255);
		}

		return;
	}

	int ImageFile::getExpectedNumber()
	{
		return this->expected_number;
	}

	Perceptron::Perceptron(int pixels, int outputs) :
		inputData(pixels, 1),
		hiddenLayer(pixels, 1),
		outputLayer(outputs, 1),
		ihWeights(pixels, pixels),
		hoWeights(pixels, outputs)
	{
	}

	Perceptron::~Perceptron()
	{
	}

	void Perceptron::randomizeWeights()
	{
		// kernRandomizeMatrix
	}

	void Perceptron::loadBrain(std::string brainfile)
	{
		// readFile into Matrix
		// copy into correct matricies
	}

	void Perceptron::saveBrain(std::string brainfile)
	{
		// Read matricxies
		// Output to file as a format to be defined
	}

	void Perceptron::loadTrainingDataSet(int expected_result, Matrix * input)
	{
		// Load data and store the expected result
	}

	void Perceptron::train(int iterations)
	{
		// Run the machine on the data set 'iteratations' times
		// Includes backprop
	}

	void Perceptron::loadDataSet(Matrix * input)
	{
		// Load a data set to run
	}

	void Perceptron::run()
	{
		// Run the machine on the data set.
	}

	int Perceptron::getLastResult()
	{
		// Get the result of the last run.
	}

	// TODO: implement required elements for MLP sections 1 and 2 here
}
