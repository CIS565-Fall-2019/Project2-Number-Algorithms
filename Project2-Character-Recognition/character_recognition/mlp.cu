#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <cublas_v2.h>
#include <curand.h>
#include "common.h"
#include "mlp.h"

namespace CharacterRecognition {
    using Common::PerformanceTimer;
    PerformanceTimer& timer()
    {
        static PerformanceTimer timer;
        return timer;
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
		kjWeights(pixels, pixels),
		jiWeights(pixels, outputs)
	{
	}

	Perceptron::~Perceptron()
	{
	}

	void Perceptron::randomizeWeights()
	{
		// Create an RNG via curand and then populate the weights array with those numbers.
		curandGenerator_t gen;

		// Create and seed generator
		curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
		curandSetPseudoRandomGeneratorSeed(gen, ::time(NULL));

		// Populate weight matricies
		curandGenerateUniform(gen, this->kjWeights.dev_data, this->kjWeights.getLen());
		curandGenerateUniform(gen, this->jiWeights.dev_data, this->jiWeights.getLen());

		// Cleanup
		curandDestroyGenerator(gen);
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

	void Perceptron::train()
	{
		// Training: We want to run our data, calculate the backprop variables,
		// and average the results over many runs.
		// So what we do, we run the machine and then add a step to collect information
		// about the weights and activations of the machine. These will then be stored off
		// in a seperate array.
		// After running, the calcualted weights can either be fed back in and re-trained
		// or output to a file for the user to recover.

		// 1) Run the Perceptron over the input data
		this->run();

		// 2) Collect backprop information.
		this->backprop();
	}

	void Perceptron::loadDataSet(Matrix * input)
	{
		// Load a data set to run
	}

	void Perceptron::run()
	{
		// Run the machine on the data set.
		// Step 1) Calculate values of hidden layer.
		matrixMul(&inputData, &kjWeights, &hiddenLayer);

		// Step 2) Apply Hidden Layer Bias
		// TODO: Would be nice.

		// STEP 3) Apply sigmoid function
		sigmoid(&hiddenLayer);

		// Step 4) Hidden layer now populated, get output layer
		matrixMul(&hiddenLayer, &jiWeights, &outputLayer);

		// Step 5) Apply sigmoid to output layers
		sigmoid(&outputLayer);

		// Setp 6) Store the result, ie the brightest node in the output layer
		// Output layer is small, so do this on CPU.
		outputLayer.copyDevToCpu();
		result = std::max_element(
			outputLayer.cpu_data,
			outputLayer.cpu_data + outputLayer.getLen()
		) - outputLayer.cpu_data;
	}

	int Perceptron::getLastResult()
	{
		// Get the result of the last run.
	}

	void Perceptron::backprop()
	{
		// Backprop algoritm runs in two phases.
		// From the output, compute the deltas that should be made to the jiWeights
		// Then, from there, calculate the deltas that should be applied to the kjWeights

		// 1) jiWeights

	}

	void Perceptron::updateHiddenToOutputWeights()
	{

	}

	//////////////////////////////////
	//////////////////////////////////
	// KERNEL OPERATIONS
	//////////////////////////////////
	//////////////////////////////////

	void sigmoid(Matrix* m) {
		// TODO: Optimize block utilization
		int threads = m->getLen();
		kernSigmoid << <1, threads >> > (m->getLen(), m->dev_data);
	}

	__global__ void kernSigmoid(int n, float* data) {
		int idx = threadIdx.x + (blockDim.x) * threadIdx.y + (blockDim.y * blockDim.x) * threadIdx.z;
		if (idx > n) {
			return;
		}

		data[n] = devSigmoid(data[n]);
	}

	void activation(Matrix* m) {
		// TODO: Optimize block utilization
		int threads = m->getLen();
		kernActivation << <1, threads >> > (m->getLen(), m->dev_data);
	}

	__global__ void kernActivation(int n, float* data) {
		int idx = threadIdx.x + (blockDim.x) * threadIdx.y + (blockDim.y * blockDim.x) * threadIdx.z;
		if (idx > n) {
			return;
		}

		data[n] = devInverseSigmoid(data[n]);
	}

	__device__ float devSigmoid(float n) {
		return 1.0f / (1 + expf(-1.0f * n));
	}

	__device__ float devInverseSigmoid(float n) {
		return 1.0f / (1 + expf(n));
	}

	__global__ void kernUpdateHiddenToOutputWeights(int n, const float* jiWeights, const float* outputLayer, const float* hiddenLayer,  const float* expectedLayer, float* jiWeightsDelta) {
		int tidx = threadIdx.x + (blockDim.x) * threadIdx.y + (blockDim.y * blockDim.x) * threadIdx.z;
		int bidx = blockIdx.x + (gridDim.x) * blockIdx.y + (gridDim.y * gridDim.x) * blockIdx.z;
		int idx = tidx + bidx * (blockDim.x * blockDim.y * blockDim.z);
		if (idx > n) {
			return;
		}

		// Leverage architecture to our advantage
		// Uses I blocks of J threads
		int j = tidx;
		int i = bidx;
		int rowcount = blockDim.x;

		float lambda = 1.0f; // TODO: Make this adjustable, set it to E/10, where E is ???
		float theta = -logf((1.0f/hiddenLayer[j]) - 1);
		float omega = expectedLayer[i] - outputLayer[i];
		float psi = omega * devInverseSigmoid(theta);
		
		jiWeightsDelta[j + i * rowcount] += lambda * hiddenLayer[j] * psi;
	}

	__global__ void kernUpdateInputToHiddenWeights(int n, const float* kjWeights, const float* hiddenLayer, const float* inputLayer, float* kjWeightsDelta) {
		int tidx = threadIdx.x + (blockDim.x) * threadIdx.y + (blockDim.y * blockDim.x) * threadIdx.z;
		int bidx = blockIdx.x + (gridDim.x) * blockIdx.y + (gridDim.y * gridDim.x) * blockIdx.z;
		int idx = tidx + bidx * (blockDim.x * blockDim.y * blockDim.z);
		if (idx > n) {
			return;
		}

		// Leverage architecture to our advantage
		// Uses I blocks of J threads
		int k = tidx;
		int j = bidx;
		int rowcount = blockDim.x;

		//float lambda = 1.0f; // TODO: Make this adjustable, set it to E/10, where E is ???
		//float theta = -logf((1.0f / hiddenLayer[j]) - 1);
		//float omega = expectedLayer[i] - outputLayer[i];
		//float psi = omega * devInverseSigmoid(theta);

		//jiWeightsDelta[j + i * rowcount] += lambda * hiddenLayer[j] * psi;
	}
}
