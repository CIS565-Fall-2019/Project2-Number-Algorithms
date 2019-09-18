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

constexpr int BLOCKSIZE = 1024;

namespace CharacterRecognition {
    using Common::PerformanceTimer;
    PerformanceTimer& timer()
    {
        static PerformanceTimer timer;
        return timer;
    }

	void initCublas()
	{
		cublasStatus_t status;
		status = cublasCreate(&ch);
		if (status != CUBLAS_STATUS_SUCCESS) {
			std::cout << "Failed to initialize cublas: " << status << std::endl;
		}
	}

	void deleteCublas()
	{
		cublasStatus_t status;
		status = cublasDestroy(ch);
		if (status != CUBLAS_STATUS_SUCCESS) {
			std::cout << "Failed to destroy cublas: " << status << std::endl;
		}
	}

	void matrixMul(cublasHandle_t ch, const Matrix* A, const Matrix* B, Matrix* C) {
		cublasStatus_t status;
		const float alpha = 1.0f; // Factor to multiply A by
		const float beta = 0.0f; // Factor to multiply C by prior to result.

		assert(A->colcnt == B->rowcnt);
		assert(A->rowcnt == C->rowcnt);
		assert(B->colcnt == C->colcnt);

		// Do a Matrix Multiply
		status = cublasSgemm(
			ch, 
			CUBLAS_OP_N, 
			CUBLAS_OP_N, 
			A->rowcnt, 
			B->colcnt, 
			B->rowcnt, 
			&alpha, 
			A->dev_data,
			A->rowcnt,
			B->dev_data,
			B->rowcnt,
			&beta, 
			C->dev_data,
			A->rowcnt);

		if (status != CUBLAS_STATUS_SUCCESS) {
			std::cout << "Failed to perform matrix multiply: " << status << std::endl;
		}

		cudaDeviceSynchronize();
		checkCUDAError("Failed matrixMul");
	}

	void matrixMul(cublasHandle_t ch, const Matrix* A, cublasOperation_t aop, const Matrix* B, cublasOperation_t bop, Matrix* C) {
		const float alpha = 1.0f; // Factor to multiply A by
		const float beta = 0.0f; // Factor to multiply C by prior to result.

		//assert(A->colcnt == B->rowcnt);
		//assert(A->rowcnt == C->rowcnt);
		//assert(B->colcnt == C->colcnt);

		// Do a Matrix Multiply
		cublasSgemm(
			ch,
			aop,
			bop,
			A->rowcnt,
			B->colcnt,
			B->rowcnt,
			&alpha,
			A->dev_data,
			A->rowcnt,
			B->dev_data,
			B->rowcnt,
			&beta,
			C->dev_data,
			A->rowcnt);

		cudaDeviceSynchronize();
		checkCUDAError("Failed matrixMul");
	}

	void matrixSub(cublasHandle_t ch, const Matrix* A, const Matrix* B, Matrix* C) {
		const float alpha = 1.0f; // Factor to multiply A by
		const float beta = -1.0f; // Factor to multiply B by

		assert(A->colcnt == B->colcnt);
		assert(A->rowcnt == B->rowcnt);
		assert(A->colcnt == C->colcnt);
		assert(A->rowcnt == C->rowcnt);

		// Do a Matrix Subtraction
		cublasSgeam(
			ch,
			CUBLAS_OP_N,
			CUBLAS_OP_N,
			A->rowcnt,
			B->colcnt,
			&alpha,
			A->dev_data,
			A->rowcnt,
			&beta,
			B->dev_data,
			B->rowcnt,
			C->dev_data,
			A->rowcnt
		);

		cudaDeviceSynchronize();
		checkCUDAError("Failed matrixSub");
	}

	void matrixAdd(cublasHandle_t ch, const Matrix* A, const Matrix* B, Matrix* C) {
		const float alpha = 1.0f; // Factor to multiply A by
		const float beta = 1.0f; // Factor to multiply B by

		assert(A->colcnt == B->colcnt);
		assert(A->rowcnt == B->rowcnt);
		assert(A->colcnt == C->colcnt);
		assert(A->rowcnt == C->rowcnt);

		// Do a Matrix Subtraction
		cublasSgeam(
			ch,
			CUBLAS_OP_N,
			CUBLAS_OP_N,
			A->rowcnt,
			B->colcnt,
			&alpha,
			A->dev_data,
			A->rowcnt,
			&beta,
			B->dev_data,
			B->rowcnt,
			C->dev_data,
			A->rowcnt
		);

		cudaDeviceSynchronize();
		checkCUDAError("Failed matrixAdd");
	}

	Matrix::Matrix(int colcnt, int rowcnt) : colcnt(colcnt), rowcnt(rowcnt)
	{
		this->cpuAlloc();
		for (int i = 0; i < this->getLen(); i++) {
			this->cpu_data[i] = 0;
		}

		this->devAlloc();
		this->copyCpuToDev();
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
			checkCUDAError("Failed to free dev_data for Matrix!");
		}
	}

	void Matrix::copyCpuToDev()
	{
		cudaMemcpy(this->dev_data, this->cpu_data, this->getLen() * sizeof(float), ::cudaMemcpyHostToDevice);
		checkCUDAError("Failed to memcpy in copyCpuToDev()");
	}

	void Matrix::copyDevToCpu()
	{
		cudaMemcpy(this->cpu_data, this->dev_data, this->getLen() * sizeof(float), ::cudaMemcpyDeviceToHost);
		checkCUDAError("Failed to memcpy in copyDevToCpu()");
	}

	void Matrix::copyMatrix(Matrix * m)
	{
		assert(this->colcnt == m->colcnt);
		assert(this->rowcnt == m->rowcnt);

		memcpy(this->cpu_data, m->cpu_data, m->getLen() * sizeof(float));
		cudaMemcpy(this->dev_data, m->dev_data, m->getLen() * sizeof(float), ::cudaMemcpyDeviceToDevice);
		checkCUDAError("Failed to memcpy in copyMatrix()");
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

		pixels = std::min(pixels, PIXELS);
		
		for (int i = 0; i < pixels; i++) {
			int tmp = 0;
			bytes_read += std::fscanf(this->fd, "%i", &tmp);
			m->cpu_data[i] = ((float)tmp / 255.0f);
		}

		return;
	}

	int ImageFile::getExpectedNumber()
	{
		return this->expected_number;
	}

	Perceptron::Perceptron(int pixels, int outputs) :
		inputLayer(pixels, 1),
		hiddenLayer(pixels / 5, 1),
		outputLayer(outputs, 1),
		expectedLayer(outputs, 1),
		kjWeights(pixels / 5, pixels),
		jiWeights(outputs, pixels / 5),
		kjWeightsDelta(pixels / 5, pixels),
		jiWeightsDelta(outputs, pixels / 5),
		jiOmega(outputs, 1),
		jiPsi(outputs, 1),
		jiTheta(outputs, 1),
		kjTheta(pixels / 5, 1),
		kjOmega(pixels / 5, 1),
		kjPsi(pixels / 5, 1),
		result(0),
		tr_runs(0.0f)
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

		// Synchronize
		cudaDeviceSynchronize();

		// Cleanup
		curandDestroyGenerator(gen);
	}

	void Perceptron::loadBrain(std::string brainfile)
	{
		// readFile into Matrix
		// copy into correct matricies
		// TODO
	}

	void Perceptron::saveBrain(std::string brainfile)
	{
		// Read matricxies
		// Output to file as a format to be defined
		// TODO
	}

	void Perceptron::loadTrainingDataSet(ImageFile * input)
	{
		// Load data 
		loadDataSet(input);

		// Update expected layer for training
		for (int i = 0; i < expectedLayer.getLen(); i++) {
			expectedLayer.cpu_data[i] = 0;
		}
		expectedLayer.cpu_data[input->getExpectedNumber()] = 1.0f;
	}

	void Perceptron::loadDataSet(ImageFile * input)
	{
		// Load data and store the expected result
		input->readImage(&this->inputLayer);
		inputLayer.copyCpuToDev();
	}

	// Mostly for debug
	void Perceptron::updateCpu() {
		inputLayer.copyDevToCpu();
		hiddenLayer.copyDevToCpu();
		outputLayer.copyDevToCpu();
		kjWeights.copyDevToCpu();
		jiWeights.copyDevToCpu();
		kjWeightsDelta.copyDevToCpu();
		jiWeightsDelta.copyDevToCpu();
	}

	void Perceptron::impl_run(bool training)
	{
		// Run the machine on the data set.
		matrixMul(ch, &inputLayer, &kjWeights, &hiddenLayer); // Step 1) Calculate values of hidden layer.
		if (training) {
			kjTheta.copyMatrix(&hiddenLayer); // Step 1.1) Save off hidden layer before sigmoids for backprop
		}

		reluActivate(&hiddenLayer); // STEP 2) Apply activation function
		matrixMul(ch, &hiddenLayer, &jiWeights, &outputLayer); // Step 3) Hidden layer now populated, get output layer
		if (training) {
			jiTheta.copyMatrix(&outputLayer); // Step 3.1) Save off output layer before sigmoids for backprop
		}

		outputLayer.copyDevToCpu();

		softmaxActivate(&outputLayer); // Step 4) Apply activation to output layers

		// Setp 5) Store the result, ie the brightest node in the output layer
		// Output layer is small, so do this on CPU.
		outputLayer.copyDevToCpu();
		result = std::max_element(
			outputLayer.cpu_data,
			outputLayer.cpu_data + outputLayer.getLen()
		) - outputLayer.cpu_data;

		// Inc. Run Counter for Backprop
		if (training) {
			this->tr_runs++;
		}
	}

	void Perceptron::run()
	{
		impl_run(false);
	}

	void Perceptron::train()
	{
		impl_run(true);
	}

	int Perceptron::getLastResult()
	{
		// Get the result of the last run.
		return result;
	}

	void Perceptron::updateBackprop()
	{
		// Backprop algoritm runs in two phases.
		// From the output, compute the deltas that should be made to the jiWeights
		// Then, from there, calculate the deltas that should be applied to the kjWeights

		// 1.0) Calculate delat to ji Weights
		// 1.1) Calculate iTheta ... Done during run()
		matrixSub(ch, &expectedLayer, &outputLayer, &jiOmega); // 1.2) Calculate iOmega ... Done by subtracting expectedLayer - outputLayer
		calcPsi(&jiOmega, &jiTheta, &jiPsi); // 1.3) Calculate iPsi ... a little fancier
		calcDeltaChange(0.01f, &hiddenLayer, &jiPsi, &jiWeightsDelta); // 1.4) Lastly, calculate the delta to each weight.

		// 2.0) Now repeat for the kj Weights
		calcOmega(&jiPsi, &jiWeights, &kjOmega); // This omega is done with a special function, unlike subtraction from last layer
		calcPsi(&kjOmega, &kjTheta, &kjPsi);
		calcDeltaChange(0.01f, &inputLayer, &kjPsi, &kjWeightsDelta);

		// Old way did not work, lets try from scratch...
	}

	void Perceptron::applyBackprop()
	{
		// Average over the number of runs
		float t = 1.0f / this->tr_runs;
		this->tr_runs = 0.0f; // Reset

		cublasSaxpy(ch, kjWeights.getLen(), &t, kjWeightsDelta.dev_data, 1, kjWeights.dev_data, 1);
		cublasSaxpy(ch, jiWeights.getLen(), &t, jiWeightsDelta.dev_data, 1, jiWeights.dev_data, 1);
	}

	__global__ void kernCalcPsi(int n, float * omega, float * theta, float * psi) {
		int idx = getGlobalIdx_3D_3D();
		if (idx < n) {
			psi[idx] = omega[idx] + devInverseSigmoid(theta[idx]);
		}
	}

	void Perceptron::calcPsi(const Matrix * omega, const Matrix * theta, Matrix * psi)
	{
		// Call a kernel to handle this.
		assert(omega->colcnt == theta->colcnt);
		assert(omega->rowcnt == theta->rowcnt);
		assert(omega->colcnt == psi->colcnt);
		assert(omega->rowcnt == psi->rowcnt);
		assert(omega->rowcnt == 1);

		int n = omega->colcnt;
		kernCalcPsi<<<1, n>>>(n, omega->dev_data, theta->dev_data, psi->dev_data);
		cudaDeviceSynchronize();
		checkCUDAError("Failed calcPsi");
	}

	__global__ void kernCalcDeltaChange(int n, float lambda, float * layer, float * psi, float * deltaOut) {
		int gloablId = getGlobalIdx_3D_3D();
		int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
		int threadId = (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;

		if (gloablId < n) {
			deltaOut[gloablId] += lambda * layer[threadId] * psi[blockId];
		}
	}

	void Perceptron::calcDeltaChange(const float lambda, const Matrix * leftLayer, const Matrix * psi, Matrix * weightDelta)
	{
		// Call a kernel to handle this
		int blocks = weightDelta->rowcnt;
		int threadsperblock = weightDelta->colcnt;
		int totalthreads = blocks * threadsperblock;

		kernCalcDeltaChange<<<blocks, threadsperblock >>>(totalthreads, 0.01f, leftLayer->dev_data, psi->dev_data, weightDelta->dev_data);
		cudaDeviceSynchronize();
		checkCUDAError("Failed calcDeltaChange");
	}

	void Perceptron::calcOmega(const Matrix * psi, const Matrix * weights, Matrix * omega)
	{
		// Transpose matrix since we are multiplying the other way
		matrixMul(ch, psi, CUBLAS_OP_N, weights, CUBLAS_OP_T,omega);
		checkCUDAError("Failed calcOmega");
	}

	void reluActivate(Matrix * m)
	{
		dim3 fullBlocksPerGrid((m->getLen() + BLOCKSIZE - 1) / BLOCKSIZE);
		kernReluActivate << <fullBlocksPerGrid, BLOCKSIZE >> > (m->getLen(), m->dev_data, m->dev_data);
		checkCUDAError("Failed reluActivate");
	}
	// Softmax involves exponentiating each value, summing them, and then dividing so that the sum of the vector is 1.
	void softmaxActivate(Matrix * m) {
		cublasStatus_t status;
		dim3 fullBlocksPerGrid((m->getLen() + BLOCKSIZE - 1) / BLOCKSIZE);
		float expSum = 0.0f;
		float invExpSum = 0.0f;

		// Prescale down
		// TODO: For some reason my values are huge at this stage (10,000~) and exponentiating them
		// is impossible. I scale everything down to make the system at least WORK.
		invExpSum = 0.0001f;
		status = cublasSscal(ch, m->getLen(), &invExpSum, m->dev_data, 1);
		if (status != CUBLAS_STATUS_SUCCESS) {
			std::cout << "Failed cublasSscal: " << status << std::endl;
		}

		// Exponentiate
		kernExponentiate << <fullBlocksPerGrid, BLOCKSIZE >> > (m->getLen(), m->dev_data, m->dev_data);
		checkCUDAError("Failed kernExponentiate");

		// Sum 
		status = cublasSasum(ch, m->getLen(), m->dev_data, 1, &expSum);
		if (status != CUBLAS_STATUS_SUCCESS) {
			std::cout << "Failed cublasSasum: " << status << std::endl;
		}

		// Normalize
		invExpSum = 1.0f / expSum;
		status = cublasSscal(ch, m->getLen(), &invExpSum, m->dev_data, 1);
		if (status != CUBLAS_STATUS_SUCCESS) {
			std::cout << "Failed cublasSscal: " << status << std::endl;
		}
	}

	//////////////////////////////////
	//////////////////////////////////
	// KERNEL OPERATIONS
	//////////////////////////////////
	//////////////////////////////////

	__device__ int getGlobalIdx_3D_3D() { 
		int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
		int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
		return threadId;
	}

	__global__ void kernReluActivate(int n, float* in, float* out) {
		int idx = getGlobalIdx_3D_3D();
		if (idx < n) {
			out[idx] = fmaxf(0.0f, in[idx]);
		}
	}

	__device__ float devSigmoid(float n) {
		return 1.0f / (1 + expf(-1.0f * n));
	}

	__device__ float devInverseSigmoid(float n) {
		return 1.0f / (1 + expf(n));
	}

	__global__ void kernExponentiate(int n, float* in, float* out) {
		int idx = getGlobalIdx_3D_3D();
		if (idx < n) {
			out[idx] = expf(in[idx]);
		}
	}

	void testMatrixMul() {
		Matrix m_a(4, 1);     // Input Values
		Matrix m_b(2, 4); // Weights
		Matrix m_c(2, 1);     // Output Values

		// Init matrix
		for (int i = 0; i < m_a.getLen(); i++) {
			m_a.cpu_data[i] = i + 1;
		}
		for (int i = 0; i < m_b.getLen(); i++) {
			m_b.cpu_data[i] = 2;
		}

		// Populate Device
		m_a.copyCpuToDev();
		m_b.copyCpuToDev();

		matrixMul(ch, &m_a, &m_b, &m_c);

		m_c.copyDevToCpu();
	}
}
