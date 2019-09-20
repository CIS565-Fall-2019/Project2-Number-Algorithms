#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <vector>
#include "common.h"
#include "mlp.h"

#include <memory>

#define THREADS_PER_BLOCK 256

namespace CharacterRecognition {
	using namespace std;
	using Common::PerformanceTimer;
	PerformanceTimer& timer()
	{
		static PerformanceTimer timer;
		return timer;
	}

	double learningRate;
	shared_ptr<Matrix> W1, W2, B1, B2, IN, H, Y, X;;
	int inputN, hiddenN, outputN;

	// Fill the array A(nr_rows_A, nr_cols_A) with random numbers on GPU
	void GPU_fill_rand(float* A, int nr_rows_A, int nr_cols_A) {
		curandGenerator_t prng;
		curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_XORWOW);
		curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());
		// Fill the array with random numbers on the device
		curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A);
		cudaDeviceSynchronize();
		checkCUDAError("GPU_fill_rand FAILED");
	}

	void GPU_fill_rand(Matrix* p_matrix) {
		GPU_fill_rand(p_matrix->dev_data, p_matrix->numRow, p_matrix->numCol);
	}

	__global__ void kernAdd(float* A, float* B, int n) {
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (idx >= n) return;
		A[idx] += B[idx];
	}

	__global__ void kernSubtract(float* A, const float* B, int n) {
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (idx >= n) return;
		A[idx] -= B[idx];
	}

	__global__ void kernSubtract(const float* A, const float* B, float*C, int n) {
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (idx >= n) return;
		C[idx] = A[idx] - B[idx];
	}

	__global__ void kernDiffSquare(const float* A, const float* B, float* C, int n) {
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (idx >= n) return;
		const float tmp = A[idx] - B[idx];
		C[idx] = tmp * tmp;
	}

	__global__ void generate_in_a_b(float* A, float a, float b, int N) {

		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (idx >= N) return;
		A[idx] = (b - a) * A[idx] + a;
	}

	__global__ void generate_random_numbers(float* numbers, int N) {

		int i = threadIdx.x + blockIdx.x * blockDim.x;
		if (i >= N) return;
		curandState state;
		curand_init(clock64(), i, 0, &state);
		numbers[i] = curand_uniform(&state);
	}

	__global__ void kernSigmoid(float* numbers, int N) {
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		if (i >= N) return;
		numbers[i] = 1.0 / (1.0 + std::exp(-numbers[i]));
	}

	__global__ void kernSigmoidPrime(float* numbers, int N) {
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		if (i >= N) return;
		float tmp = std::exp(-numbers[i]);
		numbers[i] = tmp / ( (1.f + tmp) * (1.f + tmp) );
	}

	__global__ void kernMultiply(float* numbers, float constant, int N) {
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		if (i >= N) return;
		numbers[i] *= constant;
	}

	__global__ void kernMultiplyMatrix(float* numbers, const float* other, int N) {
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		if (i >= N) return;
		numbers[i] *= other[i];
	}


	void Matrix::initWithRandom() {
		GPU_fill_rand(dev_data, numRow, numCol);
		int n = numRow * numCol;
		int blockSize = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
		float scale = 10.0f/std::sqrtf(n);	// TODO: find a better initialization value!!
		generate_in_a_b << <blockSize, THREADS_PER_BLOCK >> > (dev_data, -scale, scale, n);
	}

	void Matrix::initWithZero() {
		memset(data, 0, dataSize);
		cudaMemset(dev_data, 0, dataSize);
	}

	void Matrix::initWithTest() {
		for (size_t row = 0; row < numRow; row++) {
			for (size_t col = 0; col < numCol; col++) {
				data[row * numCol + col] = (float)(row * numCol + col + 1.0f) / 10.f;
			}
		}
		copyToDevice();
	}

	Matrix* Matrix::dot(const Matrix* other) const {
		// C(m, n) = A(m, k) * B(k, n)
		int m = numRow;
		int k = numCol;
		if (k != other->numRow) {
			throw "Matrices not match";
		}
		int n = other->numCol;

		Matrix* product = new Matrix(m, n);
	//	IMPORTANT!!
	//	cuBLAS uses column-major order, so reverse the order to get the correct result
	//	C: A*B, cuBLAS: B*A
		gpu_blas_mmul(other->dev_data, dev_data, product->dev_data, n, k, m);
		return product;
	}

	void Matrix::add(const Matrix* other) {
		int n = numRow * numCol;
		int blockSize = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
		kernAdd<<<blockSize, THREADS_PER_BLOCK>>>(dev_data, other->dev_data, n);
	}

	Matrix* Matrix::subtract(const Matrix* other) const{
		Matrix* output = new Matrix(numRow, numCol);
		int n = numRow * numCol;
		int blockSize = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
		kernSubtract << <blockSize, THREADS_PER_BLOCK >> > (dev_data, other->dev_data, output->dev_data, n);

		return output;
	}


	Matrix* Matrix::diffSquare(const Matrix* other) const {
		Matrix* output = new Matrix(numRow, numCol);
		int n = numRow * numCol;
		int blockSize = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
		kernDiffSquare << <blockSize, THREADS_PER_BLOCK >> > (dev_data, other->dev_data, output->dev_data, n);

		return output;
	}

	void Matrix::subtract_inplace(const Matrix* other) {
		int n = numRow * numCol;
		int blockSize = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
		kernSubtract << <blockSize, THREADS_PER_BLOCK >> > (dev_data, other->dev_data, n);
	}
	
	void Matrix::sigmoid() {
		int n = numRow * numCol;
		int blockSize = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
		kernSigmoid << <blockSize, THREADS_PER_BLOCK >> > (dev_data, n);
	}

	void Matrix::sigmoidePrime() {
		int n = numRow * numCol;
		int blockSize = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
		kernSigmoidPrime << <blockSize, THREADS_PER_BLOCK >> > (dev_data, n);
	}

	Matrix* Matrix::multiply(const float constant) {
		int n = numRow * numCol;
		int blockSize = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
		kernMultiply<< <blockSize, THREADS_PER_BLOCK >> > (dev_data, constant, n);
		return this;
	}

	Matrix* Matrix::multiply(const Matrix* other) {
		int n = numRow * numCol;
		int blockSize = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
		kernMultiplyMatrix << <blockSize, THREADS_PER_BLOCK >> > (dev_data, other->dev_data, n);
		return this;
	}

	Matrix* Matrix::transpose() const {
		Matrix* trans = new Matrix(numCol , numRow);
		int const m = numRow;
		int const n = numCol;
		float const alpha(1.0);
		float const beta(0.0);
		cublasHandle_t handle;
		cublasCreate(&handle);
		cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, &alpha, dev_data, n, &beta, trans->dev_data, m, trans->dev_data, m);
		cublasDestroy(handle);

		return trans;
	}


	void init(int inputNeuron, int hiddenNeuron, int outputNeuron, double rate)
	{
		inputN = inputNeuron, hiddenN = hiddenNeuron, outputN = outputNeuron;
		learningRate = rate;

		W1 = make_shared<Matrix>(inputNeuron, hiddenNeuron);
		W2 = make_shared<Matrix>(hiddenNeuron, outputNeuron);
		B1 = make_shared<Matrix>(1, hiddenNeuron);
		B2 = make_shared<Matrix>(1, outputNeuron);

		W1->initWithRandom();
		W2->initWithRandom();
		B1->initWithZero();
		B2->initWithZero();
	}

	Matrix* computeOutput(const vector<float> & input) {
		vector<vector<float>> wrapper = { input };
		X = make_shared<Matrix>(wrapper);
		X->copyToDevice();

		H = shared_ptr<Matrix>( X->dot(W1.get()) );
		
		H->add(B1.get());
		H->sigmoid();

		Y = shared_ptr<Matrix>( H->dot(W2.get()) );
		Y->add(B2.get());
		Y->sigmoid();

		return Y.get();
	}

	float learn(const vector<float> expectedOutput) {
		// Compute gradients
		//dJdB2 = Y.subtract(Y2).multiply( H.dot(W2).add(B2).applyFunction(sigmoidePrime) );
		vector<vector<float>> wrapper = { expectedOutput };
		auto Y2 = std::make_unique<Matrix>(wrapper); // 1 x numOutput matrix

		auto dJdB2 = unique_ptr<Matrix>( Y->subtract(Y2.get()) );
		auto tmpH_W2 = unique_ptr<Matrix>( H->dot(W2.get()) );
		tmpH_W2->add(B2.get());
		tmpH_W2->sigmoidePrime();
		dJdB2->multiply(tmpH_W2.get());
		dJdB2->copyToHost();

		// dJdB1 = dJdB2.dot(W2.transpose()).multiply(X.dot(W1).add(B1).applyFunction(sigmoidePrime));
		auto W2_trans = unique_ptr<Matrix>( W2->transpose() );
		auto dJdB1 = unique_ptr<Matrix>( dJdB2->dot(W2_trans.get()) );

		auto tmpX_W1 = unique_ptr<Matrix>( X->dot(W1.get()) );
		tmpX_W1->add(B1.get());
		tmpX_W1->sigmoidePrime();	// X.dot(W1).add(B1).applyFunction(sigmoidePrime)
		dJdB1->multiply(tmpX_W1.get());

		// dJdW2 = H.transpose().dot(dJdB2);
		auto dJdW2_tmp = unique_ptr<Matrix>( H->transpose() );
		auto dJdW2 = unique_ptr<Matrix>( dJdW2_tmp->dot(dJdB2.get()) );

		// dJdW1 = X.transpose().dot(dJdB1);
		auto dJdW1_tmp = unique_ptr<Matrix>( X->transpose() );
		auto dJdW1 = unique_ptr<Matrix>( dJdW1_tmp->dot(dJdB1.get()) );

		// update weights
		W1->subtract_inplace(dJdW1->multiply(learningRate));
		W2->subtract_inplace(dJdW2->multiply(learningRate));
		B1->subtract_inplace(dJdB1->multiply(learningRate));
		B2->subtract_inplace(dJdB2->multiply(learningRate));

		// Calculate cost
		auto diffSqr = unique_ptr<Matrix>(Y2->diffSquare(Y.get()));
		thrust::device_ptr<float> thrust_diffSqr(diffSqr->dev_data);
		int n = diffSqr->numCol * diffSqr->numCol;
		float cost = thrust::reduce(thrust_diffSqr, thrust_diffSqr + n, 0.f, thrust::plus<float>());
		cost = std::sqrtf(cost / n);

		return cost;
	}

	void unitTest()
	{
		// Test transpose
		Matrix m(3, 2);
		Matrix n(2, 2);

		m.initWithTest();
		n.initWithTest();

		m.print();
		n.print();

		//Matrix* pt = m.transpose();
		//pt->copyToHost();
		//pt->print();

		// Test multiply
		//m.multiply(3.0f);
		//m.copyToHost();
		//m.print();

		// Test sigmoid
		//m.sigmoid();
		//m.copyToHost();
		//m.print();

		// Test Dot
		Matrix* p = m.dot(&n);
		p->copyToHost();
		p->print();

	}

	// Took from 
	// https://solarianprogrammer.com/2012/05/31/matrix-multiplication-cuda-cublas-curand-thrust/
	// Multiply the arrays A and B on GPU and save the result in C
	// C(m,n) = A(m,k) * B(k,n)
	void gpu_blas_mmul(const float* A, const float* B, float* C, const int m, const int k, const int n) {
		int lda = m, ldb = k, ldc = m;
		const float alf = 1;
		const float bet = 0;
		const float* alpha = &alf;
		const float* beta = &bet;

		// Create a handle for CUBLAS
		cublasHandle_t handle;
		cublasCreate(&handle);

		// Do the actual multiplication
		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

		// Destroy the handle
		cublasDestroy(handle);
	}
}
