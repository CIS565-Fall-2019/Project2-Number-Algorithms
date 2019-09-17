#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "mlp.h"
#include <cublas_v2.h>
#include <curand.h>
#include<memory.h>
#include<iostream>

#define blockSize 128

cublasHandle_t handle;
namespace CharacterRecognition {
    using Common::PerformanceTimer;
    PerformanceTimer& timer()
    {
        static PerformanceTimer timer;
        return timer;
    }

	__global__ void kernAddVectors(int n, float* g, float* bias, float* result) {
		int index = (blockIdx.x*blockDim.x) + threadIdx.x;
		if (index >= n)
			return;
		result[index] = g[index] + bias[index];
	}

	__global__ void kernUpdateParameters(int n, float *input, float *grad, float alpha) {
		int index = (blockIdx.x*blockDim.x) + threadIdx.x;
		if (index >= n)
			return;
		input[index] = input[index] - alpha * grad[index];
	}

	__global__ void kernSubVectors(int n, float* y, float* yhat, float* result) {
		int index = (blockIdx.x*blockDim.x) + threadIdx.x;
		if (index >= n)
			return;
		result[index] = yhat[index] - y[index];
	}

	__global__ void kernInitBiasVectors(int n, float* b, float value) {
		int index = (blockIdx.x*blockDim.x) + threadIdx.x;
		if (index >= n)
			return;
		b[index] = value;
	}

	__global__ void kernUpSweep(int n, int d, float *itemp) {
		int k = (blockIdx.x*blockDim.x) + threadIdx.x;
		if (k > (n - 1)) {
			return;
		}
		int power = 1 << (d + 1);
		int power_2 = 1 << d;
		if (k % power == 0 && k + power - 1 < n && k + power_2 - 1 < n)
			itemp[k + power - 1] += itemp[k + power_2 - 1];
	}

	__global__ void kernSoftmaxActivation(int n, float *g, float *output, float exp_sum) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;

		if (index >= n)
			return;
		output[index] = expf(g[index]) / exp_sum;
	}

	__global__ void kernSoftmaxDerivative(int n, float *input, float *grad, float exp_sum) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;

		if (index >= n)
			return;
		grad[index] = (exp_sum*expf(input[index]) - expf(input[index]) * expf(input[index])) / (exp_sum * exp_sum);
	}

	__global__ void kernReluActivationForward(int n, float* g, float* a) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;

		if (index >= n)
			return; 
		a[index] = fmaxf(g[index], 0);
	}

	__global__ void kernReluDerivative(int n, float* input, float* grad) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;

		if (index >= n)
			return;
		grad[index] = (input[index] > 0) ? 1 : 0;
	}

	__global__ void kernCopyVectors(int n, float *g, float *output) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= n)
			return;
		output[index] = expf(g[index]);
	}

	__global__ void kernDerivativeLoss(int n, float *y, float *y_pred, float *output) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= n) {
			return;
		}
		output[index] = -y[index] / y_pred[index] + (1 - y[index]) / (1 - y_pred[index]);
	}
	__global__ void kernElementWiseMultiplication(int n, float *input1, float *input2, float *output) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= n) {
			return;
		}
		output[index] = input1[index] * input2[index];
	}

	void random_init(float * A, int rows, int cols) {
		curandGenerator_t prng;
		curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
		curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());
		curandGenerateNormal(prng, A, rows * cols, 0, 2.0/rows);
	}

	//C(m,n) = A(m,k)*B(k,n)
	void mmul(const float* A, const float* B, float* C, const int m, const int k, const int n, int a_trans_flag, int b_trans_flag
		, int lda, int ldb, int ldc) {
		const float alf = 1;
		const float bet = 0;
		const float *alpha = &alf;
		const float *beta = &bet;
		std::cout << a_trans_flag << " " << b_trans_flag << std::endl;
		if(a_trans_flag == 0 && b_trans_flag == 0)
			cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
		else if(a_trans_flag == 0 && b_trans_flag == 1)
			cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
		else if(a_trans_flag == 1 && b_trans_flag == 0)
			cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	}

	void printCuda(float *a1, int n, std::string name) {
		float *print_a = new float[n];
		std::cout << name.c_str() << std::endl;
		std::cout << "{" << std::endl;
		cudaMemcpy(print_a, a1, n * sizeof(float), cudaMemcpyDeviceToHost);
		for (int i = 0; i < n; i++) {
			std::cout << "\t" << print_a[i] << std::endl;
		}
		std::cout << "}" << std::endl;
		delete[]print_a;
	}

	NeuralNet::NeuralNet(int input_size, int classes, vector<int>layers) {
		layer_sizes.push_back(input_size);

		// Set all layer sizes
		for (int i = 0; i < layers.size(); i++)
			layer_sizes.push_back(layers[i]);
		layer_sizes.push_back(classes);
		// Temporary variables to be pushed;
		float *z_t, *dz_t, *a_t, *da_t, *w_t, *dw_t, *b_t, *db_t, *ghat_t;
		// Some dummy mallocs to be pushed for the 0th(input) layer
		// We treat a0 as the input layer.
		cudaMalloc((void**)&z_t, sizeof(float));
		checkCUDAError("Cuda Malloc for z failed.");
		z.push_back(z_t);

		cudaMalloc((void**)&dz_t, sizeof(float));
		checkCUDAError("Cuda Malloc for dz failed.");
		dz.push_back(dz_t);

		cudaMalloc((void**)&a_t, layer_sizes[0] * 1 * sizeof(float));
		checkCUDAError("Cuda Malloc for a failed.");
		a.push_back(a_t);

		cudaMalloc((void**)&da_t, layer_sizes[0] * 1 * sizeof(float));
		checkCUDAError("Cuda Malloc for da failed.");
		da.push_back(da_t);

		cudaMalloc((void**)&w_t, sizeof(float));
		checkCUDAError("Cuda Malloc for weights failed.");
		w.push_back(w_t);

		cudaMalloc((void**)&dw_t, sizeof(float));	
		checkCUDAError("Cuda Malloc for derivative of weights failed.");
		dw.push_back(dw_t);
	
		cudaMalloc((void**)&b_t, sizeof(float));
		checkCUDAError("Cuda Malloc for bias failed.");
		b.push_back(b_t);

		cudaMalloc((void**)&db_t, sizeof(float));
		checkCUDAError("Cuda Malloc for derivatives of bias failed.");
		db.push_back(db_t);

		cudaMalloc((void**)&ghat_t, sizeof(float));
		checkCUDAError("Cuda Malloc for derivatives of bias failed.");
		ghat.push_back(ghat_t);


		// The following loop allocates sizes to all the weights, bias, a and z vectors and their gradients.
		for (int i = 1; i < layer_sizes.size(); i++) {

			cudaMalloc((void**)&z_t, layer_sizes[i] * 1 * sizeof(float));
			checkCUDAError("Cuda Malloc for z failed.");
			z.push_back(z_t);

			cudaMalloc((void**)&dz_t, layer_sizes[i] * 1 * sizeof(float));
			checkCUDAError("Cuda Malloc for dz failed.");
			dz.push_back(dz_t);

			cudaMalloc((void**)&a_t, layer_sizes[i] * 1 * sizeof(float));
			checkCUDAError("Cuda Malloc for a failed.");
			a.push_back(a_t);

			cudaMalloc((void**)&da_t, layer_sizes[i] * 1 * sizeof(float));
			checkCUDAError("Cuda Malloc for da failed.");
			da.push_back(da_t);

			cudaMalloc((void**)&w_t, layer_sizes[i] * layer_sizes[i - 1] * sizeof(float));
			checkCUDAError("Cuda Malloc for weights failed.");
			w.push_back(w_t);

			cudaMalloc((void**)&dw_t, layer_sizes[i] * layer_sizes[i - 1] * sizeof(float));
			checkCUDAError("Cuda Malloc for derivative of weights failed.");
			dw.push_back(dw_t);

			cudaMalloc((void**)&b_t, layer_sizes[i] * 1 * sizeof(float));
			checkCUDAError("Cuda Malloc for bias failed.");
			b.push_back(b_t);

			cudaMalloc((void**)&db_t, layer_sizes[i] * 1 * sizeof(float));
			checkCUDAError("Cuda Malloc for derivatives of bias failed.");
			db.push_back(db_t);

			cudaMalloc((void**)&ghat_t, layer_sizes[i] * 1 * sizeof(float));
			checkCUDAError("Cuda Malloc for derivatives of activations failed");
			ghat.push_back(ghat_t);
			
		}
		// Avoid those memory leaks :)
		cudaFree(z_t);
		cudaFree(dz_t);
		cudaFree(a_t);
		cudaFree(da_t);
		cudaFree(w_t);
		cudaFree(dw_t);
		cudaFree(b_t);
		cudaFree(db_t);
		cudaFree(ghat_t);

		dim3 fullBlocksPerGrid;
		// The following for loop initializes weights according to normal distribution 
		// We are using he-normal initialization here because of ReLU activation function
		for (int i = 1; i < layer_sizes.size(); i++) {
		//	fullBlocksPerGrid = ((layer_sizes[i]*layer_sizes[i-1] + blockSize - 1) / blockSize);
		//	kernInitBiasVectors << <fullBlocksPerGrid, blockSize >> > (layer_sizes[i] * layer_sizes[i-1] , w[i], 0);
			random_init(w[i], layer_sizes[i], layer_sizes[i - 1]);
		}
		// The following loop initializes the bias to a small value.
		// It invokes a kernel which fills the bias vector with the desired value
		for (int i = 1; i < layer_sizes.size(); i++) {
			fullBlocksPerGrid = ((layer_sizes[i] + blockSize - 1) / blockSize);
			kernInitBiasVectors << <fullBlocksPerGrid, blockSize >> > (layer_sizes[i], b[i], 0.1);
		}
		// Create a cublas handle for matrix multiplication
		cublasCreate(&handle);
	}

	
	float* NeuralNet::forward(float *input) {

		// a^[0] will be the input
		cudaMemcpy(a[0], input, layer_sizes[0] * sizeof(float), cudaMemcpyHostToDevice);
		// The activation for every layer but the last is relu, so the steps will be the same.
		// The equations here are
		//z[l] = w[l]a[l-1] + b[l]
		// a[l] = relu(z[l])
		dim3 fullBlocksPerGrid;
		int L = layer_sizes.size() - 1;
		for (int i = 1; i < L; i++) {
			// Do the matrix multiplication to find w[l]a[l-1] and store in z[l]
			mmul(w[i], a[i - 1], z[i], layer_sizes[i], layer_sizes[i - 1], 1, 0,0,layer_sizes[i], layer_sizes[i-1],layer_sizes[i]);
			// Add the bias vector to it
			fullBlocksPerGrid = ((layer_sizes[i] + blockSize - 1) / blockSize);
			kernAddVectors << <fullBlocksPerGrid, blockSize >> > (layer_sizes[i], b[i], z[i], z[i]);
			// Apply the Relu activation function
			kernReluActivationForward << <fullBlocksPerGrid, blockSize >> > (layer_sizes[i], z[i], a[i]);
		}
		// Now the softmax output for the final layer which will give the probability of belonging to each class
		// We will first calculate the z for the final layer
		mmul(w[L], a[L - 1], z[L], layer_sizes[L], layer_sizes[L - 1], 1,0,0,layer_sizes[L], layer_sizes[L-1], layer_sizes[L]);
		fullBlocksPerGrid = ((layer_sizes[L] + blockSize - 1) / blockSize);
		kernAddVectors << <fullBlocksPerGrid, blockSize >> > (layer_sizes[L], b[L], z[L], z[L]);
		// We will then calculate the sum(e^(z[L]))
		// Doing it on the CPU because in the stream compaction code, the cpu implementation was faster for smaller inputs.
		float *y_pred = new float[layer_sizes[L]];
		cudaMemcpy(y_pred, z[L], layer_sizes[L] * sizeof(float), cudaMemcpyDeviceToHost);
		float exp_sum = 0;
		for (int i = 0; i < layer_sizes[L]; i++) {
			exp_sum += expf(y_pred[i]);
		}
		// Now apply softmax activation

		fullBlocksPerGrid = ((layer_sizes[L] + blockSize - 1) / blockSize);
		
		kernSoftmaxActivation << <fullBlocksPerGrid, blockSize >> > (layer_sizes[L], z[L], a[L], exp_sum);
		cudaMemcpy(y_pred, a[L], layer_sizes[L] * sizeof(float), cudaMemcpyDeviceToHost);
		return y_pred;
		
	}
	void NeuralNet::backward(float *y) {
		int L = layer_sizes.size() - 1;
		// We will first populate da[L] as the derivative of loss with respect to y_pred.
		float *y_cuda;
		cudaMalloc((void**)&y_cuda, layer_sizes[L] * sizeof(float));
		cudaMemcpy(y_cuda, y, layer_sizes[L] * sizeof(float), cudaMemcpyHostToDevice);
		dim3 fullBlocksPerGrid;
		fullBlocksPerGrid = ((layer_sizes[L] + blockSize - 1) / blockSize);
		kernDerivativeLoss<<<fullBlocksPerGrid, blockSize>>>(layer_sizes[L], y_cuda, a[L], da[L]);
		// The equations for the backpropagation are
		// dz[l] = da[l]*g'[l](z[l]) where * means element wise
		// dw[l] = dz[l]a[l-1].T
		// db[l] = dz[l]
		// da[l-1] = W[l].Tdz[l]
		// Now the softmax derivative for the last but one layer
		float *sum_cp = new float[layer_sizes[L]];
		cudaMemcpy(sum_cp, z[L], layer_sizes[L] * sizeof(float), cudaMemcpyDeviceToHost);
		float exp_sum = 0;
		for (int i = 0; i < layer_sizes[L]; i++) {
			exp_sum += expf(sum_cp[i]);
		}
		kernSoftmaxDerivative << <fullBlocksPerGrid, blockSize >> > (layer_sizes[L], z[L], ghat[L], exp_sum);
		kernElementWiseMultiplication << <fullBlocksPerGrid, blockSize >> > (layer_sizes[L], da[L], ghat[L], dz[L]);
		// dw[l] = dz[l]a[l-1].T
		mmul(dz[L], a[L - 1], dw[L], layer_sizes[L], 1, layer_sizes[L-1], 0, 1,layer_sizes[L], layer_sizes[L-1], layer_sizes[L]);
		//db[l] = dz[l]
		cudaMemcpy(db[L], dz[L], layer_sizes[L] * sizeof(float), cudaMemcpyDeviceToDevice);
		//da[l - 1] = W[l].Tdz[l]
		mmul(w[L], dz[L], da[L - 1], layer_sizes[L - 1], layer_sizes[L], 1, 1, 0, layer_sizes[L], layer_sizes[L], layer_sizes[L - 1]);
		//Now for the ReLU layers
		for (int i = L - 1; i >= 1; i--) {
			kernReluDerivative << <fullBlocksPerGrid, blockSize >> > (layer_sizes[i], z[i], ghat[i]);
			kernElementWiseMultiplication << <fullBlocksPerGrid, blockSize >> > (layer_sizes[i], da[i], ghat[i], dz[i]);
			mmul(dz[i], a[i - 1], dw[i], layer_sizes[i], 1, layer_sizes[i - 1], 0, 1, layer_sizes[i], layer_sizes[i - 1], layer_sizes[i]);
			cudaMemcpy(db[i], dz[i], layer_sizes[i] * sizeof(float), cudaMemcpyDeviceToDevice);
			mmul(w[i], dz[i], da[i - 1], layer_sizes[i - 1], layer_sizes[i], 1, 1, 0, layer_sizes[i], layer_sizes[i], layer_sizes[i - 1]);
		}
		// Now we will update the weights and bias
		for (int i = 1; i <= L; i++) {
			printCuda(w[i], layer_sizes[i] * layer_sizes[i - 1], "W");
			printCuda(dw[i], layer_sizes[i] * layer_sizes[i - 1], "dw");
			fullBlocksPerGrid = ((layer_sizes[i]*layer_sizes[i-1] + blockSize - 1) / blockSize);
			kernUpdateParameters <<< fullBlocksPerGrid, blockSize >> > (layer_sizes[i] * layer_sizes[i - 1], w[i], dw[i], 0.01);
			printCuda(w[i], layer_sizes[i] * layer_sizes[i - 1], "W_updated");
			fullBlocksPerGrid = ((layer_sizes[i] + blockSize - 1) / blockSize);
			kernUpdateParameters <<< fullBlocksPerGrid, blockSize >>> (layer_sizes[i], b[i], db[i], 0.01);
		}
		// Avoid the memory leaks
		cudaFree(y_cuda);

	}
	NeuralNet::~NeuralNet() {
		// Here comes the destructor, will free those memories ...
		for (auto x : w)
			cudaFree(x);
		for (auto x : dw)
			cudaFree(x);
		for (auto x : b)
			cudaFree(x);
		for (auto x : db)
			cudaFree(x);
		for (auto x : z)
			cudaFree(x);
		for (auto x : dz)
			cudaFree(x);
		for (auto x : a)
			cudaFree(x);
		for (auto x : da)
			cudaFree(x);
		for (auto x : ghat)
			cudaFree(x);

		
		cublasDestroy(handle);
	}
    // TODO: __global__

    /**
        * Example of use case (follow how you did it in stream compaction)
        */
    /*void scan(int n, int *odata, const int *idata) {
        timer().startGpuTimer();
        // TODO
        timer().endGpuTimer();
    }
    */

	// TODO: implement required elements for MLP sections 1 and 2 here
}
