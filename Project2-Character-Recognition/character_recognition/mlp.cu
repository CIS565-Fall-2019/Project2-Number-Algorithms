#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "mlp.h"
#include <vector>

/*! Block size used for CUDA kernel launch. */
#define blockSize 32
#define index(i,j,ld) (((j)*(ld))+(i))



namespace CharacterRecognition {
    using Common::PerformanceTimer;
    PerformanceTimer& timer()
    {
        static PerformanceTimer timer;
        return timer;
    }

	typedef struct _matrixSize      // Optional Command-line multiplier for matrix sizes
	{
		unsigned int WA, HA, WB, HB, WC, HC;
		_matrixSize(unsigned int WA=1, unsigned int HA=1, unsigned int WB=1, unsigned int HB=1, unsigned int WC=1, unsigned int HC=1){}
	} sMatrixSize;

	void printMat(float*P, int uWP, int uHP) {
		int i, j;
		for (i = 0; i < uHP; i++) {
			printf("\n");
			for (j = 0; j < uWP; j++)
				printf("%f ", P[index(i, j, uHP)]);
		}
	}

	void randomInit(float *data, int size)
	{
		for (int i = 0; i < size; ++i)
			data[i] = rand() / (float)RAND_MAX;
	}

	void indexInit(float *data, int size)
	{
		for (int i = 0; i < size; ++i)
			data[i] = (float)i;
	}

	void initializeCUDA(sMatrixSize &matrix_size)
	{
		matrix_size.WA = 3;
		matrix_size.HA = 4;
		matrix_size.WB = 2;
		matrix_size.HB = 3;
		matrix_size.WC = 2;
		matrix_size.HC = 4;

		printf("MatrixA(%u,%u), MatrixB(%u,%u), MatrixC(%u,%u)\n",
			matrix_size.HA, matrix_size.WA,
			matrix_size.HB, matrix_size.WB,
			matrix_size.HC, matrix_size.WC);

	}

	int getNum(int &n, float *v) {
		// Generate a random number 
		srand(time(NULL));
		// Make sure the number is within the index range 
		int index = rand() % n;
		// Get random number from the vector 
		int num = v[index];
		// Remove the number from the vector 
		std::swap(v[index], v[n - 1]);
		n--;
		// Return the removed number 
		return num;
	}

	void generateRandom(int n, float *perm) {
		float *v = (float *)malloc(n);
		// Fill the vector with the values  1, 2, 3, ..., n 
		for (int i = 0; i < n; i++) {
			v[i] = i;
		}
		// While vector has elements get a random number from the vector and print it 
		int i = 0;
		while (n > 0) {
			perm[i] = getNum(n,v);
			i++;
		}
	}

	////////////////////////////////////////////////////////////////////////////////
	//! Run a simple test matrix multiply using CUBLAS
	////////////////////////////////////////////////////////////////////////////////
	void matrixMultiply(cublasHandle_t* handle, sMatrixSize &matrix_size, float *d_A, float *d_B, float *d_C){
			const float alpha = 1.0f;
			const float beta = 0.0f;

			cublasSgemm(*handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.WB, matrix_size.HA, matrix_size.WA, &alpha, d_B, matrix_size.WB, d_A, matrix_size.WA, &beta, d_C, matrix_size.WB);
			checkCUDAError("matrix multiply");
	}

	// TODO: implement required elements for MLP sections 1 and 2 here
	__global__ void kernSigmoid(int n, float *input) {
		int index = threadIdx.x + (blockIdx.x * blockDim.x);
		if (index >= n) {
			return;
		}
		input[index] = 1.0f / (1 + exp(-input[index]));
	}
	void backward(){}

	float *forward(float *Xi, float *yi, float *wI, float *wO, sMatrixSize &hidden_matrix_size, sMatrixSize &output_matrix_size) {
		// allocate device memory
		float *dev_X, *dev_wI, *dev_wO, *dev_h1, *dev_pred;
		unsigned int size_X = hidden_matrix_size.WB * hidden_matrix_size.HB;
		unsigned int mem_size_X = sizeof(float) * size_X;
		unsigned int size_wI = hidden_matrix_size.WA * hidden_matrix_size.HA;
		unsigned int mem_size_wI = sizeof(float) * size_wI;
		unsigned int size_wO = output_matrix_size.WA * output_matrix_size.HA;
		unsigned int mem_size_wO = sizeof(float) * size_wO;
		unsigned int size_h1 = hidden_matrix_size.WC * hidden_matrix_size.HC;
		unsigned int mem_size_h1 = sizeof(float) * size_h1;
		unsigned int size_pred = output_matrix_size.WC * output_matrix_size.HC;
		unsigned int mem_size_pred = sizeof(float) * size_pred;

		// allocate host memory for the result
		float *pred = (float *)malloc(mem_size_pred);

		cudaMalloc((void **)&dev_X, mem_size_X);
		checkCUDAError("cudaMalloc dev_X");
		cudaMalloc((void **)&dev_wI, mem_size_wI);
		checkCUDAError("cudaMalloc dev_wI");
		cudaMalloc((void **)&dev_wO, mem_size_wO);
		checkCUDAError("cudaMalloc dev_wO");
		cudaMemcpy(dev_X, Xi, mem_size_X, cudaMemcpyHostToDevice);
		checkCUDAError("cudaMemcpy dev_X");
		cudaMemcpy(dev_wI, wI, mem_size_wI, cudaMemcpyHostToDevice);
		checkCUDAError("cudaMemcpy dev_wI");
		cudaMemcpy(dev_wO, wO, mem_size_wO, cudaMemcpyHostToDevice);
		checkCUDAError("cudaMemcpy dev_wO");
		cudaMalloc((void **)&dev_h1, mem_size_h1);
		checkCUDAError("cudaMalloc dev_h1");
		cudaMalloc((void **)&dev_pred, mem_size_pred);
		checkCUDAError("cudaMalloc dev_pred");

		dim3 threads(blockSize, blockSize);
		dim3 grid(hidden_matrix_size.WC / threads.x, hidden_matrix_size.HC / threads.y);

		cublasHandle_t handle;
		cublasCreate(&handle);

		//hidden layer

		matrixMultiply(&handle, hidden_matrix_size, dev_wI, dev_X, dev_h1);
		kernSigmoid <<<grid, threads>> > (hidden_matrix_size.HC, dev_h1);
		checkCUDAError("kernSigmoid");


		dim3 grid1(output_matrix_size.WC / threads.x, output_matrix_size.HC / threads.y);
		//output layer
		matrixMultiply(&handle, output_matrix_size, dev_wO, dev_h1, dev_pred);
		kernSigmoid << <grid1, threads >> > (output_matrix_size.HC, dev_pred);
		checkCUDAError("kernSigmoid");

		cudaMemcpy(pred, dev_pred, mem_size_pred, cudaMemcpyDeviceToHost);
		checkCUDAError("cudaMemcpy pred");

		cublasDestroy(handle);
		checkCUDAError("handle");

		cudaFree(dev_X);
		cudaFree(dev_wI);
		cudaFree(dev_wO);
		cudaFree(dev_h1);
		cudaFree(dev_pred);

		return pred;
	}

	void train(float *X, float *y, int sizeData, const int hiddenNodes, const int numLabels, const int numData) {
		sMatrixSize hidden_matrix_size(1, sizeData, hiddenNodes, sizeData, 1, hiddenNodes);
		sMatrixSize output_matrix_size(1, hiddenNodes, numLabels, hiddenNodes, 1, numLabels);

		srand(2006); 

		unsigned int size_wI = sizeData * hiddenNodes;
		unsigned int mem_size_wI = sizeof(float) * size_wI;
		float *wI = (float *)malloc(mem_size_wI);

		unsigned int size_wO = numLabels * hiddenNodes;
		unsigned int mem_size_wO = sizeof(float) * size_wO;
		float *wO = (float *)malloc(mem_size_wO);

		randomInit(wI, size_wI);
		randomInit(wO, size_wO);

		float *perm = (float *)malloc(numData);
		float *Xi = (float *)malloc(sizeData);
		float *yi = (float *)malloc(numLabels);

		unsigned int size_pred = output_matrix_size.WC * output_matrix_size.HC;
		unsigned int mem_size_pred = sizeof(float) * size_pred;
		float *pred = (float *)malloc(mem_size_pred);
		for (int iter = 0; iter < 1000; iter++) {
			generateRandom(numData, perm);
			printf("here");
			for (int i = 0; i < numData; i++) {
				int index = perm[i];
				printf("%i \n", index);
				memcpy(Xi, (void **)&X[sizeData*i], sizeData * sizeof(float));
				memcpy(yi, (void **)&y[numLabels*i], numLabels * sizeof(float));
				pred = forward(Xi, yi, wI, wO, hidden_matrix_size, output_matrix_size);
				for (int j = 0; j < numLabels; j++) {
					printf("%f \n", pred[j]);
				}
			}
		}

		free(Xi);
		free(yi);
		free(perm);
	}

	void testMatrixMultiply() {
		sMatrixSize matrix_size;

		initializeCUDA(matrix_size);

		// allocate host memory for matrices A and B
		unsigned int size_A = matrix_size.WA * matrix_size.HA;
		unsigned int mem_size_A = sizeof(float) * size_A;
		float *h_A = (float *)malloc(mem_size_A);
		unsigned int size_B = matrix_size.WB * matrix_size.HB;
		unsigned int mem_size_B = sizeof(float) * size_B;
		float *h_B = (float *)malloc(mem_size_B);

		// set seed for rand()
		srand(2006);

		// initialize host memory
		indexInit(h_A, size_A);
		indexInit(h_B, size_B);

		// allocate device memory
		float *d_A, *d_B, *d_C;
		unsigned int size_C = matrix_size.WC * matrix_size.HC;
		unsigned int mem_size_C = sizeof(float) * size_C;

		// allocate host memory for the result
		float *h_C = (float *)malloc(mem_size_C);

		cudaMalloc((void **)&d_A, mem_size_A);
		cudaMalloc((void **)&d_B, mem_size_B);
		cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
		cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);
		cudaMalloc((void **)&d_C, mem_size_C);

		// setup execution parameters
		dim3 threads(blockSize, blockSize);
		dim3 grid(matrix_size.WC / threads.x, matrix_size.HC / threads.y);

		// create and start timer
		printf("Computing result using CUBLAS...");

		cublasHandle_t handle;
		cublasCreate(&handle);

		////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		matrixMultiply(&handle, matrix_size, d_A, d_B, d_C);
		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		// copy result from device to host
		cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);

		// Destroy the handle
		cublasDestroy(handle);

		printf("\nMatriz A:\n");
		printMat(h_A, matrix_size.WA, matrix_size.HA);
		printf("\nMatriz B:\n");
		printMat(h_B, matrix_size.WB, matrix_size.HB);
		printf("\nMatriz C:\n");
		printMat(h_C, matrix_size.WC, matrix_size.HC);

		// clean up memory
		free(h_A);
		free(h_B);
		free(h_C);
		cudaFree(d_A);
		cudaFree(d_B);
		cudaFree(d_C);
	}


}
