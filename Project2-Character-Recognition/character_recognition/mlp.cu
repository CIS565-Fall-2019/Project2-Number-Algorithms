#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "mlp.h"
#include <vector>

/*! Block size used for CUDA kernel launch. */
#define blockSize 128
#define index(i,j,ld) (((j)*(ld))+(i))



namespace CharacterRecognition {
    using Common::PerformanceTimer;
    PerformanceTimer& timer()
    {
        static PerformanceTimer timer;
        return timer;
    }

	typedef struct _matrixSize {
		int WA, HA, WB, HB, WC, HC;
	} sMatrixSize;

	void printMat(float*P, int uWP, int uHP) {
		int i, j;
		for (i = 0; i < uHP; i++) {
			printf("\n");
			for (j = 0; j < uWP; j++)
				printf("%f ", P[index(i, j, uHP)]);
		}
	}

	void randomInit(float *data, int size) {
		for (int i = 0; i < size; ++i)
			data[i] = rand() / (float)RAND_MAX;
	}

	void fixedInit(float *data, int size) {
		if (size == 4) {
			data[0] = 10.1f;
			data[1] = 0.9f;
			data[2] = 20.0f;
			data[3] = 0.87f;
		}
		else if (size == 2) {
			data[0] = 41.0f;
			data[1] = -54.0f;
		}
		printf("checking weight 0: %f \n", data[0]);
		printf("checking weight 1: %f \n", data[1]);
	}

	void indexInit(float *data, int size) {
		for (int i = 0; i < size; ++i)
			data[i] = (float)i;
	}

	int getNum(int &n, int *v) {
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

	void generateRandom(int n, int *permuteData) {
		int *v = (int *)malloc(n);
		// Fill the vector with the values  1, 2, 3, ..., n 
		for (int i = 0; i < n; i++) {
			v[i] = i;
		}
		// While vector has elements get a random number from the vector and print it 
		int i = 0;
		while (n > 0) {
			permuteData[i] = getNum(n,v);
			i++;
		}
	}

	/*void deviceMemory(bool create = false, float *Xi = NULL, float *wI = NULL, float *wO = NULL, sMatrixSize &hidden_matrix_size = {}, sMatrixSize &output_matrix_size = {}, float *dev_X = NULL, float *dev_wI = NULL, float *dev_wO = NULL, float *dev_h1 = NULL, float *dev_pred = NULL) {
		if (create) {
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

			cudaMalloc((void **)&dev_X, mem_size_X);
			checkCUDAError("cudaMalloc dev_X");
			cudaMalloc((void **)&dev_wI, mem_size_wI);
			checkCUDAError("cudaMalloc dev_wI");
			cudaMalloc((void **)&dev_wO, mem_size_wO);
			checkCUDAError("cudaMalloc dev_wO");
			cudaMalloc((void **)&dev_h1, mem_size_h1);
			checkCUDAError("cudaMalloc dev_h1");
			cudaMalloc((void **)&dev_pred, mem_size_pred);
			checkCUDAError("cudaMalloc dev_pred");
		}
		else {
			cudaFree(dev_X);
			cudaFree(dev_wI);
			cudaFree(dev_wO);
			cudaFree(dev_h1);
			cudaFree(dev_pred);
		}
	}*/

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
		//deviceMemory(true, Xi, wI, wO, hidden_matrix_size, output_matrix_size, dev_X, dev_wI, dev_wO, dev_h1, dev_pred);
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

		cudaMalloc((void **)&dev_X, mem_size_X);
		checkCUDAError("cudaMalloc dev_X");
		cudaMalloc((void **)&dev_wI, mem_size_wI);
		checkCUDAError("cudaMalloc dev_wI");
		cudaMalloc((void **)&dev_wO, mem_size_wO);
		checkCUDAError("cudaMalloc dev_wO");
		cudaMalloc((void **)&dev_h1, mem_size_h1);
		checkCUDAError("cudaMalloc dev_h1");
		cudaMalloc((void **)&dev_pred, mem_size_pred);
		checkCUDAError("cudaMalloc dev_pred");
		// allocate host memory for result
		//unsigned int size_pred = output_matrix_size.WC * output_matrix_size.HC;
		//unsigned int mem_size_pred = sizeof(float) * size_pred;
		float *pred = (float *)malloc(mem_size_pred);

		cublasHandle_t handle;
		cublasCreate(&handle);

		//hidden layer
		dim3 threads(blockSize);
		dim3 grid((hidden_matrix_size.WC*hidden_matrix_size.HC + blockSize - 1) / blockSize);

		matrixMultiply(&handle, hidden_matrix_size, dev_wI, dev_X, dev_h1);
		kernSigmoid <<<grid, threads>> > (hidden_matrix_size.HC*hidden_matrix_size.WC, dev_h1);
		checkCUDAError("kernSigmoid");


		//dim3 grid1(output_matrix_size.WC / threads.x, output_matrix_size.HC / threads.y);
		dim3 grid1((output_matrix_size.WC*output_matrix_size.HC + blockSize - 1) / blockSize);
		//output layer
		matrixMultiply(&handle, output_matrix_size, dev_wO, dev_h1, dev_pred);
		kernSigmoid << <grid1, threads >> > (output_matrix_size.HC*output_matrix_size.WC, dev_pred);
		checkCUDAError("kernSigmoid");

		cudaMemcpy(pred, dev_pred, mem_size_pred, cudaMemcpyDeviceToHost);
		checkCUDAError("cudaMemcpy pred");

		cublasDestroy(handle);
		checkCUDAError("handle");

		//deviceMemory();
		cudaFree(dev_X);
		cudaFree(dev_wI);
		cudaFree(dev_wO);
		cudaFree(dev_h1);
		cudaFree(dev_pred);

		return pred;
	}

	void train(float *X, float *y, int sizeData, const int hiddenNodes, const int numLabels, const int numData) {
		sMatrixSize hidden_matrix_size = {hiddenNodes, sizeData, 1, sizeData, 1, hiddenNodes };
		sMatrixSize output_matrix_size = {numLabels, hiddenNodes, 1, hiddenNodes, 1, numLabels };

		unsigned int size_wI = hidden_matrix_size.WA * hidden_matrix_size.WA;
		unsigned int mem_size_wI = sizeof(float) * size_wI;
		float *wI = (float *)malloc(mem_size_wI);

		unsigned int size_wO = output_matrix_size.HA * output_matrix_size.WA;
		unsigned int mem_size_wO = sizeof(float) * size_wO;
		float *wO = (float *)malloc(mem_size_wO);

		fixedInit(wI, size_wI);
		fixedInit(wO, size_wO);

		int *permuteData = (int *)malloc(numData);
		float *Xi = (float *)malloc(sizeData);
		float *yi = (float *)malloc(numLabels);

		unsigned int size_pred = output_matrix_size.WC * output_matrix_size.HC;
		unsigned int mem_size_pred = sizeof(float) * size_pred;
		float *pred = (float *)malloc(mem_size_pred);

		for (int iter = 0; iter < 1; iter++) {
			generateRandom(numData, permuteData);
			printf("predicting iteration %i \n", iter);
			for (int i = 0; i < numData; i++) {
				int index = permuteData[i];
				memcpy(Xi, (void **)&X[sizeData*index], sizeData * sizeof(float));
				memcpy(yi, (void **)&y[numLabels*index], numLabels * sizeof(float));

				printf("index %i \n", index);
				printf("data: %f %f label: %f \n", Xi[0] , Xi[1], yi[0]);

				pred = forward(Xi, yi, wI, wO, hidden_matrix_size, output_matrix_size);
				for (int j = 0; j < numLabels; j++) {
					printf("prediction: %f \n", pred[j]);
				}
			}
			printf("forward done \n");
		}
		printf("predictions done \n");

		free(wI);
		free(wO);
		free(Xi);
		free(yi);
		free(permuteData);
		free(pred);
	}

	void testMatrixMultiply() {
		sMatrixSize matrix_size = { 3, 4, 2, 3, 2, 4};

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
		printf("Computing result using CUBLAS... \n");

		cublasHandle_t handle;
		cublasCreate(&handle);

		////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		matrixMultiply(&handle, matrix_size, d_A, d_B, d_C);
		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		// copy result from device to host
		cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);

		// Destroy the handle
		cublasDestroy(handle);

		printf("\n\n Matriz A:");
		printMat(h_A, matrix_size.WA, matrix_size.HA);
		printf("\n\n Matriz B:");
		printMat(h_B, matrix_size.WB, matrix_size.HB);
		printf("\n\n Matriz C:");
		printMat(h_C, matrix_size.WC, matrix_size.HC);
		printf("\n\n");

		// clean up memory
		free(h_A);
		free(h_B);
		free(h_C);
		cudaFree(d_A);
		cudaFree(d_B);
		cudaFree(d_C);
	}

}
