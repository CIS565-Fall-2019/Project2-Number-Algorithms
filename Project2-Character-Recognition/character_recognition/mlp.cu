

#include <cuda.h>
#include <cuda_runtime.h>
//#include <device_functions.h>
//#include "device_launch_parameters.h"
#include "common.h"
#include "mlp.h"
#include <vector>
#include <thrust/scan.h>

/*! Block size used for CUDA kernel launch. */
#define blockSize 32
#define index(i,j,ld) (((j)*(ld))+(i))
#define GPULOSS 0

float *dev_X, *dev_wI, *dev_wO, *dev_h1, *dev_pred, *dev_loss, *dev_y;

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
			for (j = 0; j < uWP; j++)
				printf(" %f ", P[index(i, j, uHP)]);
			printf("\n");
		}
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
	}


	void randomInit(float *data, int size) {
		for (int i = 0; i < size; ++i)
			data[i] = rand() / (float)RAND_MAX;
	}


	void indexInit(float *data, int size) {
		for (int i = 0; i < size; ++i)
			data[i] = (float)i;
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


	void generateRandom(int n, float *permuteData) {
		float *v = (float *)malloc(n);
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


	void deviceMemory(float *Xi, float *wI, float *wO, sMatrixSize &hidden_matrix_size, sMatrixSize &output_matrix_size, bool create = false) {
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

			cudaMemcpy(dev_X, Xi, mem_size_X, cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy dev_X");
			cudaMemcpy(dev_wI, wI, mem_size_wI, cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy dev_wI");
			cudaMemcpy(dev_wO, wO, mem_size_wO, cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy dev_wO");
		}
		else {
			cudaFree(dev_X);
			cudaFree(dev_wI);
			cudaFree(dev_wO);
			cudaFree(dev_h1);
			cudaFree(dev_pred);
			cudaFree(dev_y);
			cudaFree(dev_loss);
		}
	}


	void matrixMultiply(cublasHandle_t* handle, sMatrixSize &matrix_size, float *d_A, float *d_B, float *d_C){
			const float alpha = 1.0f;
			const float beta = 0.0f;
			cublasSgemm(*handle, CUBLAS_OP_T, CUBLAS_OP_N, matrix_size.WA, matrix_size.WB, matrix_size.HA, &alpha, d_A, matrix_size.HA, d_B, matrix_size.HB, &beta, d_C, matrix_size.HC);
			checkCUDAError("matrix multiply");
	}

	__global__ void kernSigmoid(int n, float *input) {
		int index = threadIdx.x + (blockIdx.x * blockDim.x);
		if (index >= n) {
			return;
		}
		input[index] = 1.0f / (1 + exp(-input[index]));
	}

	__global__ void kernMSE(int n, float *pred, float *yi, float *loss) {
		int index = threadIdx.x + (blockIdx.x * blockDim.x);
		if (index >= n) {
			return;
		}
		loss[index] = powf(pred[index] - yi[index], 2);
	}
	

	float MSE(float *pred, float *yi, sMatrixSize &output_matrix_size) {
#if GPULOSS
		float *loss;
		unsigned int size_pred = output_matrix_size.WC * output_matrix_size.HC;
		unsigned int mem_size_pred = sizeof(float) * size_pred;

		cudaMalloc((void **)&dev_loss, mem_size_pred);
		checkCUDAError("cudaMalloc dev_pred");
		cudaMalloc((void **)&dev_y, mem_size_pred);
		checkCUDAError("cudaMalloc dev_pred");
		cudaMalloc((void **)&dev_pred, mem_size_pred);
		checkCUDAError("cudaMalloc dev_pred");

		cudaMemcpy(dev_y, yi, mem_size_pred, cudaMemcpyHostToDevice);
		checkCUDAError("cudaMemcpy dev_y");
		cudaMemcpy(dev_pred, pred, mem_size_pred, cudaMemcpyHostToDevice);
		checkCUDAError("cudaMemcpy dev_pred");

		dim3 threads(blockSize);
		dim3 grid((output_matrix_size.HC + blockSize - 1) / blockSize);

		kernMSE << <grid, threads >> > (output_matrix_size.HC, dev_pred, dev_y, dev_loss);

		thrust::inclusive_scan(dev_loss, dev_loss + output_matrix_size.HC, dev_loss);

		cudaMemcpy(loss, dev_loss + mem_size_pred - 1, sizeof(float), cudaMemcpyDeviceToHost);
		checkCUDAError("cudaMemcpy dev_loss");

		cudaFree(dev_loss);
		cudaFree(dev_y);
		cudaFree(dev_pred);
		return *loss;
#else
		float loss = 0;
		for (int j = 0; j < output_matrix_size.HC; j++) {
			printf("prediction: %f \n", pred[j]);
			loss += pow(pred[0] - yi[0], 2);
		}
		return loss;
#endif // GPULOSS	
	}


	void backward(float lr, float loss, float *wI, float *wO){
		printf("Backward propagate the error to the weights \n");
		// wI += gradient*lr;
		// wO += gradient*lr;
	}


	void forward(float *pred, float *Xi, float *wI, float *wO, sMatrixSize &hidden_matrix_size, sMatrixSize &output_matrix_size) {
		// allocate device memory
		deviceMemory(Xi, wI, wO, hidden_matrix_size, output_matrix_size, true);
		int mem_size_pred = sizeof(float) * output_matrix_size.WC * output_matrix_size.HC;

		cublasHandle_t handle;
		cublasCreate(&handle);
		dim3 threads(blockSize);

		//hidden layer
		dim3 grid((hidden_matrix_size.WC*hidden_matrix_size.HC + blockSize - 1) / blockSize);

		matrixMultiply(&handle, hidden_matrix_size, dev_wI, dev_X, dev_h1);
		kernSigmoid <<<grid, threads>> > (hidden_matrix_size.HC*hidden_matrix_size.WC, dev_h1);
		checkCUDAError("kernSigmoid");

		//output layer
		dim3 grid1((output_matrix_size.WC*output_matrix_size.HC + blockSize - 1) / blockSize);

		matrixMultiply(&handle, output_matrix_size, dev_wO, dev_h1, dev_pred);
 		kernSigmoid << <grid1, threads >> > (output_matrix_size.HC*output_matrix_size.WC, dev_pred);
		checkCUDAError("kernSigmoid");

		cudaMemcpy(pred, dev_pred, mem_size_pred, cudaMemcpyDeviceToHost);
		checkCUDAError("cudaMemcpy pred");

		cublasDestroy(handle);
		checkCUDAError("handle");

		deviceMemory(Xi, wI, wO, hidden_matrix_size, output_matrix_size);
	}

	void train(float *X, float *y, const int iterations, const float lr, const int sizeData, const int hiddenNodes, const int numLabels, const int numData) {
		sMatrixSize hidden_matrix_size = {hiddenNodes, sizeData, 1, sizeData, 1, hiddenNodes };
		sMatrixSize output_matrix_size = {numLabels, hiddenNodes, 1, hiddenNodes, 1, numLabels };

		unsigned int size_wI = hidden_matrix_size.HA * hidden_matrix_size.WA;
		unsigned int mem_size_wI = sizeof(float) * size_wI;
		float *wI = (float *)malloc(mem_size_wI);

		unsigned int size_wO = output_matrix_size.HA * output_matrix_size.WA;
		unsigned int mem_size_wO = sizeof(float) * size_wO;
		float *wO = (float *)malloc(mem_size_wO);

		randomInit(wI, size_wI);
		randomInit(wO, size_wO);

		float *permuteData = (float *)malloc(sizeof(float)*numData);
		float *Xi = (float *)malloc(sizeof(float)*sizeData);
		float *yi = (float *)malloc(sizeof(float)*numLabels);

		unsigned int size_pred = output_matrix_size.WC * output_matrix_size.HC;
		unsigned int mem_size_pred = sizeof(float) * size_pred;
		float *pred = (float *)malloc(mem_size_pred);
		float avgLoss = 0.0;

		for (int iter = 0; iter < iterations; iter++) {
			generateRandom(numData, permuteData);
			printf("training iteration %i \n", iter);
			for (int i = 0; i < numData; i++) {
				int index = permuteData[i];
				memcpy(Xi, (void **)&X[sizeData*index], sizeData * sizeof(float));
				memcpy(yi, (void **)&y[numLabels*index], numLabels * sizeof(float));

				printf("index: %i data: %f %f label: %f \n", index, Xi[0] , Xi[1], yi[0]);

				forward(pred, Xi, wI, wO, hidden_matrix_size, output_matrix_size);
				for (int j = 0; j < numLabels; j++) {
					printf("prediction: %f \n\n", pred[j]);
				}

				float loss = MSE(pred, yi, output_matrix_size);
				printf("Least square error loss: %f \n \n", loss);
				avgLoss += loss;
				backward(lr, loss, wI, wO);
			}
		}
		printf("training done \n");

		avgLoss /= numData;
		printf("Average Loss: %f \n", avgLoss);

		free(wI); free(wO); free(Xi); free(yi); free(permuteData); free(pred);
	}

	void test(float *X, float *y, float *wI, float *wO, const int sizeData, const int hiddenNodes, const int numLabels, const int numData){
		sMatrixSize hidden_matrix_size = { hiddenNodes, sizeData, 1, sizeData, 1, hiddenNodes };
		sMatrixSize output_matrix_size = { numLabels, hiddenNodes, 1, hiddenNodes, 1, numLabels };

		float *Xi = (float *)malloc(sizeof(float)*sizeData);
		float *yi = (float *)malloc(sizeof(float)*numLabels);

		unsigned int size_pred = output_matrix_size.WC * output_matrix_size.HC;
		unsigned int mem_size_pred = sizeof(float) * size_pred;
		float *pred = (float *)malloc(mem_size_pred);
		float avgLoss = 0.0;

		for (int i = 0; i < numData; i++) {
			memcpy(Xi, (void **)&X[sizeData*i], sizeData * sizeof(float));
			memcpy(yi, (void **)&y[numLabels*i], numLabels * sizeof(float));

			printf("data: %f %f label: %f \n", i, Xi[0], Xi[1], yi[0]);

			forward(pred, Xi, wI, wO, hidden_matrix_size, output_matrix_size);

			float loss = MSE(pred, yi, output_matrix_size);
			printf("Least square error loss: %f \n \n", loss);
			avgLoss += loss;
		}
		printf("testing done \n");

		avgLoss /= numData;
		printf("Average Loss: %f \n", avgLoss);

		free(wI); free(wO); free(Xi); free(yi); free(pred);
	}

	void testMatrixMultiply(int HA, int WA, int HB, int WB) {
		sMatrixSize matrix_size = { WA, HA, WB, HB, WB, HA};

		// allocate host memory for matrices A and B
		unsigned int size_A = matrix_size.WA * matrix_size.HA;
		unsigned int mem_size_A = sizeof(float) * size_A;
		float *h_A = (float *)malloc(mem_size_A);
		unsigned int size_B = matrix_size.WB * matrix_size.HB;
		unsigned int mem_size_B = sizeof(float) * size_B;
		float *h_B = (float *)malloc(mem_size_B);

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
		dim3 grid(matrix_size.HB / threads.x, matrix_size.WA / threads.y);

		// create and start timer
		printf("Computing result using CUBLAS... \n");

		cublasHandle_t handle;
		cublasCreate(&handle);

		matrixMultiply(&handle, matrix_size, d_A, d_B, d_C);

		// copy result from device to host
		cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);

		// Destroy the handle
		cublasDestroy(handle);

		printf("\n Matriz A: \n");
		printMat(h_A, matrix_size.WA, matrix_size.HA);
		printf("\n Matriz B: \n ");
		printMat(h_B, matrix_size.WB, matrix_size.HB);
		printf("\n Matriz C: \n");
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
