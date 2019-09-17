#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "mlp.h"
#include <thrust/scan.h>
#include <fstream>
#include <iostream>
using namespace std;

//#include "cublas_v2.h"

# define blockSize 128
# define block 15

namespace CharacterRecognition {
    using Common::PerformanceTimer;
    PerformanceTimer& timer()
    {
        static PerformanceTimer timer;
        return timer;
    }
    
	float *dev_input;
	float *dev_hiddenLayer;
	float *dev_output;
	float *dev_weightsIH;
	float *dev_weightsHO;
	float *dev_newWeightsIH;
	float *dev_newWeightsHO;
	float *dev_actualOutput;
	float *dev_gradB;
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

	// Multiply the arrays A and B on GPU and save the result in C
	// C(m,n) = A(m,k) * B(k,n)
	/*
	void gpu_blas_mmul(const float *A, const float *B, float *C, const int m, const int k, const int n) {
		int lda = m, ldb = k, ldc = m;
		const float alf = 1;
		const float bet = 0;
		const float *alpha = &alf;
		const float *beta = &bet;

		// Create a handle for CUBLAS
		cublasHandle_t handle;
		cublasCreate(&handle);

		// Do the actual multiplication
		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

		// Destroy the handle
		cublasDestroy(handle);
	}
	*/
	__global__ void kernMatrixMultiplication(float *M,float *N, float *Out,int m, int n,int k)  {
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		//printf("The values of m , n and k are :%d , %d %d \n", m, n , k);
		//printf("The values of row and col are: %d & %d \n", row, col);
		float sum = 0;
		if (col < k && row < m) {
			for (int i = 0; i < n; i++) {
				sum += M[row*n + i] * N[i*k + col];
				//printf("hello the value of Sum is : %0.3f\n",sum);
			}
			//printf("The values are %d & %d \n", row, col);
			Out[row*k + col] = sum;
			//printf("The value is: %0.2f \n", Out[row*k + col]);
		}

	}
	
	void printArray(int n, float *a, bool abridged = false) {
		printf("    [ ");
		for (int i = 0; i < n; i++) {
			if (abridged && i + 2 == 15 && n > 16) {
				i = n - 2;
				printf("... ");
			}
			printf("%0.2f ", a[i]);
		}
		printf("]\n");
	}

	__global__ void kernSigmoidFunction(int N, float* A) {
		int index = threadIdx.x + blockIdx.x * blockDim.x;
		if (index >= N)
			return;

		A[index] = exp(-1*A[index]);
		A[index] = 1.0 / (1.0 + A[index]);

	}

	__global__ void kernSoftMax(int N, float *A, int d) {
		int index = threadIdx.x + blockIdx.x * blockDim.x;
		if (index >= N)
			return;
		//printf("The index values are :%d\n", index);
		float sum = 0;
		//printf("The values are %d and %d :\n", N, d);
		for (int i = index * d; i < index*d + d; i++) {
			sum += exp(A[i]);
			//printf("%d \n", i);
		}

		for (int i = index * d; i < index*d + d; i++) {
			A[i] = exp(A[i]) / sum;
		}
	}

	// TODO: implement required elements for MLP sections 1 and 2 here

	__global__ void kernCalculateLoss(int N, float *output, float *actualOutput, float *loss,int d) {
		int index = threadIdx.x + blockIdx.x * blockDim.x;
		if (index >= N)
			return;

		for (int i = index * d; i < index*d + d; i++) {
			if (actualOutput[i] == 1.0)
				loss[index] = -log(output[i]);
		}
		
	}

	__global__ void kernSubtraction(int N,float *A, float *B, float *C) {
		int index = threadIdx.x + blockIdx.x * blockDim.x;
		if (index >= N)
			return;

		C[index] = A[index] - B[index];
	}

	__global__ void kernSoftMaxGradient(int N,int d, float *A, float *B, float *C) {
		int index = threadIdx.x + blockIdx.x * blockDim.x;
		if (index >= N)
			return;

		
		for (int i = index * d; i < index*d + d; i++) {
			if (B[i] == 1.0) 
				C[i] = A[i] - 1;
			else
				C[i] = A[i];
			C[i] /= N;

 		}
	}

	__global__ void kernSigmoidGrad(int N, float *A,float *B) {
		int index = threadIdx.x + blockIdx.x * blockDim.x;
		if (index >= N)
			return;

		B[index] = A[index] * (1 - A[index]);
	}

	__global__ void kernDotProduct(int N,float *A, float *B,float *C) {
		int index = threadIdx.x + blockIdx.x * blockDim.x;
		if (index >= N)
			return;

		C[index] = A[index] * B[index];
		//printf("Values are for index %d is: %0.2f \n", index,C[index]);
	}

	__global__ void gpu_matrix_transpose(float* mat_in, float* mat_out, unsigned int rows, unsigned int cols)
	{
		unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

		if (idx < cols && idy < rows)
		{
			unsigned int pos = idy * cols + idx;
			unsigned int trans_pos = idx * rows + idy;
			mat_out[trans_pos] = mat_in[pos];
		}
	}

	__global__ void kernUpdateWeights(int N, float *A, float *B,float *C,float step_size) {
		int index = threadIdx.x + blockIdx.x * blockDim.x;
		if (index >= N)
			return;

		B[index] = A[index] - step_size * C[index];

	}

	__global__ void kernGetAccuracy(int N, float *A,int *B,int d) {
		int index = threadIdx.x + blockIdx.x * blockDim.x;
		if (index >= N)
			return;

		int max_index = 0;
		float max_val = -1;;
		for (int i = index * d; i < index*d + d; i++) {
			if (max_val < A[i]) {
				max_val = A[i];
				max_index = i;
			}
		}
		B[index] = max_index % d;
	}
	void createNN(float *input, float* hidden, float *output, float *weightsA, float *weightsB, int n, int h, int m, int d) {
		
		dim3 blockDim(block, block);
		dim3 fullBlocks1((n + blockSize - 1) / blockSize);
		dim3 fullBlocks2((n*h + blockSize - 1) / blockSize);
		//dim3 fullBlocks3((n + blockSize - 1) / blockSize);
		dim3 fullBlocksMult1((h + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);
		dim3 fullBlocksMult2((m + blockDim.x - 1) / blockDim.x, (n + blockDim.x - 1) / blockDim.x);

		kernMatrixMultiplication << <fullBlocksMult1, blockDim >> > (input,weightsA,hidden,n, d,h);
		checkCUDAErrorFn("Multiplication 1 failed");
		//gpu_blas_mmul(dev_input, dev_weightsIH, dev_hiddenLayer, n, d, h);

		kernSigmoidFunction << <fullBlocks2, blockSize >> > (n*h, hidden);
		checkCUDAErrorFn("Kernel Activation function failed");

		kernMatrixMultiplication << <fullBlocksMult2, blockDim >> > (hidden, weightsB, output, n, h, m);
		checkCUDAErrorFn("Multiplication 2 failed");
		//gpu_blas_mmul(dev_hiddenLayer, dev_weightsHO, dev_output, h, d, m);

		kernSoftMax << <fullBlocks1, blockSize >> > (n,output,m);
		checkCUDAErrorFn("Kernel Soft Max function failed");

		//kernSigmoidFunction << <fullBlocks3, blockSize >> > (m, dev_output);
		//checkCUDAErrorFn("Kernel Activation function failed");
	}
	
	void trainNN(float *input, float *hidden,float *output,float *actualOutput, float *weightsA,float *weightsB,
		float *newWeightsA,float *newWeightsB,int n, int h ,int m, int d) {
		
		float *hiddenTrans;
		float *gradSoftMax;
		float *weightsBTrans;
		float *devGrad;
		float *dev_hiddenLayerGrad;
		float *devGrad2;
		float *inputTrans;
		float *dev_gradA;

		cudaMalloc((void**)&dev_gradB, (h*m) * sizeof(float));
		checkCUDAErrorFn("Malloc geadient B weights failed");

		cudaMalloc((void**)&gradSoftMax, (n*m) * sizeof(float));
		checkCUDAErrorFn("Malloc Soft gradient failed");

		cudaMalloc((void**)&weightsBTrans, (m*h) * sizeof(float));
		checkCUDAErrorFn("Malloc weightsB Transpose failed");

		cudaMalloc((void**)&devGrad, (n*h) * sizeof(float));
		checkCUDAErrorFn("Malloc devGrad failed");

		cudaMalloc((void**)&devGrad2, (n*h) * sizeof(float));
		checkCUDAErrorFn("Malloc devGrad2 failed");

		cudaMalloc((void**)&hiddenTrans, (n*h) * sizeof(float));
		checkCUDAErrorFn("Malloc hiddenTrans failed");

		cudaMalloc((void**)&dev_hiddenLayerGrad, (n*h) * sizeof(float));
		checkCUDAErrorFn("Malloc hiddenlayer gradient failed");

		cudaMalloc((void**)&inputTrans, (n*d) * sizeof(float));
		checkCUDAErrorFn("Malloc input trans failed");

		cudaMalloc((void**)&dev_gradA, (n*h) * sizeof(float));
		checkCUDAErrorFn("Malloc gradient A failed");

		// Wrote the structure as of now, needs to check later
		dim3 blockDim(block, block);
		dim3 fullBlocksMult1((h + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);
		gpu_matrix_transpose << <fullBlocksMult1, blockDim >> > (hidden,hiddenTrans,n,h);
		checkCUDAErrorFn("Kernel transpose hidden failed");

		dim3 fullBlocksMult2((m + blockDim.x - 1) / blockDim.x, (h + blockDim.y - 1) / blockDim.y);
		dim3 fullBlocksMult3((n + blockSize - 1) / blockSize);

		//kernSubtraction << <fullBlocksMult3, blockSize >> > (n*m,output, actualOutput,tempOutput);

		kernSoftMaxGradient << <fullBlocksMult3, blockSize >> > (n,m,output,actualOutput,gradSoftMax);
		/*
		float *check0 = new float[d*h];

		cudaMemcpy(check0, gradSoftMax, sizeof(float) * (n*m), cudaMemcpyDeviceToHost);
		checkCUDAErrorFn("Copying data to output failed");

		printf("Gradient Soft Max \n");
		printArray(n*m, check0, true);
		*/

		kernMatrixMultiplication << <fullBlocksMult2, blockDim >> > (hiddenTrans, gradSoftMax ,dev_gradB,h,n,m);
		checkCUDAErrorFn("Kernel Matrix Multiplication hiiden and loss failed");

		//float *check0 = new float[d*h];
		
		gpu_matrix_transpose << <fullBlocksMult2, blockDim >> > (weightsB, weightsBTrans, h, m);
		checkCUDAErrorFn("Kernel Transpose for weightsB failed");

		kernMatrixMultiplication << <fullBlocksMult1, blockDim >> > (gradSoftMax,weightsBTrans,devGrad,n,m,h);
		checkCUDAErrorFn("Kernel Matrix Multiplication for Devgrad failed");

		dim3 fullBlocksMult4((n*h + blockSize - 1) / blockSize);
		kernSigmoidGrad << <fullBlocksMult4,blockSize >> > (h*n,hidden,dev_hiddenLayerGrad);
		checkCUDAErrorFn("Kernel Sigmoid gradient failed");

		kernDotProduct << <fullBlocksMult4,blockSize >> > (n*h,dev_hiddenLayerGrad,devGrad,devGrad2);
		checkCUDAErrorFn("Kernel Dot Product failed");

		dim3 fullBlocksMult5((d + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);
		dim3 fullBlocksMult6((h + blockDim.x - 1) / blockDim.x, (d + blockDim.y - 1) / blockDim.y);

		gpu_matrix_transpose << <fullBlocksMult5, blockDim >> > (input, inputTrans, n, d);
		checkCUDAErrorFn("Kernel Transpose for input failed");

		kernMatrixMultiplication << <fullBlocksMult6, blockDim >> > (inputTrans,devGrad2,dev_gradA,d,n,h);
		checkCUDAErrorFn("Kernel Matrix Multiplication for gradA failed");

		/*
		float *check2 = new float[d*h];

		cudaMemcpy(check2, dev_gradA, sizeof(float) * (d*h), cudaMemcpyDeviceToHost);
		checkCUDAErrorFn("Copying data to output failed");

		printf("Grad A \n");
		printArray(d*h, check2, true);
		*/
		/*
		float eta_rate = 0.3;

		float *check = new float[d*h];

		cudaMemcpy(check, dev_gradA, sizeof(float) * (d*h), cudaMemcpyDeviceToHost);
		checkCUDAErrorFn("Copying data to output failed");
		*/
		/*
		printf("Grad A \n");
		printArray(d*h, check, true);


		float *check2 = new float[h*m];

		cudaMemcpy(check2, dev_gradB, sizeof(float) * (h*m), cudaMemcpyDeviceToHost);
		checkCUDAErrorFn("Copying data to output failed");

		printf("Grad B \n");
		printArray(h*m, check2, true);
		*/
		float eta_rate = 0.3;
		dim3 fullBlocksMult7((d*h + blockSize - 1) / blockSize);
		kernUpdateWeights << <fullBlocksMult7,blockSize >> > (d*h,weightsA,newWeightsA,dev_gradA,eta_rate);
		checkCUDAErrorFn("kernel update weights A failed");


		dim3 fullBlocksMult8((h*m + blockSize - 1) / blockSize);
		kernUpdateWeights << <fullBlocksMult8,blockSize >> > (h*m, weightsB, newWeightsB,dev_gradB, eta_rate);
		checkCUDAErrorFn("Kernel update weights B failed");

		cudaFree(dev_gradB);
		cudaFree(gradSoftMax);
		cudaFree(hiddenTrans);
		cudaFree(devGrad);
		cudaFree(devGrad2);
		cudaFree(dev_hiddenLayerGrad);
		cudaFree(inputTrans);
		cudaFree(dev_gradA);
		cudaFree(weightsBTrans);


		//kernMatrixMultiplication << <fullBlocksMult1,(blockSize, blockSize)>> > (hiddenTrans,output- actualOutput,h,n,m);
		//checkCUDAErrorFn("Kernel Matrix Multiplication hiiden and loss failed");

		//gpu_blas_mmul((output - actualOutput), hidden, dev_gradB, m, d, h); //(Still to caclulate mean)
		//gpu_blas_mmul(hidden, (1 - hidden), dev_arr1, h, d, h); // Check the dimensions
		//gpu_blas_mmul(weightsA, dev_arr1, dev_arr2, h, d, h); // Still to look on transpose
		//gpu_blas_mmul((output - actualOutput),input,dev_arr3,m,d,n);// Look into it for transpose

	}
	
	float calculateLoss(int n, float *dev_output, float *dev_actualOutput, float *dev_loss, int m) {
		dim3 fullBlocks1((n + blockSize - 1) / blockSize);
		kernCalculateLoss << <fullBlocks1, blockSize >> > (n, dev_output, dev_actualOutput, dev_loss, m);
		float *loss = new float[n];
		cudaMemcpy(loss, dev_loss, sizeof(float) * (n), cudaMemcpyDeviceToHost);
		checkCUDAErrorFn("Copying data to hidden layer failed");

		float totalLoss = 0;
		for (int i = 0; i < n; i++)
			totalLoss += loss[i];
		return totalLoss;
	}
	void createAndTrainNN(int n,int h,int m,int d, float *idata, float *hidden, float *odata, float *weightsIH, float *weightsHO,float *actualOutput) {		
		
		float *dev_loss;
		int *dev_predict;

		cudaMalloc((void**)&dev_input, (n*d) * sizeof(float));
		checkCUDAErrorFn("Malloc idata into input failed");

		cudaMalloc((void**)&dev_hiddenLayer, (n*h) * sizeof(float));
		checkCUDAErrorFn("Malloc idata into hidden layer failed");

		cudaMalloc((void**)&dev_output, (n*m) * sizeof(float));
		checkCUDAErrorFn("Malloc idata into output failed");

		cudaMalloc((void**)&dev_weightsIH, (d*h) * sizeof(float));
		checkCUDAErrorFn("Malloc idata into weights b/w input & hidden failed");

		cudaMalloc((void**)&dev_weightsHO, (h*m) * sizeof(float));
		checkCUDAErrorFn("Malloc idata into weights b/w hidden & output failed");

		cudaMalloc((void**)&dev_actualOutput, (n*m) * sizeof(float));
		checkCUDAErrorFn("Malloc actual output memeory failed");

		cudaMalloc((void**)&dev_newWeightsIH, (d*h) * sizeof(float));
		checkCUDAErrorFn("Malloc actual output memeory failed");

		cudaMalloc((void**)&dev_newWeightsHO, (h*m) * sizeof(float));
		checkCUDAErrorFn("Malloc actual output memeory failed");

		cudaMalloc((void**)&dev_loss, (n) * sizeof(float));
		checkCUDAErrorFn("Malloc actual output memeory failed");

		cudaMalloc((void**)&dev_predict, (n) * sizeof(float));
		checkCUDAErrorFn("Malloc predict memeory failed");

		cudaMemcpy(dev_input, idata, sizeof(float) * (n*d), cudaMemcpyHostToDevice);
		checkCUDAErrorFn("Copying idata to input failed");

		cudaMemcpy(dev_actualOutput, actualOutput, sizeof(float) * (n*m), cudaMemcpyHostToDevice);
		checkCUDAErrorFn("Copying real output failed failed");

		cudaMemcpy(dev_weightsIH, weightsIH, sizeof(float) * (d*h), cudaMemcpyHostToDevice);
		checkCUDAErrorFn("Copying weights array 1 failed");

		cudaMemcpy(dev_weightsHO, weightsHO , sizeof(float) * (h*m), cudaMemcpyHostToDevice);
		checkCUDAErrorFn("Copying weights array 2 failed");

		//printf("Inside the function \n");

		dim3 fullBlocks1((n + blockSize - 1) / blockSize);
		dim3 fullBlocks2((h + blockSize - 1) / blockSize);
		dim3 fullBlocks3((m + blockSize - 1) / blockSize);
		dim3 fullBlocksMult((m + blockSize - 1) / blockSize);

		//kernMultiplyWeights << <fullBlocks2, blockSize >> > (n,hiddenLayerLen,dev_input,dev_hiddenLayer,dev_weightsIH);

		createNN(dev_input, dev_hiddenLayer, dev_output , dev_weightsIH, dev_weightsHO, n, h, m, d);
		
		float totalLoss;
		totalLoss = calculateLoss(n, dev_output, dev_actualOutput, dev_loss, m);

			//thrust::device_pointer()
			//thrust::inclusive_scan(dev_loss,dev_loss+n,dev_loss);
		/*	float *check = new float[n];

			cudaMemcpy(check, dev_loss, sizeof(float) * (n), cudaMemcpyDeviceToHost);
			checkCUDAErrorFn("Copying data to hidden layer failed");

			printArray(n, check, true);
			*/
			//printf("Total loss: %0.2f\n", totalLoss);
			//float totalError = 1;
			//if (totalLoss > totalError) {

		int iterations = 0;
		float totalError = 0.1;
		
		ofstream outputFile;
		outputFile.open("lossCharacterTrainingLossValues.csv");
		outputFile << "Average Total Loss vs Iterations for Training" << endl;
		while (totalLoss > totalError && iterations < 2000) {
			trainNN(dev_input, dev_hiddenLayer, dev_output, dev_actualOutput, dev_weightsIH, dev_weightsHO, dev_newWeightsIH, dev_newWeightsHO, n, h, m, d);
			dev_weightsIH = dev_newWeightsIH;
			dev_weightsHO = dev_newWeightsHO;
			createNN(dev_input, dev_hiddenLayer, dev_output, dev_weightsIH, dev_weightsHO, n, h, m, d);
			totalLoss = calculateLoss(n, dev_output, dev_actualOutput, dev_loss, m) / n;
			iterations++;
			printf("Iteration: %d \n", iterations);
			printf("Total loss is :%0.3f\n", totalLoss);
			outputFile << totalLoss << endl;
		}
		outputFile.close();
		float *check = new float[n*m];

		cudaMemcpy(check, dev_output, sizeof(float) * (n*m), cudaMemcpyDeviceToHost);
		checkCUDAErrorFn("Copying data to hidden layer failed");

		printArray(n*m, check, true);

		//dim3 fullBlocks4((n + blockSize - 1) / blockSize);
		kernGetAccuracy << < fullBlocks1,blockSize>> > (n*m,dev_output,dev_predict,m);
		checkCUDAErrorFn("Kernel accuracy failed");

		int *predict = new int[n];
		cudaMemcpy(predict, dev_predict, sizeof(float) * (n), cudaMemcpyDeviceToHost);
		checkCUDAErrorFn("Copying predict data failed");

		for (int i = 0; i < n; i++) {
			printf("The outcome for the data point %d is: %d\n", i+1,predict[i]);
		}

		cudaMemcpy(hidden, dev_hiddenLayer, sizeof(float) * (n*h), cudaMemcpyDeviceToHost);
		checkCUDAErrorFn("Copying data to hidden layer failed");

		cudaMemcpy(odata, dev_output, sizeof(float) * (n*m), cudaMemcpyDeviceToHost);
		checkCUDAErrorFn("Copying data to output failed");

		cudaMemcpy(weightsIH, dev_newWeightsIH, sizeof(float) * (d*h), cudaMemcpyDeviceToHost);
		checkCUDAErrorFn("Copying data to hidden layer failed");

		cudaMemcpy(weightsHO, dev_newWeightsHO, sizeof(float) * (h*m), cudaMemcpyDeviceToHost);
		checkCUDAErrorFn("Copying data to hidden layer failed");

		cudaFree(dev_input);
		cudaFree(dev_hiddenLayer);
		cudaFree(dev_output);
		cudaFree(dev_weightsIH);
		cudaFree(dev_weightsHO);
		cudaFree(dev_newWeightsIH);
		cudaFree(dev_newWeightsHO);
		cudaFree(dev_actualOutput);
		cudaFree(dev_actualOutput);
		cudaFree(dev_loss);

	}

}
