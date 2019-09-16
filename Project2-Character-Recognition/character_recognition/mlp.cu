#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "mlp.h"
#include <curand.h>

#define blockSize 128
#define blockWidth 16


namespace CharacterRecognition {
    using Common::PerformanceTimer;
    PerformanceTimer& timer()
    {
        static PerformanceTimer timer;
        return timer;
    }
     
	// Initlialiations
	
	//layers
	double *dev_iLayer;
	double *dev_hLayer;
	double *dev_oLayer;
	double *dev_smaxDen;
	double *dev_losses;
	int *dev_gtruth;

	//weights
	double *dev_w_kj;
	double *dev_w_ji;

	//Derivatives
	double *dev_dL_dw_ji;
	double *dev_dL_dw_kj;
	double *dev_dL_dscores;
	double *dev_dL_dscores_2;

	double *dev_hLayer_T;
	double *dev_iLayer_T;



	
	void printArray(int n, int *a, bool abridged = false) {
		printf("    [ ");
		for (int i = 0; i < n; i++) {
			if (abridged && i + 2 == 15 && n > 16) {
				i = n - 2;
				printf("... ");
			}
			printf("%3d ", a[i]);
		}
		printf("]\n");
	}
	void printFloatArray(int n, double *a, bool abridged = false) {
		printf("    [ ");
		for (int i = 0; i < n; i++) {
			if (abridged && i + 2 == 15 && n > 16) {
				i = n - 2;
				printf("... ");
			}
			printf("%3f ", a[i]);
		}
		printf("]\n");
	}



	// Kernel for Gradient update on Weights
	__global__ void kernUpdateWeights(int N, double *dev_dw, double *dev_w, double LR) {

		int tid = threadIdx.x + blockIdx.x * blockDim.x;

		if (tid < N) {
			dev_w[tid] += -LR * dev_dw[tid];
		}
	}

	// Kernel for derivative of sigmoid
	__global__ void kernGradSigmoid(int N, int C, double *dev_hLayer) {

		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		
		if (tid < N*C) {
			dev_hLayer[tid] = dev_hLayer[tid] * (1 - dev_hLayer[tid]);
		}
	}

	// Matrix Transpose
	__global__ void kernMatrixTranspose(int N, int C, double *matrix, double *matrix_T) {

		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;

		if (col < C && row < N) {
			matrix_T[C*row + col] = matrix[N*col + row];
		}
	}

	// Divide by N
	__global__ void kernDivNdscores(int N, int C, double *dev_dL_dscores) {
		
		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		if (tid < N*C) {
			dev_dL_dscores[tid] /= N;
		}
	}

	// Compute dscores gradient
	__global__ void kernSetdscores(int N, int C,  double *dev_dL_dscores, int *dev_gtruth) {
		int tid = threadIdx.x + blockIdx.x * blockDim.x;

		if (tid < N) {
			dev_dL_dscores[tid*C + dev_gtruth[tid]] -= 1;
		}
	}

	// compute loss per example
	__global__ void kernLossPerN(int N, int C, double* dev_oLayer, int* dev_gtruth, double* dev_losses) {
		int tid = threadIdx.x + blockIdx.x * blockDim.x;

		if (tid < N) {
			dev_losses[tid] = -log(dev_oLayer[tid*C + dev_gtruth[tid]]);
		}
	}

	// kernel to compute exp softmax
	__global__ void kernSoftmax(int N, int C, double* scores, double *sums) {
		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		if (tid < N) {
			for (int i = 0; i < C; i++) {
				scores[tid*C + i] = exp(scores[tid*C + i]) / sums[tid];
			}
		}
	}

	// kernel to exp sum across classes
	__global__ void kernSumRow(int N, int C, double* scores, double *sums) {
		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		if (tid < N) {
			for (int i = 0; i < C; i++) {
				sums[tid] += exp(scores[tid*C + i]);
			}
		}
	}


	// kernel to init weights
	__global__ void kernInitWeights(int N, double* weights) {
		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		
		if (tid < N) {
			weights[tid] = 0.5;
		}

	}

	// kern for sigmoid // f(x) = 1/(1 + e^-x).
	__global__ void kernSigmoid(int N, double *idata) {

		int tid = blockIdx.x * blockDim.x + threadIdx.x;

		if (tid < N) {
			idata[tid] = 1.0 / (1.0 + std::exp(-idata[tid]));
		}
	}
	
	// kern for element wise product 
	__global__ void kernElementProduct(int N, double *matrixA,  double* matrixB, double* matrixC) {

		int tid = blockIdx.x * blockDim.x + threadIdx.x;

		if (tid < N) {
			matrixC[tid] = matrixA[tid] * matrixB[tid];
		}
	}


	// kernel to to matmul // A mxn // B nxk // C mxk
	__global__ void kernMatrixMultiply(const double *dev_A, const double *dev_B, double *dev_C, int m, int n, int k) {

		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;

		double sum = 0;
		if (col < k && row < m)
		{
			for (int i = 0; i < n; i++)
				sum += dev_A[row * n + i] * dev_B[i * k + col];
			dev_C[row * k + col] = sum;
		}
	}
	
	// Dumb reduction
	__global__ void kernReduction(int N, double *dev_losses) {

		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		double sum = 0.0;
		if (tid == 0) {
			for (int i = 0; i < N; i++) {
				sum += dev_losses[tid];
			}
			dev_losses[N-1]=sum;
		}

	}

	void trainMLP(int N, int D, int C, double *idata, int *preds, int *gtruth, int epochs, double *losses, const double LR) {
		
		timer().startGpuTimer();

		// N = number of examples
		// D = dim of each example 
		// C = number of classes

		// NETWORK DEFITION_____________
		// Compute f1		= W1*X1
		// Compute X2		= Sig(f1)
		// Compute Scroes S = W2*X2
		// Compute Probab P = Softmax(S)
		// Compute Loss   L = CEntropy(P)

		//================================================================
		//======================INITIALIZATIONS===========================
		//================================================================
		
		// Allocate input layer
		cudaMalloc((void**)&dev_iLayer, N*D*sizeof(double));
		checkCUDAErrorFn("cudaMalloc dev_iLayer failed!");

		cudaMemcpy(dev_iLayer, idata, N*D*sizeof(double), cudaMemcpyHostToDevice);
		checkCUDAErrorFn("cudaMemcpyToSymbol from idata to dev_iLayer failed!");


		// Allocate hidden layer
		cudaMalloc((void**)&dev_hLayer, N*C* sizeof(double));
		checkCUDAErrorFn("cudaMalloc dev_hLayer failed!");


		// Allocate output layer
		cudaMalloc((void**)&dev_oLayer, N*C* sizeof(double));
		checkCUDAErrorFn("cudaMalloc dev_oLayer failed!");


		// Allocate softmax Den holder
		cudaMalloc((void**)&dev_smaxDen, N* sizeof(double));
		checkCUDAErrorFn("cudaMalloc dev_smaxDen failed!");


		// Allocate losses holder
		cudaMalloc((void**)&dev_losses, N*sizeof(double));
		checkCUDAErrorFn("cudaMalloc dev_losses failed!");


		// Allocate gtruth holder
		cudaMalloc((void**)&dev_gtruth , N * sizeof(int));
		checkCUDAErrorFn("cudaMalloc dev_gtruth failed!");

		cudaMemcpy(dev_gtruth, gtruth, N*sizeof(int), cudaMemcpyHostToDevice);
		checkCUDAErrorFn("cudaMemcpyToSymbol from gtruth to dev_gtruth failed!");


		// Allocate Weights
		cudaMalloc((void**)&dev_w_kj, D*C*sizeof(double));
		checkCUDAErrorFn("cudaMalloc dev_w_kj failed!");

		cudaMalloc((void**)&dev_w_ji, C*C* sizeof(double));
		checkCUDAErrorFn("cudaMalloc dev_w_ji failed!");

		
		// Allocate Derivatives
		cudaMalloc((void**)&dev_dL_dw_kj, D*C* sizeof(double));
		checkCUDAErrorFn("cudaMalloc dev_w_kj failed!");

		cudaMalloc((void**)&dev_dL_dw_ji, C*C* sizeof(double));
		checkCUDAErrorFn("cudaMalloc dev_w_ji failed!");

		cudaMalloc((void**)&dev_dL_dscores, N*C*sizeof(double));
		checkCUDAErrorFn("cudaMalloc dev_dL_dscores failed!");

		cudaMalloc((void**)&dev_dL_dscores_2, N*C * sizeof(double));
		checkCUDAErrorFn("cudaMalloc dev_dL_dscores_2 failed!");

		cudaMalloc((void**)&dev_hLayer_T, N*C* sizeof(double));
		checkCUDAErrorFn("cudaMalloc dev_hLayer_T failed!");
		
		cudaMalloc((void**)&dev_iLayer_T, N*D* sizeof(double));
		checkCUDAErrorFn("cudaMalloc dev_hLayer_T failed!");


		// Init weights kj
		kernInitWeights <<<((D*C + blockSize - 1) / blockSize), blockSize >> > (D*C, dev_w_kj);
		checkCUDAErrorFn("kernInitWeights dev_w_kj failed!");

		// Init weights ji
		kernInitWeights <<<((C*C + blockSize - 1) / blockSize), blockSize >> > (C*C, dev_w_ji);
		checkCUDAErrorFn("kernInitWeights dev_w_ji failed!");

		
		//================================================================
		//======================TRAINING LOOP=============================
		//================================================================
		
		for (int i = 0; i < epochs; i++) {
			
			//================================================================
			//========================= FORWARD ==============================
			
			// STEP 1
			// f1 = W1*X1 (Matrix Mul)
			//=================================
			// dev_hLayer = dev_iLayer*dev_w_kj 
			//   NxC      =    NxD         DxC 

			dim3 dimBlock(blockWidth, blockWidth);
			dim3 dimGrid;
			dimGrid.x = (N + dimBlock.x - 1) / dimBlock.x;
			dimGrid.y = (C + dimBlock.y - 1) / dimBlock.y;
			kernMatrixMultiply <<<dimGrid, dimBlock >>> (dev_iLayer, dev_w_kj, dev_hLayer, N, D, C);

			// Copy back to cpu
			double *tmp = new double[N*C];
			//cudaMemcpy(tmp, dev_hLayer, N*C* sizeof(double), cudaMemcpyDeviceToHost);
			//checkCUDAErrorFn("cudaMemcpyFromSymbol from dev_arrayA to odata failed!");
			//printf("Post matmul\n");
			//printFloatArray(N*C, tmp, true);

			// STEP 2
			// X2         = Sigmoid(f1) 
			//================================
			// dev_hLayer = sigmoid(dev_hLayer)
			//   NxC      =    NxC 
			kernSigmoid <<<((N*C + blockSize - 1) / blockSize), blockSize >> > (N*C, dev_hLayer);


			// Copy back to cpu
			//cudaMemcpy(tmp, dev_hLayer, N*C* sizeof(double), cudaMemcpyDeviceToHost);
			//checkCUDAErrorFn("cudaMemcpyFromSymbol from dev_arrayA to odata failed!");
			//printf("Post sigmoid\n");
			//printFloatArray(N*C, tmp, true);

			// STEP 3
			// Scores S = W2*X2 (Matrix Mul)
			//================================
			// dev_oLayer = dev_hLayer*dev_w_ji 
			//   NxC      =    NxC         CxC
			kernMatrixMultiply <<<dimGrid, dimBlock>>> (dev_hLayer, dev_w_ji, dev_oLayer, N, C, C);
			checkCUDAErrorFn("kernMatrixMultiply failed!");

			// Copy back to cpu
			//cudaMemcpy(tmp, dev_oLayer, N*C * sizeof(double), cudaMemcpyDeviceToHost);
			//checkCUDAErrorFn("cudaMemcpyFromSymbol from dev_arrayA to odata failed!");
			//printf("Post S=W2*x2\n");
			//printFloatArray(N*C, tmp, true);

			// STEP 4
			// P = Softmax(S) 
			//===============
			// dev_smaxDen = Sum_Over_classses(dev_olayer)
			// dev_olayer = dev_olayer/Sum_Over_classses
			//   NxC      =    NxC         1
			kernSumRow<<<((N + blockSize - 1) / blockSize), blockSize >>> (N, C, dev_oLayer, dev_smaxDen);
			kernSoftmax << <((N + blockSize - 1) / blockSize), blockSize >> > (N, C, dev_oLayer, dev_smaxDen);
			checkCUDAErrorFn("kernSumRow or kernSoftmax failed!");

			// Copy back to cpu
			//cudaMemcpy(tmp, dev_smaxDen, N*sizeof(double), cudaMemcpyDeviceToHost);
			//checkCUDAErrorFn("cudaMemcpyFromSymbol from dev_oLayer to tmp failed!");
			//printf("Post dev_smaxDen\n");
			//printFloatArray(N, tmp, true);

			// Copy back to cpu
			cudaMemcpy(tmp, dev_oLayer, N*C * sizeof(double), cudaMemcpyDeviceToHost);
			checkCUDAErrorFn("cudaMemcpyFromSymbol from dev_oLayer to tmp failed!");
			printf("Post Softmax\n");
			printFloatArray(N*C, tmp, true);

			// Compute Loss | Cross Entropy Loss
			//==================================
			// Compute Loss   L = CEntropy(P)
			kernLossPerN<<<((N + blockSize - 1) / blockSize), blockSize >> > (N, C, dev_oLayer, dev_gtruth, dev_losses);
			checkCUDAErrorFn("kernLossPerN  failed!");

			// Copy back to cpu
			cudaMemcpy(tmp, dev_losses, N*sizeof(double), cudaMemcpyDeviceToHost);
			checkCUDAErrorFn("cudaMemcpyFromSymbol from dev_losses to tmp failed!");
			printf("Post dev_losses\n");
			printFloatArray(N, tmp, true);

			// Dumb Reduction
			kernReduction<< <((N + blockSize - 1) / blockSize), blockSize >> > (N, dev_losses);
			// Copy back to cpu
			cudaMemcpy(tmp, dev_losses+N-1, sizeof(double), cudaMemcpyDeviceToHost);
			checkCUDAErrorFn("cudaMemcpyFromSymbol from dev_losses to tmp failed!");
			printf("Epoch: %3d | dev_loss %3f \n", i, tmp[0]);
			
			// Track loss here
			losses[i] = tmp[0];

			//=================================================================
			//========================= BACKPROP ==============================
			
			// STEP 1 : Gradient wrt w_ji
			// dW_ji = Probs_k - [1](gth == k) dev_dL_dscores; 
			cudaMemcpy(dev_dL_dscores, dev_oLayer, N*C*sizeof(double), cudaMemcpyDeviceToDevice);
			checkCUDAErrorFn("cudaMemcpyFromSymbol from probabs to dev_dL_dscores failed!");
			
			kernSetdscores << <((N + blockSize - 1) / blockSize), blockSize >> > (N, C, dev_dL_dscores, dev_gtruth);
			checkCUDAErrorFn("kernSetdscores failed!");

			kernDivNdscores <<<((N*C + blockSize - 1) / blockSize), blockSize >>> (N, C, dev_dL_dscores);
			checkCUDAErrorFn("kernDivNdscores failed!");

			dimGrid.x = (N + dimBlock.x - 1) / dimBlock.x;
			dimGrid.y = (C + dimBlock.y - 1) / dimBlock.y;
			kernMatrixTranspose <<<dimGrid, dimBlock >> > (N, C, dev_hLayer, dev_hLayer_T);

			dimGrid.x = (C + dimBlock.x - 1) / dimBlock.x;
			dimGrid.y = (C + dimBlock.y - 1) / dimBlock.y;
			kernMatrixMultiply << <dimGrid, dimBlock >> > (dev_hLayer_T, dev_dL_dscores, dev_dL_dw_ji, C, N, C);
			checkCUDAErrorFn("kernMatrixMultiply for dev_dL_dw_ji failed!");
			
			// STEP 2 : Gradient wrt w_kj

			// Mul dscores * dev_w_kj == dev_dL_dscores_2
			dimGrid.x = (C + dimBlock.x - 1) / dimBlock.x;
			dimGrid.y = (C + dimBlock.y - 1) / dimBlock.y;
			kernMatrixMultiply << <dimGrid, dimBlock >> > (dev_dL_dscores, dev_w_kj, dev_dL_dscores_2, N, C, C);
			checkCUDAErrorFn("kernMatrixMultiply for dev_dL_dw_ji failed!");

			// compute sig gradient on dev_hlayer
			kernGradSigmoid <<<((N*C + blockSize - 1) / blockSize), blockSize >>> (N, C, dev_hLayer);
			checkCUDAErrorFn("kernGradSigmoid failed!");

			//Element wise mul dev_dL_dscores_2 = dev_dL_dscores_2 . dev_hlayer[sig gradient] 
			kernElementProduct <<<((N*C + blockSize - 1) / blockSize), blockSize >>> (N*C, dev_dL_dscores_2, dev_hLayer, dev_dL_dscores_2);
			checkCUDAErrorFn("kernElementProduct failed!");

			// Transpose X1
			dimGrid.x = (N + dimBlock.x - 1) / dimBlock.x;
			dimGrid.y = (D + dimBlock.y - 1) / dimBlock.y;
			kernMatrixTranspose <<<dimGrid, dimBlock >>> (N, D, dev_iLayer, dev_iLayer_T);

			// matrix Mul 
			dimGrid.x = (C + dimBlock.x - 1) / dimBlock.x;
			dimGrid.y = (C + dimBlock.y - 1) / dimBlock.y;
			kernMatrixMultiply << <dimGrid, dimBlock >> > (dev_iLayer_T, dev_dL_dscores_2, dev_dL_dw_kj, D, N, C);
			checkCUDAErrorFn("kernMatrixMultiply for dev_dL_dw_ji failed!");


			//=================================================================
			//========================= Update Weights=========================

			// Update weights kj
			kernUpdateWeights << <((D*C + blockSize - 1) / blockSize), blockSize >> > (D*C, dev_dL_dw_kj, dev_w_kj, LR);
			checkCUDAErrorFn("kernInitWeights dev_w_kj failed!");

			// InitUpdate weights ji
			kernUpdateWeights << <((C*C + blockSize - 1) / blockSize), blockSize >> > (C*C, dev_dL_dw_ji, dev_w_ji, LR);
			checkCUDAErrorFn("kernInitWeights dev_w_ji failed!");
			
			// COntinue to next epoch 
			double *tmp2 = new double[D*D];
			cudaMemcpy(tmp2, dev_dL_dw_kj, D*C*sizeof(double), cudaMemcpyDeviceToHost);
			checkCUDAErrorFn("dev_dL_dw_kj memcopy failed!");
			printFloatArray(D*C, tmp2, true);
			cudaMemcpy(tmp2, dev_dL_dw_ji, C*C * sizeof(double), cudaMemcpyDeviceToHost);
			checkCUDAErrorFn("dev_dL_dw_ji memcopy failed!");
			printFloatArray(C*C, tmp2, true);

			printf("\n");
		}


		printf("Finished training.\n");
		
		timer().endGpuTimer();
		}
}
