#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "mlp.h"

#include <curand.h>
#include <curand_kernel.h>

#define blockSize 128
#define blockWidth 16


namespace CharacterRecognition {
	using Common::PerformanceTimer;
	PerformanceTimer& timer()
	{
		static PerformanceTimer timer;
		return timer;
	}

	//=====Initlialiations=======

	//layers
	double *dev_iLayer;
	double *dev_hLayer;
	double *dev_oLayer;

	double *dev_losses;
	double *dev_LossAvg;

	// gtruth and preds
	int *dev_gtruth;
	int *dev_preds;
	double * dev_preds_probab;

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
	double *dev_w_ji_T;


	//=============================================
	// Rnadom Number Generation using cuRand on GPU
	//=============================================
	curandState *devState;

	__global__ void kernInitCurand(curandState *state, int N, unsigned long seed) {
		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		if (tid < N) {
			curand_init(seed, tid, 0, &state[tid]);
		}
	}

	__global__ void KernGenRand(curandState *state, int N, double *w) {
		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		if (tid < N) {
			w[tid] = (2.0*curand_uniform(&state[tid]) - 1.0); // Between -1 and 1
		}
	}

	//===================================================================
	//=====KERNEL DEFNITIONS FOR Forward and Backward====================
	//===================================================================


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
			printf("%0.2f ", a[i]);
		}
		printf("]\n");
	}


	// Kernel for Gradient update on Weights
	__global__ void kernUpdateWeights(int N, double *dev_dw, double *dev_w, double LR) {

		int tid = threadIdx.x + blockIdx.x * blockDim.x;

		if (tid < N) {
			dev_w[tid] = dev_w[tid] - (LR * dev_dw[tid]);
		}
	}

	// Kernel for derivative of sigmoid
	__global__ void kernGradSigmoid(int N, int H, double *dev_hLayer) {

		int tid = threadIdx.x + blockIdx.x * blockDim.x;

		if (tid < N*H) {
			dev_hLayer[tid] = dev_hLayer[tid] * (1 - dev_hLayer[tid]);
		}
	}

	// Matrix Transpose
	__global__ void kernMatrixTranspose(int rows, int cols, double *matrix, double *matrix_T) {

		int idy = blockIdx.y * blockDim.y + threadIdx.y;
		int idx = blockIdx.x * blockDim.x + threadIdx.x;

		if (idx < cols && idy < rows) {
			int pos = idy * cols + idx;
			int tpos = idx * rows + idy;

			matrix_T[tpos] = matrix[pos];
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
	__global__ void kernSetdscores(int N, int C, double *dev_dL_dscores, int *dev_gtruth) {
		int tid = threadIdx.x + blockIdx.x * blockDim.x;

		if (tid < N) {
			dev_dL_dscores[tid*C + dev_gtruth[tid]] -= 1;
		}
	}

	// compute predictions
	__global__ void kernPredsN(int N, int C, double* dev_oLayer, int* dev_gtruth, int* dev_preds, double * dev_preds_probab) {
		int tid = threadIdx.x + blockIdx.x * blockDim.x;

		if (tid < N) {
			dev_preds[tid] = dev_oLayer[tid*C + dev_gtruth[tid]] > 0.5 ? dev_gtruth[tid] : (dev_gtruth[tid] == 0 ? 1 : 0);
			dev_preds_probab[tid] = dev_oLayer[tid*C + dev_gtruth[tid]];
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
	__global__ void kernSoftmax(int N, int C, double* scores) {
		int tid = threadIdx.x + blockIdx.x * blockDim.x;

		if (tid < N) {
			double sums = 0.0;

			for (int i = 0; i < C; i++) {
				sums += exp(scores[tid*C + i]);
			}

			for (int i = 0; i < C; i++) {
				scores[tid*C + i] = exp(scores[tid*C + i]) / sums;
			}
		}
	}

	// kern for sigmoid // f(x) = 1/(1 + e^-x).
	__global__ void kernSigmoid(int N, double *idata) {

		int tid = blockIdx.x * blockDim.x + threadIdx.x;

		if (tid < N) {
			idata[tid] = 1.0 / (1.0 + exp(-1*idata[tid]));
		}
	}

	// kern for element wise product 
	__global__ void kernElementProduct(int N, double *matrixA, double* matrixB) {

		int tid = blockIdx.x * blockDim.x + threadIdx.x;

		if (tid < N) {
			matrixA[tid] = matrixA[tid] * matrixB[tid];
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
	__global__ void kernReduction(int N, double *dev_losses, double *dev_LossAvg) {

		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		double sum = 0.0;
		if (tid == 0) {
			for (int i = 0; i < N; i++) {
				sum += dev_losses[i];
			}
			dev_LossAvg[0] = sum / N;
		}

	}

	// Ele wise addition A = A+B
	__global__ void kernAddition(int N, double *dev_A, double *dev_B) {

		int tid = blockIdx.x * blockDim.x + threadIdx.x;

		if (tid < N) {
			dev_A[tid] += dev_B[tid];
		}

	}

	void trainMLP(int N, int D, int H, int C, double *idata, int *preds, int *gtruth, int epochs, 
		          double *lossAvgPerEpoch, const double LR, double *w1, double *w2, unsigned long seed) {

		timer().startGpuTimer();

		// N = number of examples
		// D = dim of each example 
		// H = Hidden state nodes
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
		cudaMalloc((void**)&dev_iLayer, N*D * sizeof(double));
		checkCUDAErrorFn("cudaMalloc dev_iLayer failed!");

		cudaMemcpy(dev_iLayer, idata, N*D * sizeof(double), cudaMemcpyHostToDevice);
		checkCUDAErrorFn("cudaMemcpyToSymbol from idata to dev_iLayer failed!");


		// Allocate hidden layer
		cudaMalloc((void**)&dev_hLayer, N*H* sizeof(double));
		checkCUDAErrorFn("cudaMalloc dev_hLayer failed!");


		// Allocate output layer
		cudaMalloc((void**)&dev_oLayer, N*C* sizeof(double));
		checkCUDAErrorFn("cudaMalloc dev_oLayer failed!");


		// Allocate losses holder
		cudaMalloc((void**)&dev_losses, N * sizeof(double));
		checkCUDAErrorFn("cudaMalloc dev_losses failed!");

		cudaMalloc((void**)&dev_LossAvg, 1*sizeof(double));
		checkCUDAErrorFn("cudaMalloc dev_LossAvg failed!");


		// Allocate gtruth and preds
		cudaMalloc((void**)&dev_gtruth, N * sizeof(int));
		checkCUDAErrorFn("cudaMalloc dev_gtruth failed!");

		cudaMemcpy(dev_gtruth, gtruth, N * sizeof(int), cudaMemcpyHostToDevice);
		checkCUDAErrorFn("cudaMemcpyToSymbol from gtruth to dev_gtruth failed!");

		cudaMalloc((void**)&dev_preds, N * sizeof(int));
		checkCUDAErrorFn("cudaMalloc dev_preds failed!");

		cudaMalloc((void**)&dev_preds_probab, N * sizeof(double));
		checkCUDAErrorFn("cudaMalloc dev_preds_probab failed!");

		// Allocate Weights
		cudaMalloc((void**)&dev_w_kj, D*H * sizeof(double)); //w1
		checkCUDAErrorFn("cudaMalloc dev_w_kj failed!");

		cudaMalloc((void**)&dev_w_ji, C*H * sizeof(double)); //w2
		checkCUDAErrorFn("cudaMalloc dev_w_ji failed!");


		// Allocate Derivatives
		cudaMalloc((void**)&dev_dL_dw_kj, D*H * sizeof(double)); //dw1
		checkCUDAErrorFn("cudaMalloc dev_w_kj failed!");

		cudaMalloc((void**)&dev_dL_dw_ji, C*H * sizeof(double)); //dw2
		checkCUDAErrorFn("cudaMalloc dev_w_ji failed!");

		cudaMalloc((void**)&dev_dL_dscores, N*C * sizeof(double));
		checkCUDAErrorFn("cudaMalloc dev_dL_dscores failed!");

		cudaMalloc((void**)&dev_dL_dscores_2, N*C * sizeof(double));
		checkCUDAErrorFn("cudaMalloc dev_dL_dscores_2 failed!");


		// Allocate transposes
		cudaMalloc((void**)&dev_hLayer_T, N*H * sizeof(double));
		checkCUDAErrorFn("cudaMalloc dev_hLayer_T failed!");

		cudaMalloc((void**)&dev_iLayer_T, N*D * sizeof(double));
		checkCUDAErrorFn("cudaMalloc dev_hLayer_T failed!");

		cudaMalloc((void**)&dev_w_ji_T, C*H * sizeof(double));
		checkCUDAErrorFn("cudaMalloc dev_w_ji_T failed!");

		//==============================
		// Initialise Weights
		//==============================
		cudaMalloc((void**)&devState, H*D * sizeof(curandState));

		kernInitCurand << <((D*H + blockSize - 1) / blockSize), blockSize >> > (devState, D*H, seed);
		checkCUDAErrorFn("KernInitCurand failed!");
		KernGenRand << <((D*H + blockSize - 1) / blockSize), blockSize >> > (devState, D*H, dev_w_kj);//w1
		checkCUDAErrorFn("KernGenRand dev_w_kj failed!");

		kernInitCurand << <((H*C + blockSize - 1) / blockSize), blockSize >> > (devState, H*C, seed); 
		checkCUDAErrorFn("KernInitCurand failed!");
		KernGenRand << <((H*C + blockSize - 1) / blockSize), blockSize >> > (devState, H*C, dev_w_ji);//w2
		checkCUDAErrorFn("KernGenRand dev_w_kj failed!");


		//================================================================
		//======================TRAINING LOOP=============================
		//================================================================
		double *tmp = new double[N*D];
		double *tmp2 = new double[N*D];
		double *lossesN = new double[N];
		printf("--------------------------------------------\n");
		printf("One Hidden Layer MLP | Configuration \n");
		printf("--------------------------------------------\n");
		printf("Number of Examples            | N = %d \n",N);
		printf("Dimensionality of each Example| D = %d \n",D);
		printf("Number of Hidden Layer Nodes  | H = %d \n",H);
		printf("Total Number of Classes       | C = %d \n",C);
		printf("Activation    = Sigmoid \n");
		printf("Loss Function = Cross Entropy \n");
		printf("--------------------------------------------\n");

		//printf("\nInput DATA  ");
		//printf("\nInput DATA  ");
		//printFloatArray(N*D, idata, true);
		dim3 dimBlock(blockWidth, blockWidth);
		dim3 dimGrid;

		for (int i = 0; i < epochs; i++) {

			//================================================================
			//========================= FORWARD ==============================

			// STEP 1
			// f1 = W1*X1 (Matrix Mul)
			//=================================
			// dev_hLayer = dev_iLayer*dev_w_kj 
			//   NxH      =    NxD         DxH 
			dimGrid.x = (H + dimBlock.x - 1) / dimBlock.x;
			dimGrid.y = (N + dimBlock.y - 1) / dimBlock.y;
			kernMatrixMultiply << <dimGrid, dimBlock >> > (dev_iLayer, dev_w_kj, dev_hLayer, N, D, H);

			// STEP 2
			// X2         = Sigmoid(f1) 
			// dev_hLayer = sigmoid(dev_hLayer)
			//   NxH     =    NxH 
			kernSigmoid << <((N*H + blockSize - 1) / blockSize), blockSize >> > (N*H, dev_hLayer);


			// STEP 3
			// Scores S = W2*X2 (Matrix Mul)
			// dev_oLayer = dev_hLayer*dev_w_ji 
			//   NxC      =    NxH         HxC
			dimGrid.x = (C + dimBlock.x - 1) / dimBlock.x;
			dimGrid.y = (N + dimBlock.y - 1) / dimBlock.y;
			kernMatrixMultiply << <dimGrid, dimBlock >> > (dev_hLayer, dev_w_ji, dev_oLayer, N, H, C);
			checkCUDAErrorFn("kernMatrixMultiply failed!");


			// STEP 4
			// P = Softmax(S) 
			// dev_smaxDen = Sum_Over_classses(dev_olayer)
			// dev_olayer = dev_olayer/Sum_Over_classses
			//   NxC      =    NxC         1
			kernSoftmax << <((N + blockSize - 1) / blockSize), blockSize >> > (N, C, dev_oLayer);
			checkCUDAErrorFn("kernSoftmax failed!");

			// STEP 5
			// Compute Losses | Cross Entropy Loss
			kernLossPerN << <((N + blockSize - 1) / blockSize), blockSize >> > (N, C, dev_oLayer, dev_gtruth, dev_losses);
			checkCUDAErrorFn("kernLossPerN  failed!");

			// Cpoy loss to CPU
			//cudaMemcpy(lossesN, dev_losses, N * sizeof(double), cudaMemcpyDeviceToHost);
			//checkCUDAErrorFn("cudaMemcpyFromSymbol from dev_losses to lossesN failed!");
			//printf("Post dev_losses [Loss = CEntropy(P)]\n");
			//printFloatArray(N, lossesN, true);


			// Compute Predictions
			kernPredsN << <((N + blockSize - 1) / blockSize), blockSize >> > (N, C, dev_oLayer, dev_gtruth, dev_preds, dev_preds_probab);
			cudaMemcpy(preds, dev_preds, N * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAErrorFn("cudaMemcpyDeviceToHost from dev_preds to preds failed!");
			cudaMemcpy(tmp2, dev_preds_probab, N * sizeof(double), cudaMemcpyDeviceToHost);
			checkCUDAErrorFn("cudaMemcpyDeviceToHost from dev_preds_probab to tmp failed!");


			// STEP 5.2
			// Compute Avg of Losses
			kernReduction << <((N + blockSize - 1) / blockSize), blockSize >> > (N, dev_losses, dev_LossAvg);
			// Copy back to cpu
			cudaMemcpy(lossAvgPerEpoch + i, dev_LossAvg, sizeof(double), cudaMemcpyDeviceToHost);
			checkCUDAErrorFn("cudaMemcpyFromSymbol from dev_LossAvg to tmp failed!");
			
			if (i % 1000 == 0) {
				printf("Epoch : %3d | LossAvg %3f \n", i, lossAvgPerEpoch[i]);
				printf("GroundTruth :");
				printArray(N, gtruth, true);
				printf("Predictions :");
				printArray(N, preds, true);
				printf("Confidence  :");
				printFloatArray(N, tmp2, true);
				printf("\n");
			}

			//=================================================================
			//========================= BACKPROP ==============================

			//===============================
			// STEP 1 : Gradient wrt w_kj W2
			//===============================
			// dW_ji = Probs_k - [1](gth == k) dev_dL_dscores;

			cudaMemcpy(dev_dL_dscores, dev_oLayer, N*C * sizeof(double), cudaMemcpyDeviceToDevice);
			checkCUDAErrorFn("cudaMemcpyFromSymbol from probabs to dev_dL_dscores failed!");

			kernSetdscores << <((N + blockSize - 1) / blockSize), blockSize >> > (N, C, dev_dL_dscores, dev_gtruth);
			checkCUDAErrorFn("kernSetdscores failed!");

			kernDivNdscores << <((N*C + blockSize - 1) / blockSize), blockSize >> > (N, C, dev_dL_dscores);
			checkCUDAErrorFn("kernDivNdscores failed!");

			dimGrid.x = (H + dimBlock.x - 1) / dimBlock.x;
			dimGrid.y = (N + dimBlock.y - 1) / dimBlock.y;
			kernMatrixTranspose << <dimGrid, dimBlock >> > (N, H, dev_hLayer, dev_hLayer_T);

			dimGrid.x = (C + dimBlock.x - 1) / dimBlock.x;
			dimGrid.y = (H + dimBlock.y - 1) / dimBlock.y;
			kernMatrixMultiply << <dimGrid, dimBlock >> > (dev_hLayer_T, dev_dL_dscores, dev_dL_dw_ji, H, N, C);
			checkCUDAErrorFn("kernMatrixMultiply for dev_dL_dw_ji failed!");

			//===============================
			// STEP 2 : Gradient wrt w_kj W1
			//===============================

			// Transpose Wji (W2)
			dimGrid.x = (C + dimBlock.x - 1) / dimBlock.x;
			dimGrid.y = (H + dimBlock.y - 1) / dimBlock.y;
			kernMatrixTranspose << <dimGrid, dimBlock >> > (H, C, dev_w_ji, dev_w_ji_T);

			// Transpose Input Data
			dimGrid.x = (D + dimBlock.x - 1) / dimBlock.x;
			dimGrid.y = (N + dimBlock.y - 1) / dimBlock.y;
			kernMatrixTranspose << <dimGrid, dimBlock >> > (N, D, dev_iLayer, dev_iLayer_T);

			// Mul dev_dL_dscores * dev_w_kj_T == dev_dL_dscores_2
			//             NxC          CxH             NxH
			dimGrid.x = (H + dimBlock.x - 1) / dimBlock.x;
			dimGrid.y = (N + dimBlock.y - 1) / dimBlock.y;
			kernMatrixMultiply << <dimGrid, dimBlock >> > (dev_dL_dscores, dev_w_ji_T, dev_dL_dscores_2, N, C, H);
			checkCUDAErrorFn("kernMatrixMultiply for dev_dL_dscores_2 failed!");

			// compute sig gradient on dev_hlayer N*H [IN PLACE]
			kernGradSigmoid << <((N*H + blockSize - 1) / blockSize), blockSize >> > (N, H, dev_hLayer);
			checkCUDAErrorFn("kernGradSigmoid failed!");

			//Element wise mul dev_dL_dscores_2 [INPLACE] = dev_dL_dscores_2 . dev_hlayer[sig gradient] 
			kernElementProduct << <((N*H + blockSize - 1) / blockSize), blockSize >> > (N*H, dev_dL_dscores_2, dev_hLayer);
			checkCUDAErrorFn("kernElementProduct failed!");

			// matrix Mul final with Xi_T
			dimGrid.x = (H + dimBlock.x - 1) / dimBlock.x;
			dimGrid.y = (D + dimBlock.y - 1) / dimBlock.y;
			kernMatrixMultiply << <dimGrid, dimBlock >> > (dev_iLayer_T, dev_dL_dscores_2, dev_dL_dw_kj, D, N, H);
			checkCUDAErrorFn("kernMatrixMultiply for dev_dL_dw_kj failed!");


			//=================================================================
			// STEP 3 : Update Weights ========================================
			//=================================================================

			// Update weights kj W1
			kernUpdateWeights << <((D*H + blockSize - 1) / blockSize), blockSize >> > (D*H, dev_dL_dw_kj, dev_w_kj, LR);
			checkCUDAErrorFn("kernUpdateWeights dev_w_kj failed!");

			// InitUpdate weights ji W2
			kernUpdateWeights << <((H*C + blockSize - 1) / blockSize), blockSize >> > (H*C, dev_dL_dw_ji, dev_w_ji, LR);
			checkCUDAErrorFn("kernUpdateWeights dev_w_ji failed!");

			//printf("\n-----------------------------------------------------\n\n");
		}


		printf("Finished training.\n");
		float count = 0.0;
		for (int n = 0; n < N; n++) {
			if (preds[n] == gtruth[n]) {
				count += 1;
			}
		}
		float acc = count / N;
		printf("Accuracy: %0.2f Percent \n", acc*100.0);

		// SAVE WEIGHTS
		cudaMemcpy(w1, dev_w_kj, H*D*sizeof(double), cudaMemcpyDeviceToHost);
		checkCUDAErrorFn("cudaMemcpyFromSymbol from  dev_w_kj to w1 failed!");

		cudaMemcpy(w2, dev_w_ji, H*C*sizeof(double), cudaMemcpyDeviceToHost);
		checkCUDAErrorFn("cudaMemcpyFromSymbol from  dev_w_ji to w2 failed!");

		//printf("losses:\n");
		//printFloatArray(epochs, lossAvgPerEpoch, true);

		//====================
		// CleanUp
		//====================
		cudaFree(dev_iLayer);
		cudaFree(dev_hLayer);
		cudaFree(dev_oLayer);

		cudaFree(dev_losses);

		cudaFree(dev_gtruth);
		cudaFree(dev_preds);
		cudaFree(dev_preds_probab);

		cudaFree(dev_w_kj);
		cudaFree(dev_w_ji);

		cudaFree(dev_dL_dw_ji);
		cudaFree(dev_dL_dw_kj);

		cudaFree(dev_dL_dscores);
		cudaFree(dev_dL_dscores_2);

		cudaFree(dev_hLayer_T);
		cudaFree(dev_iLayer_T);

		delete(tmp);
		delete(tmp2);
		delete(lossesN);

		timer().endGpuTimer();
	}
}
