#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <math_constants.h>
#include "common.h"
#include "mlp.h"

#define ALLOWKERNEL5 0
#define RANDSEED 0x0bad1bad2bad123
#define LAMBDA 0.1 //the learning delta

//These are definitions for index math in the 1d-2d world
#define UL(idx, w) (idx - w - 1)
#define UC(idx, w) (idx - w)
#define UR(idx, w) (idx - w + 1)
#define CL(idx, w) (idx - 1)
#define CC(idx, w) (idx)
#define CR(idx, w) (idx + 1)
#define DL(idx, w) (idx + w - 1)
#define DC(idx, w) (idx + w)
#define DR(idx, w) (idx + w + 1)


namespace CharacterRecognition {
    using Common::PerformanceTimer;
    PerformanceTimer& timer()
    {
        static PerformanceTimer timer;
        return timer;
    }

	//##################################
	// SIZE DEFINES
	//##################################

#define NUMFILTERS 6
#define KERNWIDTH 3
#define POOLWIDTH 3

#define F0SIZE 10201

#define SINCONVRAWSIZE ((sqrt(F0SIZE) - (KERNWIDTH - 1)) * (sqrt(F0SIZE) - (KERNWIDTH - 1)))
#define CONVRAWSIZE (SINCONVRAWSIZE * NUMFILTERS)
#define SINCONVPOOLSIZE (CONVRAWSIZE / (POOLWIDTH * POOLWIDTH))
#define CONVPOOLSIZE (SINCONVPOOLSIZE * NUMFILTERS)
#define F1SIZE (CONVPOOLSIZE)

#define W1SIZE (F0SIZE * F1SIZE)

#define F2SIZE 500
#ifndef RSIZE
#define RSIZE 52
#endif
#define W2SIZE (F2SIZE * RSIZE)



	//##################################
	// DEVICE POINTER MEMORY
	//##################################

	float* dF0;//features 0 (orig data)
	float* dC0;//convolutional memory for first layer
	float* dF1;//features 1
	float* dW1;//weights 1
	float* dW1D;//delta value for weights 1
	float* dPj;//psi_j result matrix
	float* dOj;//omega_j result matrix
	float* dF2;//features 2
	float* dF2A;//features 2 (activated)
	float* dW2;//weights 2
	float* dW2D;//delta value for weights 2
	float* dPi;//psi_i result matrix
	float* dR;//result
	float* dRA;//result(activated)
	float* dRE;//result error



	//CONVOLUTIONAL MEMORY
	float* dCR;//convolutional output, raw
	float* dCP;//convolutional output, pooled

	//Convolution kernel initialization
	filter3 kern1 = {	1.0 / 16,	1.0 / 8,	1.0 / 16,
						1.0 / 8,	1.0 / 4,	1.0 / 8,
						1.0 / 16,	1.0 / 8,	1.0 / 16 };//gaussian
	filter3 kern2 = { -1, -1, -1, -1, 8, -1, -1, -1, -1 };//outline
	filter3 kern3 = { 1, 2, 1, 0, 0, 0, -1, -2, -1 };//sobel top
	filter3 kern4 = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };//sobel right
	filter3 kern5 = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };//sobel bottom
	filter3 kern6 = { 1, 0, -1, 2, 0, -2, 1, 0, -1 };//sobel left
	filter3 allKernels[NUMFILTERS] = { kern1, kern2, kern3, kern4, kern5, kern6 };

	//##################################
	// FUNCTION DELCARATIONS
	//##################################

	/**
	Gets the "index" for the thread
	Currently, only supporting single-dimensional block indexes
	Computes all relevant x, y, z transformations
	*/
	__device__ int getIndex();

	//##################################
	// DEVICE POINTER MALLOC AND FREE
	//##################################

	void kmallocBuffers() {
		cudaMalloc((void**)& dF0, F0SIZE * sizeof(float));
		checkCUDAErrorFn("cudaMalloc failed\n", NULL, __LINE__);
		cudaMalloc((void**)& dC0, CONVRAWSIZE * sizeof(float));
		checkCUDAErrorFn("cudaMalloc failed\n", NULL, __LINE__);
		cudaMalloc((void**)& dF1, F1SIZE * sizeof(float));
		checkCUDAErrorFn("cudaMalloc failed\n", NULL, __LINE__);
		cudaMalloc((void**)& dW1, W1SIZE * sizeof(float));
		checkCUDAErrorFn("cudaMalloc failed\n", NULL, __LINE__);
		cudaMalloc((void**)& dW1D, W1SIZE * sizeof(float));
		checkCUDAErrorFn("cudaMalloc failed\n", NULL, __LINE__);
		cudaMalloc((void**)& dPj, F2SIZE * sizeof(float));
		checkCUDAErrorFn("cudaMalloc failed\n", NULL, __LINE__);
		cudaMalloc((void**)& dOj, F2SIZE * sizeof(float));
		checkCUDAErrorFn("cudaMalloc failed\n", NULL, __LINE__);
		cudaMalloc((void**)& dF2, F2SIZE * sizeof(float));
		checkCUDAErrorFn("cudaMalloc failed\n", NULL, __LINE__);
		cudaMalloc((void**)& dF2A, F2SIZE * sizeof(float));
		checkCUDAErrorFn("cudaMalloc failed\n", NULL, __LINE__);
		cudaMalloc((void**)& dW2, W2SIZE *sizeof(float));
		checkCUDAErrorFn("cudaMalloc failed\n", NULL, __LINE__);
		cudaMalloc((void**)& dW2D, W2SIZE * sizeof(float));
		checkCUDAErrorFn("cudaMalloc failed\n", NULL, __LINE__);
		cudaMalloc((void**)& dPi, RSIZE * sizeof(float));
		checkCUDAErrorFn("cudaMalloc failed\n", NULL, __LINE__);
		cudaMalloc((void**)& dR, RSIZE * sizeof(float));
		checkCUDAErrorFn("cudaMalloc failed\n", NULL, __LINE__);
		cudaMalloc((void**)& dRE, RSIZE * sizeof(float));
		checkCUDAErrorFn("cudaMalloc failed\n", NULL, __LINE__);
		cudaMalloc((void**)& dRA, RSIZE * sizeof(float));
		checkCUDAErrorFn("cudaMalloc failed\n", NULL, __LINE__);

	}//kmallocBuffers

	void kfreeBuffers() {
		cudaFree(dF0);
		checkCUDAErrorFn("cudaFree failed\n", NULL, __LINE__);
		cudaFree(dC0);
		checkCUDAErrorFn("cudaFree failed\n", NULL, __LINE__);
		cudaFree(dF1);
		checkCUDAErrorFn("cudaFree failed\n", NULL, __LINE__);
		cudaFree(dW1);
		checkCUDAErrorFn("cudaFree failed\n", NULL, __LINE__);
		cudaFree(dW1D);
		checkCUDAErrorFn("cudaFree failed\n", NULL, __LINE__);
		cudaFree(dPj);
		checkCUDAErrorFn("cudaFree failed\n", NULL, __LINE__);
		cudaFree(dOj);
		checkCUDAErrorFn("cudaFree failed\n", NULL, __LINE__);
		cudaFree(dF2);
		checkCUDAErrorFn("cudaFree failed\n", NULL, __LINE__);
		cudaFree(dF2A);
		checkCUDAErrorFn("cudaFree failed\n", NULL, __LINE__);
		cudaFree(dW2);
		checkCUDAErrorFn("cudaFree failed\n", NULL, __LINE__);
		cudaFree(dW2D);
		checkCUDAErrorFn("cudaFree failed\n", NULL, __LINE__);
		cudaFree(dPi);
		checkCUDAErrorFn("cudaFree failed\n", NULL, __LINE__);
		cudaFree(dR);
		checkCUDAErrorFn("cudaFree failed\n", NULL, __LINE__);
		cudaFree(dRE);
		checkCUDAErrorFn("cudaFree failed\n", NULL, __LINE__);
		cudaFree(dRA);
		checkCUDAErrorFn("cudaFree failed\n", NULL, __LINE__);

	}//kfreeBuffers
        
	//##################################
	// DEVICE FUNCTIONS
	//##################################

	__device__ int getIndex() {
		int threadIndex = threadIdx.x + (blockDim.x) * threadIdx.y + (blockDim.y * blockDim.x) * threadIdx.z;
		int overallIndex = threadIndex + blockIdx.x * (blockDim.x * blockDim.y * blockDim.z);

		return overallIndex;
	}//getIndex

	//##################################
	// DEVICE GLOBAL FUNCTIONS
	//##################################

	__global__ void kTranspose(float* A, float* Aswap, int m, int n) {
		int index = getIndex();
		if (index >= m * n) return;

		int srcR = index / n;
		int srcC = index % n;
		int dstR = srcC;
		int dstC = srcR;
		//int srcIndex = srcR * n + srcC;
		//int dstIndex = dstR * m + dstC;
		//Aswap[dstIndex] = A[srcIndex];
		Aswap[dstR * m + dstC] = A[srcR * n + srcC];
	}//kTranspose

	/**
	Performs our activation function on our results to put them in the range between 0 and 1
	Does so in-place
	*/
	__global__ void kActivateResults(float* results, float* resultsA, int N) {
		int index = getIndex();
		if (index >= N) return;
		resultsA[index] = 1.0 / (1.0 + expf(-1 * results[index]));
	}//activateResults

	__global__ void kActivateInverse(float* results, float* resultsIA, int N) {
		int index = getIndex();
		if (index >= N) return;
		//resultsIA[index] = logf(results[index] / (1.0 - results[index]));
		float ex = expf(results[index]);
		resultsIA[index] = ex / ((ex + 1) * (ex + 1));
	}//kActivateInverse




	//##################################
	// HOST HELPER FUNCTIONS
	//##################################

	void activateResults(float* results, float* resultsActivated, int numResults) {
		dim3 tpb = dim3(BLOCKSIZE);
		dim3 bpg = dim3((numResults + BLOCKSIZE - 1) / BLOCKSIZE);

		kActivateResults<<<bpg, tpb>>>(results, resultsActivated, numResults);
		checkCUDAErrorFn("kActivateResults failed\n", NULL, __LINE__);
	}//activateResults

	__global__ void shiftByFactor(float* A, int N, float mulFactor, float offset) {
		int index = getIndex();
		if (index > N) return;
		A[index] = mulFactor * A[index] + offset;
	}//shiftByFactor

	void gpuFillRand(float* A, int nr_rows_A, int nr_cols_A, float lo, float hi){
		curandGenerator_t prng;
		curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_XORWOW);

		curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) RANDSEED);

		curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A);

		//shift the random numbers into the given range
		float mulFactor = hi - lo;
		float offset = lo;
		
		int numElements = nr_rows_A * nr_cols_A;
		dim3 tpb = dim3(BLOCKSIZE);
		dim3 bpg = dim3((numElements + BLOCKSIZE - 1) / BLOCKSIZE);

		shiftByFactor<<<bpg, tpb>>>(A, numElements, mulFactor, offset);
		checkCUDAErrorFn("shiftByFactor failed\n", NULL, __LINE__);
		cudaDeviceSynchronize();//safety
	}//gpuFillRand



	void matMul(cublasHandle_t* handle, const float* A, const float* B, float* C, int m, int k, int n) {

		//Since cublas expects column-major indexing, our A is effectively AT (kxm), and our B is effectively BT (nxk)
		//As such, we're going to be doing BT * AT = CT (nxm)
		//Then, we transpose C "in place" before we return
		//And by that I mean we don't do that, because for some reason the multiplication works how I want
		float alpha = 1.0;
		float beta = 0.0;

		//Future development: put the result into Cswap, transpose into C
		cublasSgemm(*handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B, n, A, k, &beta, C, n) ;
		checkCUDAErrorFn("the internal matrix multiply failed\n", NULL, __LINE__);
		//cudaDeviceSynchronize();

		//no need to transpose?? not sure why, but this function operates
		//transpose(C, Cswap, n, m);
	}//matMul

	//##################################
	// ERROR CALCULATIONS (?)
	//##################################

	float_v calcErrorSingle(InputData record, float* resultArray, float* kResultArray) {
		float_v retval = float_v();
		float_v trueResult = record.resultArray;
		for (int i = 0; i < trueResult.size(); i++) {
			float error = resultArray[i] - trueResult[i];
			retval.push_back(error);
		}//for

		if (kResultArray) {
			cudaMemcpy(kResultArray, retval.data(), trueResult.size() * sizeof(float), cudaMemcpyHostToDevice);
			checkCUDAErrorFn("cudaMemcpy failed\n", NULL, __LINE__);
		}//if

		return retval;
	}//calcError

	float_v calcSumSquareErrors(float_vv errorVals) {
		float_v result = float_v(errorVals[0].size(), 0.0f);
		for (int i = 0; i < errorVals.size(); i++) {
			for (int j = 0; j < errorVals[0].size(); j++) {
				result[j] += errorVals[i][j] * errorVals[i][j] / 2.0;
			}//for j
		}//for i

		return result;
	}//calcSumSquareErrors

	float calcEnergy(float_v errors) {
		float sum = 0;
		for (int i = 0; i < errors.size(); i++) {
			sum += (errors[i] * errors[i]);
		}//for
		return sum / errors.size();//averaging the energy function?
	}//calcEnergy

	//##################################
	// WEIGHT CHANGES
	//##################################

	__global__ void kCalcWeightChange1(float* thetaA, float* omega, float* data, int cmax, int rmax,
		float* weightChange, float* psiOut) {
		int index = getIndex();
		if (index >= rmax * cmax) return;

		int r = index / cmax;
		int c = index % cmax;

		float rA = thetaA[r];
		float psi = (rA * (1 - rA)) * omega[r];
		weightChange[index] = -1.0 * LAMBDA * psi * data[c];
		psiOut[r] = psi;
		return;

	}//kCalcWeightChange1

	__global__ void kCalcWeightChange2(float* thetaA, float* omegaError, float* data, int cmax, int rmax, 
					float* weightChange, float* psiOut) {
		int index = getIndex();
		if (index >= rmax * cmax) return;

		int r = index / cmax;
		int c = index % cmax;

		float rA = thetaA[r];
		float psi = (rA * (1 - rA)) * omegaError[r];
		weightChange[index] = -1.0 * LAMBDA * psi * data[c];
		psiOut[r] = psi;
		return;

	}//kCalcWeightChange2

	void calcWeightChange1(float* thetaResultA, float* omegaError, float* features, int kmax, int jmax, float* weightChange, float* psiOut) {
		dim3 tpb = dim3(BLOCKSIZE);
		dim3 bpgij = dim3(((jmax * kmax) + BLOCKSIZE - 1) / BLOCKSIZE);

		kCalcWeightChange1<<<bpgij, tpb>>>(thetaResultA, omegaError, features, kmax, jmax, weightChange, psiOut);

	}//calcWeightChange1

	void calcWeightChange2(float* thetaResultA, float* omegaError, float* features, int jmax, int imax, float* weightChange, float* psiOut) {
		/*
		result: [0:imax)(52), error: [0:imax)(52), data: [0, jmax)(10201), weightChange (outvar) ixj matrix
		*/
		dim3 tpb = dim3(BLOCKSIZE);
		dim3 bpgij = dim3(((imax * jmax) + BLOCKSIZE - 1) / BLOCKSIZE);

		kCalcWeightChange2<<<bpgij, tpb>>>(thetaResultA, omegaError, features, jmax, imax, weightChange, psiOut);

	}//calcWeightChange

	//##################################
	// HOST MAIN FUNCTIONS
	//##################################

	void backPropagate(cublasHandle_t* handle) {
		/*
		Derivative of E wrt weights at (j,i) (up to 10201 and 52, respectively):
			-sum of sse error * invActivate(result)[i] * hiddenData[j]
		psi_i = error_i * invActivate(result[i])
		Theta_i = result[i]
		omega_i = error_i
		Psi_j	= Omega_j * invActivate(Theta_j)
		Theta_j = hiddenData[j]
		Omega_j = (sum(psi_i) for i)

		weightChange_ji = lambda * hiddenData[j] * psi_i
		weightChange_kj = lambda * hiddenData[k] * psi_i

		*/
		cublasHandle_t mHandle; bool handling = false;
		if (handle == NULL) {
			handling = true;
			handle = &mHandle;
			cublasCreate(handle);
		}//if
		float alpha = 1.0;

		float testOut[F2SIZE];

		//final layer weight delta calculation
		calcWeightChange2(dRA, dRE, dF2A, F2SIZE, RSIZE, dW2D, dPi);
		//apply the weight change
		cublasSaxpy(*handle, F2SIZE * RSIZE, &alpha, dW2D, 1, dW2, 1);

		//calculate Omega_j off the psi_i values
		matMul(handle, dW2, dPi, dOj, F2SIZE, RSIZE, 1);
		checkCUDAErrorFn("matMul failed\n", NULL, __LINE__);

		//DEBUG OUTPUT
		cudaMemcpy(testOut, dPi, RSIZE * sizeof(float), cudaMemcpyDeviceToHost);
		checkCUDAErrorFn("cudamemcpy failed\n", NULL, __LINE__);

		//next-to-last layer weight delta calculation
		calcWeightChange1(dF2A, dOj, dF0, F0SIZE, F2SIZE, dW1D, dPj);
		checkCUDAErrorFn("calcWeightChange failed\n", NULL, __LINE__);
		//apply the weight change
		cublasSaxpy(*handle, F0SIZE * F2SIZE, &alpha, dW1D, 1, dW1, 1);
		checkCUDAErrorFn("saxpy failed\n", NULL, __LINE__);

		cudaMemcpy(testOut, dW1, F2SIZE * sizeof(float), cudaMemcpyDeviceToHost);
		checkCUDAErrorFn("cudamemcpy failed\n", NULL, __LINE__);


		if (handling) {
			cublasDestroy(*handle);
		}//if
	}//backPropagate

	float_v forwardPropagate(InputData x, float* resultArray, cublasHandle_t* handle) {
		//Make our cublas handle if not handed one
		cublasHandle_t mHandle; bool handling = false;
		if (handle == NULL) {
			handling = true;
			handle = &mHandle;
			cublasCreate(handle);
		}//if
		float* dataPtr = x.fData.data();
		float testOut[F2SIZE] = {};

		cudaMemcpy(dF0, dataPtr, F0SIZE * sizeof(float), cudaMemcpyHostToDevice);
		checkCUDAErrorFn("cudaMemcpy failed\n", NULL, __LINE__);

		//Fully connected layer w/ W1
		matMul(handle, dF0, dW1, dF2, 1, F0SIZE, F2SIZE);
		checkCUDAErrorFn("matMul failed\n", NULL, __LINE__);

		//DEBUG OUTPUT
		cudaMemcpy(testOut, dW1, F2SIZE * sizeof(float), cudaMemcpyDeviceToHost);
		checkCUDAErrorFn("cudaMemcpy failed\n", NULL, __LINE__);
		
		//activate the first results
		activateResults(dF2, dF2A, F2SIZE);
		checkCUDAErrorFn("activateResults failed\n", NULL, __LINE__);

		//Fully connected layer w/ W2
		matMul(handle, dF2A, dW2, dR, 1, F2SIZE, RSIZE);
		checkCUDAErrorFn("matMul failed\n", NULL, __LINE__);

		//Activate results
		activateResults(dR, dRA, RSIZE);
		checkCUDAErrorFn("activateResults failed\n", NULL, __LINE__);

		cudaMemcpy(resultArray, dRA, RSIZE * sizeof(float), cudaMemcpyDeviceToHost);
		checkCUDAErrorFn("cudaMemcpy failed\n", NULL, __LINE__);

		if (handling) {
			cublasDestroy(*handle);
		}//if

		return calcErrorSingle(x, resultArray, dRE);
	}//forwardPropH

	void trainWeights(InputData_v records, int numIterations) {
		cublasHandle_t handle;
		cublasCreate(&handle);

		float results[RSIZE] = {};//floating space for the results to be put

		//initialize random weights between -1 and 1
		gpuFillRand(dW1, F1SIZE, F2SIZE, -1.0, 1.0);
		gpuFillRand(dW2, F2SIZE, RSIZE, -1.0, 1.0);

		for (int iter = 0; iter < numIterations; iter++) {
			float_vv errorValues = float_vv();

			//go forward
			float_v errorVal = forwardPropagate(records[iter % records.size()], results, &handle);

			/*
			errorValues.push_back(errorVal);
			float_v sseError = calcSumSquareErrors(errorValues);
			*/
			if (iter == 0) {
				printf("==========RESULTS=========\n");
				for (int i = 0; i < RSIZE; i++) {
					printf("@%02d:  %f\t", i, results[i]);
					if ((i + 1) % 4 == 0) {
						printf("\n");
					}
				}//for
			}//if
			float energy = calcEnergy(errorVal);
			printf("@%03d: Calculated energy is %.8f\n", iter, energy);

			//go backwards
			backPropagate(&handle);
		}//for

		cublasDestroy(handle);

	}//trainWeights


	void transpose(float* A, float* Aswap, int m, int n) {
		int numElements = m * n;
		dim3 tpb = dim3(BLOCKSIZE);
		dim3 bpg = dim3((numElements + BLOCKSIZE - 1) / BLOCKSIZE);

		float testArray[2][2];

		cudaMemcpy(testArray, A, 4 * sizeof(float), cudaMemcpyDeviceToHost);
		checkCUDAErrorFn("cudaMemcpy failed\n", NULL, __LINE__);

		kTranspose << <tpb, bpg >> > (A, Aswap, m, n);
		checkCUDAErrorFn("kTranspose failed\n", NULL, __LINE__);

		cudaMemcpy(testArray, Aswap, 4 * sizeof(float), cudaMemcpyDeviceToHost);
		checkCUDAErrorFn("cudaMemcpy failed\n", NULL, __LINE__);

		cudaMemcpy(A, Aswap, numElements * sizeof(float), cudaMemcpyDeviceToDevice);
		checkCUDAErrorFn("cudaMemcpy failed\n", NULL, __LINE__);

		cudaMemcpy(testArray, A, 4 * sizeof(float), cudaMemcpyDeviceToHost);
		checkCUDAErrorFn("cudaMemcpy failed\n", NULL, __LINE__);

		return;

	}//transpose


	//##################################
	// CONVOLVING
	//##################################

	//Convolutional layer:
	//1. Convolve (into an intermediary)
	//2. Activate the intermediary
	//3. Max pool down into some feature vector (to be fed into some of the FC layers)

	/**
	Pools some number of activated convolutions down into a smaller buffer
	Does so in blockWidth x blockWidth squares
	Wants to spawn a number of threads equal to the number of resultant output "pixels"
	*/
	__global__ void kmaxPool(float* idata, float* odata, int blockWidth, int idataWidth, int odataWidth) {
		int index = getIndex();
		if (index >= odataWidth * odataWidth) return;

		int oR	= index / odataWidth;
		int oC	= index % odataWidth;
		int iR	= oR * blockWidth - (blockWidth / 2);
		int iC	= oC * blockWidth - (blockWidth / 2);
		int iindex = iR * idataWidth + iC;
		float max = -1.0e40;//stand-in for a minimum
		for (int i = 0; i < blockWidth; i++) {
			int iOffset = idataWidth * (i - (blockWidth / 2));
			for (int j = 0; j < blockWidth; j++) {
				max = fmaxf(max, idata[iindex + iOffset + (j - (blockWidth / 2))]);
			}//for
		}//for
		odata[index] = max;
	}//kmaxPool
		
	/**
	* Does a convolution from one image to another
	* A few notes:
	* Takes char data in for the input
	* Assuming we're running one thread per output pixel, and that we've sized things correctly for our filter
	* filter, idata, and odata must all be square
	* Also, currently only accepting filter widths of 3
	*/
	__global__ void kconvolve(filter3 filter, float* idata, float* odata, int odataWidth) {
		int index = getIndex();
		if (index >= odataWidth * odataWidth) return;
		int idataW = odataWidth + 2;

		//get ourselves an "idata" index
		int iindex = (index / odataWidth) * 2 + 1 + idataW;

		float sum = 0;

		uint8_t relData[9];
		//Flips the kernel here
		relData[0] = idata[DR(iindex, idataW)];
		relData[1] = idata[DC(iindex, idataW)];
		relData[2] = idata[DL(iindex, idataW)];
		relData[3] = idata[CR(iindex, idataW)];
		relData[4] = idata[CC(iindex, idataW)];
		relData[5] = idata[CL(iindex, idataW)];
		relData[6] = idata[UR(iindex, idataW)];
		relData[7] = idata[UC(iindex, idataW)];
		relData[8] = idata[UL(iindex, idataW)];
		for (int i = 0; i < 9; i++) {
			sum += relData[i] * filter.kernel[i];
		}//for 9
#if ALLOWKERNEL5
		else if (filterWidth == 5) {
			uint8_t relData[25];
			//Flips the kernel here (without the macro stuff)
			for (int i = 0; i < 5; i++) {
				int iOffset = idataW * (i - 2);
				for (int j = 0; j < 5; j++) {
					relData[5 * i + j] = idata[iindex + (j - 2) + iOffset];
				}//for
			}//for
			for (int i = 0; i < 25; i++) {
				sum += relData[i] * filter[i];
			}//for 25
		}//elif 5
#endif

		odata[index] = sum;

	}//kconvolve

	void convolve(float* idata, float* odata, int odataSize, filter3 kernel) {
		dim3 tpb = dim3(BLOCKSIZE);
		dim3 bpg = dim3(((odataSize) + BLOCKSIZE - 1) / BLOCKSIZE);

		kconvolve<<<bpg, tpb>>>(kernel, idata, odata, (int)sqrt(odataSize));
		checkCUDAErrorFn("kconvolve failed\n", NULL, __LINE__);
	}//convolve

	/**
	Does the forward propagation for convolving stuff
	Also max-pools
	Returns the size of the output layer (sure why not)
	*/
	int convolveStep(float* inputLayer, int inputLayerSize, float* outputPoolingLayer, float* outputLayer, int poolWidth) {
		int inputLayerWidth = (int)sqrt(inputLayerSize);
		int outputPoolingBlockWidth = inputLayerWidth - 2;
		int outputPoolingBlockSize = outputPoolingBlockWidth * outputPoolingBlockWidth;
		int outputPooledBlockSize = outputPoolingBlockSize / (poolWidth * poolWidth);
		int outputPooledBlockWidth = (int)sqrt(outputPooledBlockSize);
		int outputLayerSize = NUMFILTERS * outputPooledBlockSize;

		//convolve
		for (int i = 0; i < NUMFILTERS; i++) {
			convolve(inputLayer, outputPoolingLayer + (i * outputPoolingBlockSize), outputPoolingBlockSize, allKernels[i]);
		}//for

		cudaDeviceSynchronize();

		//pool
		dim3 tpb = dim3(BLOCKSIZE);
		dim3 bpg = dim3(((outputPooledBlockSize)+BLOCKSIZE - 1) / BLOCKSIZE);
		for (int i = 0; i < NUMFILTERS; i++) {
			//	__global__ void kmaxPool(float* idata, float* odata, int blockWidth, int idataWidth, int odataWidth) {
			int iBlockOffset = i * outputPoolingBlockSize;
			int oBlockOffset = i * outputPooledBlockSize;
			kmaxPool<<<bpg, tpb>>>(outputPoolingLayer + iBlockOffset, outputLayer + oBlockOffset, poolWidth, outputPoolingBlockWidth, outputPooledBlockWidth);
			checkCUDAErrorFn("kmaxpool failed\n", NULL, __LINE__);
		}//for

		return outputLayerSize;
	}//convolveStep

}//CharacterRecognition
