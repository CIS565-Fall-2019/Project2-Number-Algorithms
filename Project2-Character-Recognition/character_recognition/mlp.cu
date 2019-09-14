#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include "common.h"
#include "mlp.h"

#define ALLOWKERNEL5 1
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

#define F0SIZE 10201
#define F1SIZE 10201
#define F2SIZE 10201
#define RSIZE 52
#define W1SIZE (F1SIZE * F2SIZE)
#define W2SIZE (F2SIZE * RSIZE)


	//##################################
	// DEVICE POINTER MEMORY
	//##################################

	float* dF0;//features 0 (orig data)
	float* dW0;//weights 0
	float* dF1;//features 1
	float* dW1;//weights 1
	float* dW1D;//delta value for weights 1
	float* dF2;//features 2
	float* dW2;//weights 2
	float* dW2D;//delta value for weights 2
	float* dR;//result
	float* dRA;//result(activated)
	float* dRIA;//result, inverse activated
	float* dRE;//result error

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
		cudaMalloc((void**)& dW1, W1SIZE * sizeof(float));
		checkCUDAErrorFn("cudaMalloc failed\n", NULL, __LINE__);
		cudaMalloc((void**)& dW1D, W1SIZE * sizeof(float));
		checkCUDAErrorFn("cudaMalloc failed\n", NULL, __LINE__);
		cudaMalloc((void**)& dW2, W2SIZE *sizeof(float));
		checkCUDAErrorFn("cudaMalloc failed\n", NULL, __LINE__);
		cudaMalloc((void**)& dW2D, W2SIZE * sizeof(float));
		checkCUDAErrorFn("cudaMalloc failed\n", NULL, __LINE__);
		cudaMalloc((void**)& dR, RSIZE * sizeof(float));
		checkCUDAErrorFn("cudaMalloc failed\n", NULL, __LINE__);
		cudaMalloc((void**)& dRE, RSIZE * sizeof(float));
		checkCUDAErrorFn("cudaMalloc failed\n", NULL, __LINE__);
		cudaMalloc((void**)& dRA, RSIZE * sizeof(float));
		checkCUDAErrorFn("cudaMalloc failed\n", NULL, __LINE__);
		cudaMalloc((void**)& dRIA, RSIZE * sizeof(float));
		checkCUDAErrorFn("cudaMalloc failed\n", NULL, __LINE__);

	}//kmallocBuffers

	void kfreeBuffers() {
		cudaFree(dF0);
		checkCUDAErrorFn("cudaFree failed\n", NULL, __LINE__);
		cudaFree(dW1);
		checkCUDAErrorFn("cudaFree failed\n", NULL, __LINE__);
		cudaFree(dW1D);
		checkCUDAErrorFn("cudaFree failed\n", NULL, __LINE__);
		cudaFree(dW2);
		checkCUDAErrorFn("cudaFree failed\n", NULL, __LINE__);
		cudaFree(dW2D);
		checkCUDAErrorFn("cudaFree failed\n", NULL, __LINE__);
		cudaFree(dR);
		checkCUDAErrorFn("cudaFree failed\n", NULL, __LINE__);
		cudaFree(dRE);
		checkCUDAErrorFn("cudaFree failed\n", NULL, __LINE__);
		cudaFree(dRA);
		checkCUDAErrorFn("cudaFree failed\n", NULL, __LINE__);
		cudaFree(dRIA);
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

	/**
	* Does a convolution from one image to another
	* A few notes:
	* Takes char data in for the input
	* Assuming we're running one thread per output pixel, and that we've sized things correctly for our filter
	* filter, idata, and odata must all be square
	* Also, currently only accepting filter widths of 3 and 5
	*/
	__global__ void convolve(float* filter, int filterWidth, uint8_t* idata, float* odata, int odataWidth) {
		int index = getIndex();
		if (index >= odataWidth * odataWidth) return;
		int idataW = odataWidth + 2;

		//get ourselves an "idata" index
		int iindex = (index / odataWidth) * 2 + 1 + idataW;
		

		float sum = 0;

		if (filterWidth == 3) {
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
				sum += relData[i] * filter[i];
			}//for 9
		}//if 3
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
		else {
			return;//please don't get here
		}//else

		odata[index] = sum;
 
	}//convolve


	//##################################
	// HOST HELPER FUNCTIONS
	//##################################

	void activateResults(int numResults) {
		dim3 tpb = dim3(BLOCKSIZE);
		dim3 bpg = dim3((numResults + BLOCKSIZE - 1) / BLOCKSIZE);

		checkCUDAErrorFn("kActivateResults failed\n", NULL, __LINE__);
		kActivateResults<<<bpg, tpb>>>(dR, dRA, numResults);
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

	void transpose(float* A, float* Aswap, int m, int n) {
		int numElements = m * n;
		dim3 tpb = dim3(BLOCKSIZE);
		dim3 bpg = dim3((numElements + BLOCKSIZE - 1) / BLOCKSIZE);

		float testArray[2][2];

		cudaMemcpy(testArray, A, 4 * sizeof(float), cudaMemcpyDeviceToHost);
		checkCUDAErrorFn("cudaMemcpy failed\n", NULL, __LINE__);

		kTranspose<<<tpb, bpg>>>(A, Aswap, m, n);
		checkCUDAErrorFn("kTranspose failed\n", NULL, __LINE__);

		cudaMemcpy(testArray, Aswap, 4 * sizeof(float), cudaMemcpyDeviceToHost);
		checkCUDAErrorFn("cudaMemcpy failed\n", NULL, __LINE__);

		cudaMemcpy(A, Aswap, numElements * sizeof(float), cudaMemcpyDeviceToDevice);
		checkCUDAErrorFn("cudaMemcpy failed\n", NULL, __LINE__);

		cudaMemcpy(testArray, A, 4 * sizeof(float), cudaMemcpyDeviceToHost);
		checkCUDAErrorFn("cudaMemcpy failed\n", NULL, __LINE__);

		return;

	}//transpose

	void matMul(cublasHandle_t* handle, const float* A, const float* B, float* C, int m, int k, int n, float* Cswap) {
		/*
		bool allocating = false;
		if (Cswap == NULL) {
			float newCSwap;
			Cswap = &newCSwap;
			allocating = true;
			cudaMalloc((void**)& Cswap, m * n * sizeof(float));
			checkCUDAErrorFn("cudaMalloc failed!", NULL, __LINE__);
		}//if no swap space given, make some*/

		//Since cublas expects column-major indexing, our A is effectively AT (kxm), and our B is effectively BT (nxk)
		//As such, we're going to be doing BT * AT = CT (nxm)
		//Then, we transpose C "in place" before we return
		float alpha = 1.0;
		float beta = 0.0;

		//Future development: put the result into Cswap, transpose into C
		cublasSgemm(*handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B, n, A, k, &beta, C, n) ;
		checkCUDAErrorFn("the internal matrix multiply failed\n", NULL, __LINE__);
		//cudaDeviceSynchronize();

		//no need to transpose?? not sure why, but this function operates
		//transpose(C, Cswap, n, m);

		/*
		if (allocating) {
			cudaFree(Cswap);
			checkCUDAErrorFn("cudaFree failed\n", NULL, __LINE__);
		}//if
		*/
	}//matMul


	__global__ void kCalcWeightChange(float* resultIA, float* error, float* data, int jmax, int imax, float* weightChange) {
		int index = getIndex();
		if (index > imax * jmax) return;

		int i = index / jmax;
		int j = index % jmax;

		weightChange[index] = -1.0 * LAMBDA * resultIA[i] * data[j] * error[i];
		return;

	}//kCalcWeightChange

	void calcWeightChange(float* result, float* resultIA, float* error, float* data, int jmax, int imax, float* weightChange) {
		/*
		result: [0:imax)(52), error: [0:imax)(52), data: [0, jmax)(10201), weightChange (outvar) ixj matrix
		*/
		dim3 tpb = dim3(BLOCKSIZE);
		dim3 bpgi = dim3((imax + BLOCKSIZE - 1) / BLOCKSIZE);
		dim3 bpgj = dim3((jmax + BLOCKSIZE - 1) / BLOCKSIZE);
		dim3 bpgij = dim3(((imax * jmax) + BLOCKSIZE - 1) / BLOCKSIZE);


		//kActivateInverse(dR, dRIA, imax);
		kActivateInverse<<<bpgi, tpb>>>(result, resultIA, imax);
		checkCUDAErrorFn("kAI failed\n", NULL, __LINE__);

		kCalcWeightChange<<<bpgij, tpb>>>(resultIA, error, data, jmax, imax, weightChange);

		//for (int i = 0; i < imax; i++) {
		//	for (int j = 0; j < jmax; j++) {
		//		weightChange[i * jmax + j] = LAMBDA * data[j] * error[i] * invActivate(result)[i];
		//	}//for
		//}//for


	}//calcWeightChange

	//##################################
	// HOST MAIN FUNCTIONS
	//##################################

	void backPropError(cublasHandle_t* handle) {
		/*
		Derivative of E wrt weights at (j,i) (up to 10201 and 52, respectively):
			-sum of sse error * invActivate(result)[i] * hiddenData[j]
		psi_i = error_i * invActivate(result[i])
		Theta_i = result[i]
		omega_i = error_i
		Psi_j	= 
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
		}

		//final layer weight delta calculation
		calcWeightChange(dR, dRIA, dRE, dF0, 10201, 52, dW2D);
		//apply the weight change
		float alpha = 1.0;
		cublasSaxpy(*handle, 10201 * 52, &alpha, dW2D, 1, dW2, 1);

		if (handling) {
			cublasDestroy(*handle);
		}//if
	}//backPropError

	float_v forwardPropagateH(InputData x, float* resultArray, cublasHandle_t* handle) {
		//Make our cublas handle if not handed one
		cublasHandle_t mHandle; bool handling = false;
		if (handle == NULL) {
			handling = true;
			handle = &mHandle;
			cublasCreate(handle);
		}//if
		float* dataPtr = x.fData.data();

		cudaMemcpy(dF0, dataPtr, x.fData.size() * sizeof(float), cudaMemcpyHostToDevice);
		checkCUDAErrorFn("cudaMemcpy failed\n", NULL, __LINE__);

		//Fully connected layer
		matMul(handle, dF0, dW2, dR, 1, x.numElements, x.resultArray.size());
		checkCUDAErrorFn("matMul failed\n", NULL, __LINE__);

		//Activate results
		activateResults(x.resultArray.size());
		checkCUDAErrorFn("activateResults failed\n", NULL, __LINE__);

		cudaMemcpy(resultArray, dRA, x.resultArray.size() * sizeof(float), cudaMemcpyDeviceToHost);
		checkCUDAErrorFn("cudaMemcpy failed\n", NULL, __LINE__);

		if (handling) {
			cublasDestroy(*handle);
		}//if

		return calcErrorSingle(x, resultArray, dRE);
	}//forwardPropH

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

	float calcEnergy(float_v sse) {
		float sum = 0;
		for (int i = 0; i < sse.size(); i++) {
			sum += sse[i];
		}//for
		return sum / sse.size();//averaging the energy function?
	}//calcEnergy

	void trainWeights(InputData_v records, int numIterations) {
		cublasHandle_t handle;
		cublasCreate(&handle);
		//assuming we're just running all these bitches
		kmallocBuffers();

		float results[52] = {};//floating space for the results to be put

		//initialize random weights between -1 and 1
		gpuFillRand(dW2, 10201, 52, -1, 1);

		for (int iter = 0; iter < numIterations; iter++) {
			float_vv errorValues = float_vv();

			//go forward
			float_v errorVal = forwardPropagateH(records[iter % records.size()], results);

			errorValues.push_back(errorVal);
			float_v sseError = calcSumSquareErrors(errorValues);
			if (iter == 0) {
				printf("==========RESULTS=========\n");
				for (int i = 0; i < 52; i++) {
					printf("@%02d:  %f\t", i, results[i]);
					if ((i + 1) % 4 == 0) {
						printf("\n");
					}
				}//for
			}//if
			printf("@%03d: Calculated energy is %.8f\n", iter, calcEnergy(sseError));

			//go backwards
			backPropError(&handle);
		}//for
		printf("==========RESULTS=========\n");
		for (int i = 0; i < 52; i++) {
			printf("@%02d:  %f\t", i, results[i]);
			if ((i + 1) % 4 == 0) {
				printf("\n");
			}
		}//for
		cublasDestroy(handle);
		kfreeBuffers();

	}//trainWeights

}//CharacterRecognition
