#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include "common.h"
#include "mlp.h"

#define ALLOWKERNEL5 1

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
	// FUNCTION DELCARATIONS
	//##################################

	/**
	Gets the "index" for the thread
	Currently, only supporting single-dimensional block indexes
	Computes all relevant x, y, z transformations
	*/
	__device__ int getIndex();

        
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
	We're just going to hard-code the dimensions for now...
	*/
	__global__ void forwardFCHiddenToResults(float* hiddenStack, float* weights, float* results);

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

	void transpose(float* A, float* Aswap, int m, int n) {
		int numElements = m * n;
		dim3 tpb = dim3(BLOCKSIZE);
		dim3 bpg = dim3((numElements + BLOCKSIZE - 1) / BLOCKSIZE);

		kTranspose<<<tpb, bpg>>>(A, Aswap, m, n);
		checkCUDAErrorFn("kTransposeF failed\n", NULL, __LINE__);

		cudaMemcpy(A, Aswap, numElements * sizeof(float), cudaMemcpyDeviceToDevice);
		checkCUDAErrorFn("cudaMemcpy failed\n", NULL, __LINE__);

		return;

	}//transpose

	void matMul(cublasHandle_t* handle, const float* A, const float* B, float* C, int m, int k, int n, float* Cswap) {
		bool allocating = false;
		if (Cswap == NULL) {
			allocating = true;
			cudaMalloc((void**)& Cswap, m * n * sizeof(float));
			checkCUDAErrorFn("cudaMalloc kern_idata failed!\n", NULL, __LINE__);
		}//if no swap space given, make some

		//Since cublas expects column-major indexing, our A is effectively AT (kxm), and our B is effectively BT (nxk)
		//As such, we're going to be doing BT * AT = CT (nxm)
		//Then, we transpose C "in place" before we return
		float alpha = 1.0;
		float beta = 0.0;

		//Future development: put the result into Cswap, transpose into C
		cublasSgemm(*handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B, n, A, k, &beta, C, n) ;

		transpose(C, Cswap, n, m);

		if (allocating) {
			cudaFree(Cswap);
		}//if
	}//matMul

	//##################################
	// HOST MAIN FUNCTIONS
	//##################################


	void testMatrixMultiply() {
		cublasHandle_t handle;
		cublasCreate(&handle);
		const float A[2][3] = { {0, 1, 2},  {2, 0, 3} };
		const float B[3][2] = { {1, 0}, {0, 2}, {-1, 1} };
		float C[2][2];

		float* dA;
		float* dB;
		float* dC;

		cudaMalloc((void**)& dA, 6 * sizeof(float));
		checkCUDAErrorFn("cudaMalloc failed!\n", NULL, __LINE__);
		cudaMalloc((void**)& dB, 6 * sizeof(float));
		checkCUDAErrorFn("cudaMalloc failed!\n", NULL, __LINE__);
		cudaMalloc((void**)& dC, 4 * sizeof(float));
		checkCUDAErrorFn("cudaMalloc failed!\n", NULL, __LINE__);

		cudaMemcpy(dA, A, 6 * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dB, B, 6 * sizeof(float), cudaMemcpyHostToDevice);

		matMul(&handle, dA, dB, dC, 2, 3, 2);

		cudaMemcpy(C, dC, 4 * sizeof(float), cudaMemcpyDeviceToHost);

		printf("[");
		for (int i = 0; i < 2; i++) {
			printf("[");
			for (int j = 0; j < 2; j++) {
				printf("%f, ", C[i][j]);
			}//for
			printf("]\n");
		}//for
		printf("]\n");


		cudaFree(dA);
		checkCUDAErrorFn("cudaFree failed\n", NULL, __LINE__);
		cudaFree(dB);
		checkCUDAErrorFn("cudaFree failed\n", NULL, __LINE__);
		cudaFree(dC);
		checkCUDAErrorFn("cudaFree failed\n", NULL, __LINE__);

		cublasDestroy(handle);
	}//testMatrixMultiply

	void forwardPropagate(InputData x, float* resultArray) {
		cublasHandle_t handle;
		cublasCreate(&handle);




		cublasDestroy(handle);
	}//forwardProp
}
