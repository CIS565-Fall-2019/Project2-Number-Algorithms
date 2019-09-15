#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "mlp.h"
#include "cublas_v2.h"

# define blockSize 128
# define hiddenLayerLen 10

static __inline__ void modify(cublasHandle_t handle, float *m, int ldm, int n, int p, int q, float alpha, float beta) {
	cublasSscal(handle, n - q + 1, &alpha, &m[IDX2F(p, q, ldm)], ldm);
	cublasSscal(handle, ldm - p + 1, &beta, &m[IDX2F(p, q, ldm)], 1);
}

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

	__global__ void kernMatrixMultiplication(int n, int m, int k, float *M,float *N, float *Out)  {
		int ty = blockIdx.y * blockDim.y + threadIdx.y;
		int tx = blockIdx.x * blockDim.x + threadIdx.x;
		int sum = 0;
		if (col < k && )

	}
	__global__ void kernActivationFunction(int N, float* ) {

	}
	// TODO: implement required elements for MLP sections 1 and 2 here

	/*void matrixMultiplication(float *M,float *N,float *Out) {

		stat = cublasCreate(&handle);
		if (stat != CUBLAS_STATUS_SUCCESS) {
			printf("CUBLAS initialization failed\n");
			return EXIT_FAILURE;
		}
		stat = cublasSetMatrix(M, N, sizeof(*a), a, M, devPtrA, M);
		if (stat != CUBLAS_STATUS_SUCCESS) {
			printf("data download failed");
			cudaFree(devPtrA);
			cublasDestroy(handle);
			return EXIT_FAILURE;
		}
		modify(handle, devPtrA, M, N, 2, 3, 16.0f, 12.0f);
		stat = cublasGetMatrix(M, N, sizeof(*a), devPtrA, M, a, M);
		if (stat != CUBLAS_STATUS_SUCCESS) {
			printf("data upload failed");
			cudaFree(devPtrA);
			cublasDestroy(handle);
			return EXIT_FAILURE;
		}
	}*/

	void createNN(int n,int h,int m, const float *idata, float *hidden, float *odata, const float *weightsIH, const float *weightsHO) {		
		
		cublasStatus_t stat;
		cublasHandle_t handle;

		cudaMalloc((void**)&dev_input, n * sizeof(float));
		checkCUDAErrorFn("Malloc idata into input failed");

		cudaMemcpy(dev_input, idata, sizeof(float) * n, cudaMemcpyHostToDevice);
		checkCUDAErrorFn("Copying idata to input failed");

		cudaMalloc((void**)&dev_hiddenLayer, h * sizeof(float));
		checkCUDAErrorFn("Malloc idata into hidden layer failed");

		cudaMalloc((void**)&dev_output, m * sizeof(float));
		checkCUDAErrorFn("Malloc idata into output failed");

		cudaMalloc((void**)&dev_weightsIH, (n*h) * sizeof(float));
		checkCUDAErrorFn("Malloc idata into weights b/w input & hidden failed");

		cudaMalloc((void**)&dev_weightsHO, (h*m) * sizeof(float));
		checkCUDAErrorFn("Malloc idata into weights b/w hidden & output failed");

		cudaMemcpy(dev_weightsIH, weightsIH, sizeof(float) * (n*h), cudaMemcpyHostToDevice);
		checkCUDAErrorFn("Copying weights array 1 failed");

		cudaMemcpy(dev_weightsHO, weightsHO , sizeof(float) * (h*m), cudaMemcpyHostToDevice);
		checkCUDAErrorFn("Copying weights array 2 failed");

		dim3 fullBlocks1((n + blockSize - 1) / blockSize);
		dim3 fullBlocks2((h + blockSize - 1) / blockSize);
		dim3 fullBlocks3((m + blockSize - 1) / blockSize);
		dim3 fullBlocksMult((m + blockSize - 1) / blockSize);

		//kernMultiplyWeights << <fullBlocks2, blockSize >> > (n,hiddenLayerLen,dev_input,dev_hiddenLayer,dev_weightsIH);

		kernMatrixMultiplication(dev_input, dev_weightsIH, dev_hiddenLayer);

		//cudaMemcpy(dev_hiddenLayer, hidden, sizeof(float) * (n*h), cudaMemcpyHostToDevice);
		//checkCUDAErrorFn("Copying hidden layer units failed");

		kernActivationFunction<< <fullBlocks2, blockSize >> > (h,dev_hiddenLayer);
		checkCUDAErrorFn("Kernel Activation function failed");

	
	}
}
