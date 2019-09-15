#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "mlp.h"

#define blockSize 128


namespace CharacterRecognition {
    using Common::PerformanceTimer;
    PerformanceTimer& timer()
    {
        static PerformanceTimer timer;
        return timer;
    }
      
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

	// TODO: implement required elements for MLP sections 1 and 2 here
	
	double * iLayer;
	double * hLayer;
	double * oLayer;

	double * w_kj;
	double * w_ji;

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

	// kernel to to matmul
	// A mxn
	// B nxk
	// C mxk
	__global__ matrixMultiplyKernel(const float *dev_A, const float *dev_B, const float *dev_C, int m, int n, int k) {
		
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
	
		int sum = 0;
		if (col < k && row < m)
		{
			for (int i = 0; i < n; i++)
				sum += dev_A[row * n + i] * dev_B[i * k + col];
			
			dev_C[row * k + col] = sum;
		}
	}
	}


	void trainMLP(int n_input, int n_classes, int n_hidden, double *odata, const double *idata) {
		timer().startGpuTimer();

		// todo
		//FORWARD PASS
		// initlaise Wight layers randomly
		// initialise input layer from i data
		// Compute h1 = W1*x
		// Compute H1 = Sig(h1)
		// Compute h2 = W2*H1
		// Compute H2 = Sig(h2)
		// Compute y=Softmax(H2)
			   
		// Sample MatrixMultiplication 
		int n;
		int m;
		int k;
		dim3 dimGrid((k + blockSize - 1) / blockSize, (m + blockSize - 1) / blockSize);
		dim3 dimBlock(blockSize, blockSize);


		timer().endGpuTimer();
	}

	// MLP section 2 Character Reader
	//void initCharMLP(int N, int P, int iDim, int hDim, int oDim);
	//void readData(int N, int P, int iDim, int hDim, int oDim);
	//void trainCharMLP(int N, int P, int iDim, int hDim, int oDim);
	//void testCharMLP(int N, int P, int iDim, int hDim, int oDim);

}
