#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>
#include "common.h"
#include "mlp.h"


#define INPUT_LAYER_SIZE 4
#define HIDDEN_LAYER_SIZE 3
#define OUTPUT_LAYER_SIZE 2
float weights_IH[INPUT_LAYER_SIZE * HIDDEN_LAYER_SIZE];
float weights_HO[HIDDEN_LAYER_SIZE * OUTPUT_LAYER_SIZE];

float* hidden;
float* hidden_sigmoid;
float* output;
float* output_softmax;

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


	int forwardPass(float* idata, float** weights) {
		// Matrix Multiply Input Layer and Weights 1
		// Add Bias
		// Apply ReLU
		// Matrix Multiply Hidden layer and Weights 2
		// Add Bias
		// Apply Softmax
		// Max probability
		return -1;
	}

	void backwardPropagation() {

	}

	float calculateLoss(int* label, int* prediction) {
		return -1;
	}

	void train(int* idata, int* ilabel) {
		// Create Device Buffers for Input and Output
		// Create Weight and Bias Buffers
		// Forward Pass
		// Loss calculation
		// Back Propagation
	}

	void test(int* idata, int* ilabel, int* olabel) {

	}

	void printArray(int n, float *a, bool abridged = false) {
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

	void printCudaArray(int size, float* data) {
		float *d_data = new float[size];
		cudaMemcpy(d_data, data, size * sizeof(float), cudaMemcpyDeviceToHost);
		printArray(size, d_data, true);
	}

	void CharacterRecognition::init() {
		printf("Here");
		// Create a handle for CUBLAS
		cublasHandle_t handle;
		cublasCreate(&handle);
		//curandGenerator_t prng;
		//curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
		//// Set the seed for the random number generator using the system clock
		//curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());
	 //   // Fill the array with random numbers on the device
		//curandGenerateUniform(prng, weights_IH, INPUT_LAYER_SIZE * HIDDEN_LAYER_SIZE);
		//curandGenerateUniform(prng, weights_HO, HIDDEN_LAYER_SIZE * OUTPUT_LAYER_SIZE);

		//printCudaArray(INPUT_LAYER_SIZE * HIDDEN_LAYER_SIZE, weights_IH);
		//printCudaArray(HIDDEN_LAYER_SIZE * OUTPUT_LAYER_SIZE, weights_HO);
	}
}
