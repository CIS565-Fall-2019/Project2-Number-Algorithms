#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "mlp.h"
#include <thrust/random.h>

#define blockSize 512
namespace CharacterRecognition {
    using Common::PerformanceTimer;
    PerformanceTimer& timer()
    {
        static PerformanceTimer timer;
        return timer;
    }
	__global__ void kernInitWeightsBias(float *W, float *b, int inputDim, int num_samples, int outputDim) {
		int index = (blockIdx.x * blockDim.x) + threadIdx.x;
		if (index >= inputDim * outputDim) {
			return;
		}
		thrust::default_random_engine rng;
		thrust::uniform_real_distribution<float> dist(0.0, 1.0);
		W[index] = dist(rng);
		int y = index / outputDim;
		b[y] = 0;
	}

	__global__  void kernAffineForward(float *W, float *b, float *in, float *out, int inputDim, int outputDim, int num_samples, bool sigmoid) {
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		float val = 0;

		if (row < num_samples && col < inputDim) {
			for (int i = 0; i < inputDim; i++) {
				val += W[row * inputDim + i] * in[i * inputDim + col];
			}
			val += b[row];
		}
		out[row * outputDim + col] = sigmoid ? val : 1/(1+__expf(-val));
	}

	__global__  void kernAffineBackward() {
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		float val = 0;

		if (row < num_samples && col < inputDim) {
			for (int i = 0; i < inputDim; i++) {
				val += W[row * inputDim + i] * in[i * inputDim + col];
			}
			val += b[row];
		}
		out[row * outputDim + col] = sigmoid ? val : 1/(1+__expf(-val));
	}

	//AffineLayer 
	AffineLayer::AffineLayer(int idim, int odim) : inputDim(idim), outputDim(odim), softmax(true), eval(false) {
		//Malloc Weights & Biases
		cudaMalloc(&W, idim * odim * sizeof(float));
		checkCUDAError("cuda Malloc W failed");
		cudaMalloc(&b, odim * sizeof(float));
		checkCUDAError("cuda Malloc b failed");

		//Call Initializer Kernels
		dim3 fullBlocksPerGrid((inputDim * outputDim - 1) / blockSize);
		kernInitWeightsBias<<<fullBlocksPerGrid, blockSize>>>(W, b, inputDim, outputDim);
	}

	void AffineLayer::forward(float *in, float *out, int num_samples) {
		/*Uses W & b to perform forward pass on an Affine Layer (Assumes dimensions are correct or things will go very wrong)
		in: Input array of shape inputDim * num_samples
		out: Output array of shape outputDim * num_samples (to be filled in)
		*/
		//Malloc the input matrix and an output matrix (should I even do this? Memcpy?)
		cudaMalloc(&in, inputDim * num_samples * sizeof(float));
		checkCUDAError("cuda Malloc in failed");
		cudaMalloc(&out, outputDim * num_samples * sizeof(float));
		checkCUDAError("cuda Malloc in failed");

		//Call Affine Forward Kernel 
		dim3 affine_blocksize(8, 8);
		dim3 numBlocks((outputDim + affine_blocksize.x - 1) / affine_blocksize.x, (num_samples + affine_blocksize.y - 1) / affine_blocksize.y);
		kernAffineForward<<<numBlocks, affine_blocksize>>>(W, b, in, out, inputDim, outputDim, num_samples, sigmoid);


		//delete 
		cudaFree(&out);
		cudaFree(&in);
	}
}
