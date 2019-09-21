#include <cuda.h>
#include <cuda_runtime.h>Shape 
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
	__host__ __device__ unsigned int hash(unsigned int a) {
	  a = (a + 0x7ed55d16) + (a << 12);
	  a = (a ^ 0xc761c23c) ^ (a >> 19);
	  a = (a + 0x165667b1) + (a << 5);
	  a = (a + 0xd3a2646c) ^ (a << 9);
	  a = (a + 0xfd7046c5) + (a << 3);
	  a = (a ^ 0xb55a4f09) ^ (a >> 16);
	  return a;
	}
	__global__ void kernInitWeightsBias(float *W, float *b, int inputDim, int outputDim){
		//Random Weight Initialization & Zero Bias Initialization
		int index = (blockIdx.x * blockDim.x) + threadIdx.x;
		if (index >= inputDim * outputDim) {
			return;
		}
		thrust::default_random_engine rng(hash((int)(index * inputDim * outputDim)));
		thrust::uniform_real_distribution<float> dist(0.0, 1.0);
		//W[index] = dist(rng);
		W[index] = 0.1 * index;
		int y = index / outputDim;
		b[y] = 0;
	}

	__global__ void kernAffineForward(float *W, float *b, float *in, float *out, int inputDim, int outputDim, int numSamples, bool sigmoid) {
		/*
		W: Shape inputDim x outputDim
		b: Shape outputDim
		in: Shape numSamples x inputDim
		out: Shape numSamples x outputDim
		*/
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		int row = index / outputDim;
		int col = index % outputDim;
		float val = 0;
		if (row < numSamples && col < outputDim) {
			for (int i = 0; i < inputDim; i++) {
				val += in[row * inputDim + i] * W[i * outputDim + col];
			}
			val += b[row];
		}
		out[row * outputDim + col] = sigmoid ? 1/(1+__expf(-val)) : val;
	}

	__device__ float applySigmoid(float x) {
		return 1 / (1 + __expf(-x));
	}

	__device__ float dSigmoid(float x) {
		return x * (1 - x);
	}

	__global__ void kern_dSigmoid(float *dout, float *doutLinear, int numSamples, int outputDim) {
		//Apply softmax across entire dout matrix (dout is outputDim x 
		int index = (blockIdx.x * blockDim.x) + threadIdx.x;
		if (index >= numSamples * outputDim) {
			return;
		}
		float doutidx = dout[index];
		doutLinear[index] = doutidx * (1 - doutidx);
	}

	__global__ void kern_dIn(float *doutLinear, float *W, float *din, int inputDim, int outputDim, int numSamples) {
		/* Effectively calculates matmul(doutLinear, W.T)
		doutLinear: outputDim x numSamples - each element is dL/dY where Y = XW + b
		W: inputDim x outputDim
		din: inputDim x numSamples - each element is dL/din_(i,j)
		*/
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		int row = index / inputDim;
		int col = index % inputDim;
		float val = 0;
		if (row < numSamples && col < inputDim) {
			for (int i = 0; i < outputDim; i++) {
				val += doutLinear[row * outputDim + i] * W[col * outputDim + i];
			}
		}
		din[row * inputDim + col] = val;
	}

	__global__ void kern_dW(float *W, float *b, float *doutLinear, float *in, int inputDim, int outputDim, int numSamples, float lr) {
		/* Effectively calculates matmul(input.T, doutLinear) and applies an update
		W: inputDim x outputDim (We do gradient descent here)
		b: outputDim (we do gradient decent here too)
		doutLinear: outputDim x numSamples - each element is dL/dY where Y = XW + b
		in: inputDim x numSamples
		lr: learning rate
		*/
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		int row = index / outputDim;
		int col = index % outputDim;
		float val = 0;
		float dbval = 0;
		float currW = W[row * outputDim + col];
		float currb = b[col];
		float doutLinearIdx = 0;
		if (row < inputDim && col < outputDim) {
			for (int i = 0; i < numSamples; i++) {
				doutLinearIdx = doutLinear[i * outputDim + col];
				val += in[row * inputDim + i] * doutLinearIdx;
				dbval += doutLinearIdx;
			}
		}
		W[row * outputDim + col] = currW - lr * (val);
		b[col] = currb - lr * (dbval);
	}

	//AffineLayer 
	AffineLayer::AffineLayer(int idim, int odim) : numSamples(0), inputDim(idim), outputDim(odim), sigmoid(true), eval(false), doneFwd(false){
		//Malloc Weights, Biases 
		cudaMalloc(&W, idim * odim * sizeof(float));
		checkCUDAError("cuda Malloc W failed");
		cudaMalloc(&b, odim * sizeof(float));
		checkCUDAError("cuda Malloc b failed");

		//Call Initializer Kernels
		dim3 fullBlocksPerGrid((inputDim * outputDim + blockSize - 1) / blockSize);
		kernInitWeightsBias<<<fullBlocksPerGrid, blockSize>>>(W, b, inputDim, outputDim);
	}
	void AffineLayer::setSigmoid(bool state) {
		sigmoid = state;
	}
	void AffineLayer::setEval(bool state) {
		eval = state;
	}

	float* AffineLayer::forward(float *in, int ns) {
		/*Uses W & b to perform forward pass on an Affine Layer 
		Assumes dev_input is set (on GPU), numSamples is set and eval is set
		*/
		//Malloc the input matrix and an output matrix 
		numSamples = ns;
		cudaMalloc((void**)&dev_in, inputDim * numSamples * sizeof(float));
		checkCUDAError("cuda Malloc dev_in in failed");
		cudaMalloc((void**)&dev_out, outputDim * numSamples * sizeof(float));
		checkCUDAError("cuda Malloc dev_out in failed");

		//Memcpy the *in information into dev_in
		cudaMemcpy(dev_in, in, inputDim * numSamples * sizeof(float), cudaMemcpyHostToDevice);

		//Call Affine Forward Kernel 
		int numBlocks = (numSamples * outputDim + blockSize - 1) / blockSize;
		kernAffineForward<<<numBlocks, blockSize>>>(W, b, dev_in, dev_out, inputDim, outputDim, numSamples, sigmoid);

		//Memcpy out the *out and *in information from dev_out
		float *out = new float[outputDim * numSamples];
		cudaMemcpy(out, dev_out, outputDim * numSamples * sizeof(float), cudaMemcpyDeviceToHost);

		//free (dont free dev_in because you'll need it for backprop)
		cudaFree(&dev_out);
		return out;
	}

	float* AffineLayer::backward(float *dout, float lr){
		/* Does backprop and one gradient update for W & b & returns din
		dout: upstream gradient coming in 
		lr: learning rate
		Returns 
		*/
		//Malloc the input matrix and an output matrix 
		cudaMalloc((void**)&dev_dout, outputDim * numSamples * sizeof(float));
		checkCUDAError("cuda Malloc dev_dout in failed");
		cudaMalloc((void**)&dev_din, inputDim * numSamples * sizeof(float));
		checkCUDAError("cuda Malloc dev_din in failed");
		cudaMalloc((void**)&dev_doutLinear, outputDim * numSamples * sizeof(float));
		checkCUDAError("cuda Malloc dev_din in failed");

		//Memcpy the *dout information into dev_dout
		cudaMemcpy(dev_dout, dout, outputDim * numSamples * sizeof(float), cudaMemcpyHostToDevice);
		checkCUDAError("cuda Memcpy dout in failed");

		//Make 3 diff grid layouts
		dim3 weightBiasGrid((inputDim * outputDim + blockSize - 1) / blockSize);
		dim3 outputGrid = (numSamples * outputDim + blockSize - 1) / blockSize;
		dim3 inputGrid = ((numSamples * inputDim + blockSize - 1) / blockSize);

		if (sigmoid) {
			//Get derivative of softmax, and update 
			kern_dSigmoid<<<outputGrid, blockSize >>>(dev_dout, dev_doutLinear, inputDim, outputDim);
			cudaFree(&dev_dout);
		}

		//Use matrix to compute dIn 
		kern_dIn<<<inputGrid, blockSize >>>(dev_doutLinear, W, dev_din, inputDim, outputDim, numSamples);

		//Update dw and db
		kern_dW<<<weightBiasGrid, blockSize >>>(W, b, dev_doutLinear, dev_in, inputDim, outputDim, numSamples, lr);

		//Memcpy back the din info
		float *din = new float[outputDim * numSamples];
		cudaMemcpy(dev_din, din, inputDim * numSamples * sizeof(float), cudaMemcpyDeviceToHost);
		checkCUDAError("cuda Memcpy din in failed");
		cudaFree(dev_din);
		return din;
	}
}
