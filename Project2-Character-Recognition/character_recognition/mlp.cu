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

	void printFloatArray(float *x, int n) {
		printf("    [ ");
		for (int i = 0; i < n; i++) {
			printf("%f ", x[i]);
		}
		printf("]\n");
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
		W[index] = dist(rng);
		//W[index] = 0.1 * index;
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
			out[row * outputDim + col] = sigmoid ? 1/(1+__expf(-val)) : val;
		}
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
			din[row * inputDim + col] = val;
		}
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
				val += in[i * inputDim + row] * doutLinearIdx;
				dbval += doutLinearIdx;
			}
			W[row * outputDim + col] = currW - lr * (val);
			b[col] = currb - lr * (dbval);
		}
	}

	__global__ void kernStableSoftmax(float *pred, float *pred2, float *target, int *sums, int numSamples, int outputDim) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		int row = index / outputDim;
		float rowval = 0.0;
		if (index < numSamples * outputDim) {
			for (int i = 0; i < outputDim; i++) {
				rowval += pred2[row * outputDim + i];
			}
			sums[row] = rowval;
			pred[index] = expf(pred2[index]);
			pred[index] = pred2[index] / rowval;
		}
	}

	__global__ void kernSums(float *pred, int *sums, int numSamples, int outputDim) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		int row = index / outputDim;
		float rowval = 0.0;
		if (index < numSamples * outputDim) {
			for (int i = 0; i < outputDim; i++) {
				rowval += pred[row * outputDim + i];
			}
			sums[row] = rowval;
		}
	}

	//AffineLayer 
	AffineLayer::AffineLayer(int idim, int odim, int ns): numSamples(ns), inputDim(idim), outputDim(odim), sigmoid(true), eval(false), doneFwd(false){
		//Malloc Weights, Biases, in and out
		cudaMalloc((void**)&W, idim * odim * sizeof(float));
		checkCUDAError("cuda Malloc W failed");
		cudaMalloc((void**)&b, odim * sizeof(float));
		checkCUDAError("cuda Malloc b failed");
		cudaMalloc((void**)&dev_in, inputDim * numSamples * sizeof(float));
		checkCUDAError("cuda Malloc dev_in in failed");

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
		float *dev_out;
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
		cudaFree(dev_out);
		return out;
	}

	float* AffineLayer::backward(float *dout, float lr){
		/* Does backprop and one gradient update for W & b & returns din
		dout: upstream gradient coming in 
		lr: learning rate
		Returns 
		*/
		//Malloc the input matrix and an output matrix 
		float *dev_dout, *dev_din, *dev_doutLinear;
		cudaMalloc((void**)&dev_dout, outputDim * numSamples * sizeof(float));
		checkCUDAError("cuda Malloc dev_dout in failed");
		cudaMalloc((void**)&dev_din, inputDim * numSamples * sizeof(float));
		checkCUDAError("cuda Malloc dev_din in failed");

		//Memcpy the *dout information into dev_dout
		cudaMemcpy(dev_dout, dout, outputDim * numSamples * sizeof(float), cudaMemcpyHostToDevice);
		checkCUDAError("cuda Memcpy dout in failed");

		//Make 3 diff grid layouts
		dim3 weightBiasGrid((inputDim * outputDim + blockSize - 1) / blockSize);
		dim3 outputGrid = (numSamples * outputDim + blockSize - 1) / blockSize;
		dim3 inputGrid = ((numSamples * inputDim + blockSize - 1) / blockSize);

		if (sigmoid) {
			cudaMalloc((void**)&dev_doutLinear, outputDim * numSamples * sizeof(float));
			checkCUDAError("cuda Malloc dev_din in failed");
			//Get derivative of softmax, and update 
			kern_dSigmoid<<<outputGrid, blockSize >>>(dev_dout, dev_doutLinear, inputDim, outputDim);
		}
		else {
			dev_doutLinear = dev_dout;
		}

		//Use matrix to compute dIn 
		kern_dIn<<<inputGrid, blockSize >>>(dev_doutLinear, W, dev_din, inputDim, outputDim, numSamples);


		//Update dw and db
		kern_dW<<<weightBiasGrid, blockSize >>>(W, b, dev_doutLinear, dev_in, inputDim, outputDim, numSamples, lr);

		//DEBUG STUFF
		float *myW= new float[inputDim * outputDim];
		cudaMemcpy(myW, W, inputDim * outputDim * sizeof(float), cudaMemcpyDeviceToHost);
		printf("MY WSTARTS\n");
		printFloatArray(myW, inputDim * outputDim);
		printf("MY WENDS\n");

		//Memcpy back the din info
		float *din = new float[inputDim * numSamples];
		cudaMemcpy(din, dev_din, inputDim * numSamples * sizeof(float), cudaMemcpyDeviceToHost);
		checkCUDAError("cuda Memcpy din in failed");


		//Free Mems
		cudaFree(dev_doutLinear);
		cudaFree(dev_din);
		return din;
	}

	void cpu_softmax(float *pred, int numSamples, int outputDim) {
		float rowSum = 0;
		for (int i = 0; i < numSamples; ++i) {
			rowSum = 0;
			for (int j = 0; j < outputDim; ++j) {
				pred[i * outputDim + j] = exp(pred[i * outputDim + j]);
				rowSum += pred[i * outputDim + j];
			}
			for (int k = 0; k < outputDim; ++k){
				pred[i * outputDim + k] /= rowSum;
			}
		}
	}

	float cpu_crossEntropy(float *pred, float *target, int numSamples, int outputDim, float* dout){
		float* log_likelihood = new float[numSamples];
		float llsum = 0;
		for (int i = 0; i < numSamples; ++i) {
			for (int c = 0; c < outputDim; ++c) {
				float ting = pred[i * outputDim + c];
				dout[i * outputDim + c] = ting;
			}
		}

		for (int i = 0; i < numSamples; ++i) {
			int offset = target[i];
			float ting = pred[i * outputDim + offset];
			log_likelihood[i] = -log(ting);
			llsum += -log(ting);
			dout[i * outputDim + offset] -= 1;
			for (int c = 0; c < outputDim; ++c) {
				dout[i * outputDim + c] /= numSamples;
			}
		}
		return llsum / numSamples;
	}

	float softmax_loss(float *pred, float *target, float *dout, int numSamples, int outputDim) {
		/* Returns a float representing the loss, and updates dout
		pred: Shape numSamples x outputDim
		target: Shape numSamples
		dout: Each element
		*/

		//Apply Softmax to pred
		cpu_softmax(pred, numSamples, outputDim);

		float loss = cpu_crossEntropy(pred, target, numSamples, outputDim, dout);
		return loss;
	}

	void XORTest() {
		//Network Structure
		int numSamples = 4;
		int inputDim = 2;
		int hiddenDim[1] = { 3 };
		int outputDim = 2;

		//XOR Input Array and Target Array
		float *x = new float[numSamples * inputDim];
		float *target = new float[numSamples * outputDim];
		x[0] = 0;
		x[1] = 0;
		target[0] = 0;
		x[2] = 0;
		x[3] = 1;
		target[1] = 1;
		x[4] = 1;
		x[5] = 0;
		target[2] = 1;
		x[6] = 1;
		x[7] = 1;
		target[3] = 0;

		//Build Layers
		AffineLayer* layer1 = new AffineLayer(inputDim, hiddenDim[0], numSamples);
		//layer1->setSigmoid(false);
		AffineLayer* layer1copy = new AffineLayer(inputDim, hiddenDim[0], numSamples);
		//layer1copy->setSigmoid(false);
		AffineLayer* layer2 = new AffineLayer(hiddenDim[0], outputDim, numSamples);
		layer2->setSigmoid(false);
		float lr = 0.01;
		for (int l = 0; l < 100; ++l) {
			/* FORWARD PROP */
			float *out0, *out1;
			printf("IN\n");
			printFloatArray(x, numSamples * outputDim);
			out0 = layer1->forward(x, numSamples);
			printf("OUT0\n");
			printFloatArray(out0, numSamples * outputDim);
			out1 = layer2->forward(out0, numSamples);
			printf("OUT1\n");
			printFloatArray(out1, numSamples * outputDim);

			/* CALCULATE LOSS */
			float *dout = new float[outputDim * numSamples];
			float loss = softmax_loss(out1, target, dout, numSamples, outputDim);
			printf("LOSS:%f\n", loss);
			printFloatArray(dout, outputDim * numSamples);

			/* BACKWARD PROP */
			float *dout1, *dout0;
			dout1 = layer2->backward(dout, lr);
			dout0 = layer1->backward(dout1, lr);
			printf("DOUT0\n");
			printFloatArray(dout0, inputDim * numSamples);
			printf("======================================\n", loss);
		}
	}
}
