#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "mlp.h"
#include <cublas_v2.h>
#include <iostream>
#include <thrust/random.h>

// cuBLAS matrix multiplication
void gpu_blas_mmul(cublasHandle_t &handle, const float *A, const float *B, float *C, const int m, const int k, const int n) {
  const float alpha = 1.;
  const float beta = 0.;
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A, m, B, k, &beta, C, m);
}

// cuBLAS matrix transpose
void gpu_blas_mtrans(cublasHandle_t &handle, const float *A, float *B, const int m, const int n) {
  float alpha = 1.;
  float beta = 0.;
  cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, &alpha, A, n, &beta, A, n, B, m);
}

namespace CharacterRecognition {
  using Common::PerformanceTimer;
  PerformanceTimer& timer()
  {
      static PerformanceTimer timer;
      return timer;
  }

  const int blockSize = 128;
        
  // Kernals
  __global__ void kernCalcSigmoid(int n, float *out, float *in) {
    int idx = threadIdx.x + (blockIdx.x * blockDim.x);
    if (idx >= n) { 
      return; 
    }
    out[idx] = 1 / (1 + exp(-in[idx]));
  }

  __global__ void kernCalcSigmoidDerivative(int n, float *out, float *in) {
    int idx = threadIdx.x + (blockIdx.x * blockDim.x);
    if (idx >= n) {
      return;
    }
    float sigma = 1 / (1 + exp(-in[idx]));
    out[idx] = (1 - sigma) * sigma;
  }

  __global__ void kernElementWiseMultiplication(int n, float *out, float *in1, float *in2) {
    int idx = threadIdx.x + (blockIdx.x * blockDim.x);
    if (idx >= n) {
      return;
    }
    out[idx] = in1[idx] * in2[idx];
  }

  __global__ void kernUpdateWeight(int n, float *weights, float *gradients, float lambda) {
    int idx = threadIdx.x + (blockIdx.x * blockDim.x);
    if (idx >= n) {
      return;
    }
    weights[idx] = weights[idx] - gradients[idx] * lambda;
  }

  __global__ void kernElementWiseSubtraction(int n_cols, int n_rows, float *out, float *in1, float *in2) {
    int idx = threadIdx.x + (blockIdx.x * blockDim.x);
    if (idx >= n_cols * n_rows) {
      return;
    }
    out[idx] = in1[idx] - in2[idx];
  }

  __global__ void kernCalcSquareError(int n, float *out, float *in1, float *in2) {
    int idx = threadIdx.x + (blockIdx.x * blockDim.x);
    if (idx >= n) {
      return;
    }
    float diff = in1[idx] - in2[idx];
    out[idx] = diff * diff;
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

  __global__ void kernRandomNumber(int n, int time, float* array) {
    int idx = threadIdx.x + (blockIdx.x * blockDim.x);
    if (idx >= n) {
      return;
    }
    thrust::default_random_engine rng(hash((int)(idx * time)));
    thrust::uniform_real_distribution<float> unitDistrib(-1, 1);
    array[idx] = (float)unitDistrib(rng);
  }

	// TODO: implement required elements for MLP sections 1 and 2 here
  MLP3::MLP3(int input_size, int hidden_size, int output_size) {
    input_size_ = input_size;
    hidden_size_ = hidden_size;
    output_size_ = output_size;

    cudaMalloc((void**)&wkj_, input_size_ * hidden_size_ * sizeof(float));
    cudaMalloc((void**)&wji_, hidden_size_ * output_size_ * sizeof(float));
    cudaMalloc((void**)&gwkj_, input_size_ * hidden_size_ * sizeof(float));
    cudaMalloc((void**)&gwji_, hidden_size_ * output_size_ * sizeof(float));

    // create cublasHandle
    cublasCreate(&cublas_handle_);
  }

  MLP3::~MLP3() {
    cudaFree(wkj_);
    cudaFree(wji_);
    cudaFree(gwkj_);
    cudaFree(gwji_);

    // destroy cublasHandle
    cublasDestroy(cublas_handle_);
  }

  void MLP3::init_weights(float *wkj, float *wji) {
    cudaMemcpy(wkj_, wkj, sizeof(float) * input_size_ * hidden_size_, cudaMemcpyHostToDevice);
    cudaMemcpy(wji_, wji, sizeof(float) * hidden_size_ * output_size_, cudaMemcpyHostToDevice);
  }

  void MLP3::init_weights() {
    // TODO: initialize random weights
    int n_w1 = input_size_ * hidden_size_;
    int gridSize1 = (n_w1 + blockSize - 1) / blockSize;
    kernRandomNumber<<<gridSize1, blockSize>>>(n_w1, 1, wji_);

    int n_w2 = hidden_size_ * output_size_;
    int gridSize2 = (n_w2 + blockSize - 1) / blockSize;
    kernRandomNumber<<<gridSize1, blockSize>>>(n_w2, 1, wkj_);
  }

  void MLP3::train(float *x_train, float *y_train, int n_data, int n_epoch, bool verbose) {
    n_data_ = n_data;
    cudaMalloc((void**)&dev_input_, n_data * input_size_ * sizeof(float));
    cudaMalloc((void**)&dev_target_, n_data * output_size_ * sizeof(float));
    cudaMemcpy(dev_input_, x_train, n_data * input_size_ * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_target_, y_train, n_data * output_size_ * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&dev_hidden_, n_data * hidden_size_ * sizeof(float));
    cudaMalloc((void**)&dev_hidden_sigmoid_, n_data * hidden_size_ * sizeof(float));
    cudaMalloc((void**)&dev_output_, n_data * output_size_ * sizeof(float));
    cudaMalloc((void**)&dev_output_sigmoid_, n_data * output_size_ * sizeof(float));

    for (int epoch = 0; epoch < n_epoch; epoch++) {
      forward();
      back_propagation();
      calculate_loss();
      update_weights();
      if (verbose && ((epoch == n_epoch - 1) || (epoch + 1) % 5 == 0)) {
        std::cout << "epoch: " << epoch + 1 << ", cost: " << total_error_ << std::endl;
      }
    }

    cudaFree(dev_input_);
    cudaFree(dev_target_);
    cudaFree(dev_hidden_);
    cudaFree(dev_hidden_sigmoid_);
    cudaFree(dev_output_);
    cudaFree(dev_output_sigmoid_);
  }

  void MLP3::predict(float *x_input, float *y_pred, int n_data) {
    n_data_ = n_data;
    cudaMalloc((void**)&dev_input_, n_data * input_size_ * sizeof(float));
    cudaMemcpy(dev_input_, x_input, n_data * input_size_ * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaMalloc((void**)&dev_hidden_, n_data * hidden_size_ * sizeof(float));
    cudaMalloc((void**)&dev_hidden_sigmoid_, n_data * hidden_size_ * sizeof(float));
    cudaMalloc((void**)&dev_output_, n_data * output_size_ * sizeof(float));
    cudaMalloc((void**)&dev_output_sigmoid_, n_data * output_size_ * sizeof(float));

    forward();
    cudaMemcpy(y_pred, dev_output_sigmoid_, n_data * output_size_ * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(dev_input_);
    cudaFree(dev_hidden_);
    cudaFree(dev_hidden_sigmoid_);
    cudaFree(dev_output_);
    cudaFree(dev_output_sigmoid_);
  }

  void MLP3::forward() {
    gpu_blas_mmul(cublas_handle_, dev_input_, wkj_, dev_hidden_, n_data_, input_size_, hidden_size_);

    dim3 gridSize1((hidden_size_ * n_data_ + blockSize - 1) / blockSize);
    kernCalcSigmoid<<<gridSize1, blockSize>>>(hidden_size_ * n_data_, dev_hidden_sigmoid_, dev_hidden_);

    gpu_blas_mmul(cublas_handle_, dev_hidden_sigmoid_, wji_, dev_output_, n_data_, hidden_size_, output_size_);

    dim3 gridSize2((output_size_ * n_data_ + blockSize - 1) / blockSize);
    kernCalcSigmoid<<<gridSize2, blockSize>>>(output_size_ * n_data_, dev_output_sigmoid_, dev_output_);
  }

  void MLP3::back_propagation() {
    float* temp1, *temp2, *temp3, *temp4, *temp5, *temp6, *temp7, *temp8, *wji_T;

    cudaMalloc((void**)&temp1, n_data_ * output_size_ * sizeof(float));
    cudaMalloc((void**)&temp2, n_data_ * output_size_ * sizeof(float));
    cudaMalloc((void**)&temp3, n_data_ * output_size_ * sizeof(float));
    cudaMalloc((void**)&temp4, n_data_ * hidden_size_ * sizeof(float));
    cudaMalloc((void**)&temp5, n_data_ * hidden_size_ * sizeof(float));
    cudaMalloc((void**)&temp6, n_data_ * hidden_size_ * sizeof(float));
    cudaMalloc((void**)&wji_T, output_size_ * hidden_size_ * sizeof(float));
    cudaMalloc((void**)&temp7, n_data_ * hidden_size_ * sizeof(float));
    cudaMalloc((void**)&temp8, n_data_ * input_size_ * sizeof(float));

    // calculate wji gradient
    int num_element1 = output_size_ * n_data_;
    dim3 gridSize1((num_element1 + blockSize - 1) / blockSize);
    kernCalcSigmoidDerivative<<<gridSize1, blockSize>>>(num_element1, temp1, dev_output_sigmoid_);
    kernElementWiseSubtraction<<<gridSize1, blockSize>>>(n_data_, output_size_, temp2, dev_output_sigmoid_, dev_target_);
    kernElementWiseMultiplication<<<gridSize1, blockSize>>>(num_element1, temp3, temp1, temp2);

    gpu_blas_mtrans(cublas_handle_, dev_hidden_sigmoid_, temp4, hidden_size_, n_data_);
    gpu_blas_mmul(cublas_handle_, temp4, temp3, gwji_, hidden_size_, n_data_, output_size_);

    // calculate wkj gradient
    int num_element2 = hidden_size_ * n_data_;
    dim3 gridSize2((num_element2 + blockSize - 1) / blockSize);
    kernCalcSigmoidDerivative<<<gridSize2, blockSize>>>(num_element2, temp5, dev_hidden_sigmoid_);
    gpu_blas_mtrans(cublas_handle_, wji_, wji_T, output_size_, hidden_size_);
    gpu_blas_mmul(cublas_handle_, temp3, wji_T, temp6, n_data_, output_size_, hidden_size_);
    kernElementWiseMultiplication<<<gridSize2, blockSize>>>(num_element2, temp7, temp6, temp5);

    gpu_blas_mtrans(cublas_handle_, dev_input_, temp8, input_size_, n_data_);
    gpu_blas_mmul(cublas_handle_, temp8, temp7, gwkj_, input_size_, n_data_, hidden_size_);
  }

  void MLP3::calculate_loss() {
    float *dev_square_error;
    int n = n_data_ * output_size_;
    cudaMalloc((void**)&dev_square_error, sizeof(float) * n);
    int gridSize = (n + blockSize - 1) / blockSize;
    kernCalcSquareError<<<gridSize, blockSize>>>(n, dev_square_error, dev_target_, dev_output_sigmoid_);
    
    float *square_error = new float[n]();
    cudaMemcpy(square_error, dev_square_error, n * sizeof(float), cudaMemcpyDeviceToHost);

    total_error_ = 0.;
    for (int i = 0; i < n; i++) {
      total_error_ += square_error[i];
    }
    total_error_ /= (2.0 * output_size_);

    delete[] square_error;
    cudaFree(dev_square_error);
  }

  void MLP3::update_weights() {
    float lambda = 0.01;

    float* temp1 = new float(2);
    cudaMemcpy(temp1, wji_, 2 * sizeof(float), cudaMemcpyDeviceToHost);

    // update wji
    int gridSize1 = (hidden_size_ * output_size_ + blockSize) / blockSize;
    kernUpdateWeight<<<gridSize1, blockSize>>>(hidden_size_ * output_size_, wji_, gwji_, lambda);
    
    float* temp2 = new float(2);
    cudaMemcpy(temp2, wji_, 2 * sizeof(float), cudaMemcpyDeviceToHost);

    // update wkj
    int gridSize2 = (input_size_ * hidden_size_ + blockSize) / blockSize;
    kernUpdateWeight<<<gridSize2, blockSize>>>(input_size_ * hidden_size_, wkj_, gwkj_, lambda);
  }
}
