#include <iostream>
#include <fstream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include "common.h"
#include "mlp.h"

namespace CharacterRecognition {

		using Common::PerformanceTimer;
		PerformanceTimer& timer()
		{
			static PerformanceTimer timer;
			return timer;
		}

		//TODO: implement required elements for MLP sections 1 and 2 here
		#define blockSize 128

		//neural number
		int input_count;//size of input
		int SIZE_INPUT;// the size of input layer
		int SIZE_HiD;// the size of hidden layer
		int SIZE_OUTPUT;//the size of output class

		//data pointer
		float *dev_input;
		float *dev_hid;
		float *dev_output;
		//weights
		float *weights_inandhid;// the input->hidden layer weights
		float *weights_hidandoutput;// the hidden->output layer weights
		float *dev_real;
		float *hid_sig;
		float *out_soft;
		//gradient
		float *wgrad_i2h;
		float *wgrad_h2o;

		//learning rate
		float lr;
		//loss
		float loss_threshold;

		//function variable
		float exp_sum;

		bool flag = false;
		
		//init an array
		__global__ void Init(int n, float *data, float value) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) {
				return;
			}
			data[index] = value;
		}
		
		void printM(const float *M, 
			int row, int col) {
			for (int i = 0; i < row; ++i) {
				for (int j = 0; j < col; ++j) {
					printf("%f ", M[j * row + i]);
				}
				printf("\n");
			}
			printf("\n");
		}

		void Write_Weights2File(std::string filename,
			const float *M, int row, int col) {
			//std::cout << "print weights" << std::endl;
			std::ofstream file(filename);
			if (file.is_open()) {
				for (int i = 0; i < row; i++) {
					for (int j = 0; j < col; j++) {
						//std::cout<<i<<","<<j<<": "<<M[j * row + i]<<std::endl;
						file << M[j * row + i] << " ";
					}
					file << "\n";
				}
				file << "\n";
			}
			file.close();
		}

		/////////////////////////////update layer output////////////////////////////////////
		/*
		cublasStatus_t cublasSgemm(cublasHandle_t handle,
			cublasOperation_t transa, cublasOperation_t transb,
			int m, int n, int k,
			const float *alpha,
			const float *A, int lda,
			const float *B, int ldb,
			const float *beta,
			float *C, int ldc)
		CUBLAS_OP_N do not need transpose, CUBLAS_OP_T need transpose
		m : A's row count
		n : B's col count
		k : a's col count，b's row count
		ld row count */
		//C(m,n) = A(m,k) * B(k,n)
		void Mul(const float *A, const float *B, float *C,
			int m, int n, int k,
			cublasHandle_t &handle) {
			int lda = m, ldb = k, ldc = m; 
			const float alpha = 1;
			const float beta = 0;
			cublasSgemm(handle, 
				CUBLAS_OP_N, CUBLAS_OP_N, 
				m, n, k, 
				&alpha, 
				A, m, 
				B, k, 
				&beta, 
				C, m);
		}

		//forward
		void GetLayerOutput(int size_of_current_layer,
			int size_of_next_layer,
			const float *idata,
			const float *weights,
			float *odata,
			cublasHandle_t &handle) {		
			const float alpha = 1;
			const float beta = 0;

			int lda = 1, ldc = 1, ldb = size_of_current_layer;
			cublasSgemm(handle, 
				CUBLAS_OP_N, CUBLAS_OP_N,
				1, //A's row count
				size_of_next_layer, // B's col count 
				size_of_current_layer,//a's col count，b's row count
				&alpha, 
				idata, //Ma
				lda, //lda
				weights, //Mb
				ldb,//ldb
				&beta, 
				odata, //MC
				ldc);//ldc
		}

		/////////////////////////////sigmoid(act fun)////////////////////////////////////
		//f(x) = 1/(1 + e^-x).
		__global__ void kernSigmoid(int n, const float *idata, float *odata) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) {
				return;
			}
			odata[index] = 1.0 / (1.0 + expf(-idata[index]));
		}

		/////////////////////////////softmax////////////////////////////////////
		//e^x
		__global__ void kernExp(int n, const float *idata, float* odata) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) {
				return;
			}
			odata[index] = expf(idata[index]);
		}

		float Get_sum(int n, float *arr) {
			float sum = 0.0;
			for (int i = 0; i < n; i++) {
				sum += arr[i];
			}
			return sum;
		}

		//normalize
		__global__ void kernNormalize(int n, float arr_sum,
			const float *arr_exp, float *odata) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) {
				return;
			}
			odata[index] = arr_exp[index] / arr_sum;
		}

		void softmax(int n, float* idata, float* odata) {
			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
			//get exp for the whole input
			float* arr_exp;
			cudaMalloc((void**)&arr_exp, sizeof(float) * n);
			checkCUDAError("cudaMalloc arr_exp failed!");
			Init << <fullBlocksPerGrid, blockSize >> > (n, arr_exp, 0.0);
			kernExp<< <fullBlocksPerGrid, blockSize >> > (n, idata, arr_exp);

			//get sum 
			float* exp = new float[n];
			cudaMemcpy(exp, arr_exp, sizeof(float) * n, cudaMemcpyDeviceToHost);//get
			/*std::cout << "exp: " << std::endl;
			printM(exp, n, 1);*/
			exp_sum = Get_sum(n, exp);

			//normalize
			kernNormalize<< <fullBlocksPerGrid, blockSize >> > (n, exp_sum, arr_exp, odata);
			
			cudaFree(arr_exp);
			free(exp);
		}

		//finally get the judgemenet
		//return max_index + 1 so it is from 1 - n not from 0 to n-1
		int GetfinalJudge(int n, float* after_softmax) {
			int max_index = -1;
			float max_prob = 0.0;
			for (int i = 0; i < n; i++) {
				if (max_prob < after_softmax[i]) {
					max_prob = after_softmax[i];
					max_index = i;
				}
			}
			return max_index + 1;
		}

		/////////////////////////////loss function////////////////////////////////////
		__global__ void kernmse_loss(int n, const float* real, 
			const float* predict, float* each) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) {
				return;
			}
			each[index] = 0.5 * (real[index] - predict[index]) * (real[index] - predict[index]);
		}

		__global__ void kerncross_entropy(int n, const float* real,
			const float* predict, float* each) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) {
				return;
			}
			each[index] = -1.0 *(real[index] * logf(predict[index]));
		}

		//loss
		float compute_loss(const float* real, const float *pred, int ind) {
			//std::cout << "hi! compute loss!" << std::endl;
			float *each_cros;
			cudaMalloc((void**)&each_cros, sizeof(float) * SIZE_OUTPUT);
			checkCUDAError("cudaMalloc each_cros failed!");

			float *real_each;
			cudaMalloc((void**)&real_each, sizeof(float) * SIZE_OUTPUT);
			checkCUDAError("cudaMalloc real_each failed!");
			cudaMemcpy(real_each, real + (ind * SIZE_OUTPUT), sizeof(float) * SIZE_OUTPUT, cudaMemcpyDeviceToDevice);
			
			dim3 fullBlocksPerGrid((SIZE_OUTPUT + blockSize - 1) / blockSize);
			//kernmse_loss << <fullBlocksPerGrid, blockSize >> > (SIZE_OUTPUT, real_each, pred, each_cros);
			kerncross_entropy << <fullBlocksPerGrid, blockSize >> >(SIZE_OUTPUT, real_each, pred, each_cros);
			
			//sum
			float *each_cros_host = new float[SIZE_OUTPUT];
			cudaMemcpy(each_cros_host, each_cros, sizeof(float) * SIZE_OUTPUT, cudaMemcpyDeviceToHost);//host to device
			//std::cout << "each loss:" << std::endl;
			//printM(each_cros_host, 1, SIZE_OUTPUT);

			float loss = Get_sum(SIZE_OUTPUT, each_cros_host);

			cudaFree(each_cros);
			free(each_cros_host);
			return loss;
		}
		
		/////////////////////////////gradient////////////////////////////////////
		__global__ void kernAdjW(int n, float *M) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) {
				return;
			}
			M[index] = 2.0 * M[index] - 1.0;
		}

		//(SIZE_OUTPUT, out_soft, dev_odata, error1)
		__global__ void kernSub(int n, float* A, float* B, float* C) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) {
				return;
			}
			C[index] = A[index] - B[index];
		}

		__global__ void kernAdd(int n, float* A, float* B, float* C) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) {
				return;
			}
			C[index] = A[index] + B[index];
		}

		//sigfun is the sig result
		__global__ void kernSig_partial_deriv(int n, float* sigfun, float* odata) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) {
				return;
			}
			odata[index] *= (sigfun[index] * (1.0 - sigfun[index]));
		}

		__global__ void kernMse_deri(int n, const float *input, float *real, float *odata) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) {
				return;
			}
			odata[index] *= (input - real);
		}

		__global__ void kernUp_Wei(int n, float *wei, float* gradient, float learning_rate) {
			int index = threadIdx.x + (blockIdx.x * blockDim.x);
			if (index >= n) {
				return;
			}
			wei[index] = wei[index] - learning_rate * gradient[index];
		}

		//for simple one data meg
		void forwardpass(int instance_index, float* idata, 
			cublasHandle_t &handle) {
			float* data;
			cudaMalloc((void**)&data, SIZE_INPUT * sizeof(float));
			checkCUDAError("cudaMalloc data failed!");
			cudaMemcpy(data, idata + (instance_index * SIZE_INPUT), sizeof(float) * SIZE_INPUT, cudaMemcpyDeviceToDevice);

			float* host_data = new float[SIZE_INPUT];
			cudaMemcpy(host_data, data, sizeof(float) * SIZE_INPUT, cudaMemcpyDeviceToHost);

			//compute hidden layer
			GetLayerOutput(SIZE_INPUT,//size_of_current_layer
				SIZE_HiD,//size_of_next_layer
				data,//idata
				weights_inandhid,//weights
				dev_hid,//hidden
				handle);

			//Compute sigmoid if hidden layer
			dim3 fullBlocksPerGrid((SIZE_HiD + blockSize - 1) / blockSize);
			kernSigmoid<< <fullBlocksPerGrid, blockSize >> > (SIZE_HiD, dev_hid, hid_sig);

			//Compute output layer
			GetLayerOutput(SIZE_HiD,//size_of_current_layer
				SIZE_OUTPUT,//size_of_next_layer
				hid_sig,//hidden_sig
				weights_hidandoutput,//weights
				dev_output,//output layer
				handle);

			//Compute softmax of output layer
			softmax(SIZE_OUTPUT, dev_output, out_soft);
		}

		//Computes the gradient for the current pass. 
		void compute_grad(int instance_number, 
			const float* input, const float* real, 
			cublasHandle_t &handle) {

			float* dev_idata;
			float* dev_odata;
			cudaMalloc((void**)&dev_idata, sizeof(float) * SIZE_INPUT);
			checkCUDAError("cudaMalloc dev_idata failed!");	
			cudaMalloc((void**)&dev_odata, sizeof(float) * SIZE_OUTPUT);
			checkCUDAError("cudaMalloc dev_odata failed!");

			cudaMemcpy(dev_idata, input + (instance_number * SIZE_INPUT), sizeof(float) * SIZE_INPUT, cudaMemcpyDeviceToDevice);
			cudaMemcpy(dev_odata, real + (instance_number * SIZE_OUTPUT), sizeof(float) * SIZE_OUTPUT, cudaMemcpyDeviceToDevice);

			//Compute gradient weights between hidden and output layer
			float* error1;
			cudaMalloc((void**)&error1, SIZE_OUTPUT * sizeof(float));
			checkCUDAError("cudaMalloc error1 failed!");
			float* error2;// input and hidden layer
			cudaMalloc((void**)&error2, SIZE_HiD * sizeof(float));
			checkCUDAError("cudaMalloc error2 failed!");

			//the error between predict and real
			dim3 fullBlocksPerGrid((SIZE_OUTPUT + blockSize - 1) / blockSize);
			//softmax and entrpy_cross derivative
			kernSub<< <fullBlocksPerGrid, blockSize >> > (SIZE_OUTPUT, out_soft, dev_odata, error1);
		
			Mul(hid_sig,//A(hid_size*1)  m*k
				error1,//B (1 * out_size) k*n
				wgrad_h2o,//C (SIZE_HiD * SIZE_OUTPUT) m*n
				SIZE_HiD,//m
				SIZE_OUTPUT,//n
				1,//k
				handle);

			Mul(weights_hidandoutput,//(size_output*size_hid)
				error1,//(size_out*1)
				error2,//(hid_size*1)
				SIZE_HiD,//m
				1,//n
				SIZE_OUTPUT,//k
				handle);

			//sigmoid derivative 
			dim3 fullBlocksPerGrid2((SIZE_HiD + blockSize - 1) / blockSize);
			kernSig_partial_deriv<< <fullBlocksPerGrid2, blockSize >> >(SIZE_HiD, hid_sig, error2);

			Mul(dev_idata,//(size_input*1)
				error2,//(1*hid_size)
				wgrad_i2h,//size_input*size_hid
				SIZE_INPUT,//m
				SIZE_HiD,//n
				1,//k
				handle);

			/*//debug
			float *t1 = new float[SIZE_HiD*SIZE_INPUT];
			cudaMemcpy(t1, wgrad_i2h, sizeof(float) * SIZE_HiD*SIZE_INPUT, cudaMemcpyDeviceToHost);
			printf("grad weight hid input: \n");
			//printM(t1, SIZE_INPUT, SIZE_HiD);

			//yes
			float *t2 = new float[SIZE_HiD*SIZE_OUTPUT];
			cudaMemcpy(t2, wgrad_h2o, sizeof(float) * SIZE_HiD*SIZE_OUTPUT, cudaMemcpyDeviceToHost);
			printf("grad weight hid output: \n");
			//printM(t2, SIZE_OUTPUT, SIZE_HiD);
			*/

			cudaFree(error1);
			cudaFree(error2);
			cudaFree(dev_idata);
			cudaFree(dev_odata);
		}

		//cublas is col based!
		//ref: https://blog.csdn.net/zcy0xy/article/details/84555053#cuBLAS_12
		void build_network(int data_count, int num_feature, int num_class,
			int hid_size, float ler, float loss_thre) {

			input_count = data_count;
			SIZE_INPUT = num_feature;
			SIZE_HiD = hid_size;
			SIZE_OUTPUT = num_class;
			lr = ler;
			loss_threshold = loss_thre;

			//malloc memory for layer weights
			cudaMalloc((void**)&weights_inandhid, (SIZE_INPUT * SIZE_HiD) * sizeof(float));
			checkCUDAError("cudaMalloc weights_inandhid failed!");
			cudaMalloc((void**)&weights_hidandoutput, (SIZE_HiD * SIZE_OUTPUT) * sizeof(float));
			checkCUDAError("cudaMalloc weights_hidandoutput failed!");

			//initialize weight with random number
			ref: https://blog.csdn.net/wesley_2013/article/details/12175391
			curandGenerator_t gen;
			curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MRG32K3A);//set the random algorithm
			curandSetPseudoRandomGeneratorSeed(gen, rand());//initial random number
			//set weights
			curandGenerateUniform(gen, weights_inandhid, SIZE_INPUT * SIZE_HiD);
			curandGenerateUniform(gen, weights_hidandoutput, SIZE_HiD * SIZE_OUTPUT);

			//FROM 0-1 to -1-1
			dim3 fullBlocksPerGrid((SIZE_INPUT * SIZE_HiD + blockSize - 1) / blockSize);
			kernAdjW << <fullBlocksPerGrid, blockSize >> > (SIZE_INPUT * SIZE_HiD, weights_inandhid);
			dim3 fullBlocksPerGrid2((SIZE_HiD * SIZE_OUTPUT + blockSize - 1) / blockSize);
			kernAdjW << <fullBlocksPerGrid2, blockSize >> > (SIZE_HiD * SIZE_OUTPUT, weights_hidandoutput);

			/*//debug
			float * wih = new float[SIZE_INPUT * SIZE_HiD];
			float * who = new float[SIZE_OUTPUT * SIZE_HiD];
			cudaMemcpy(wih, weights_inandhid, sizeof(float) * (SIZE_INPUT * SIZE_HiD), cudaMemcpyDeviceToHost);
			cudaMemcpy(who, weights_hidandoutput , sizeof(float) * (SIZE_OUTPUT * SIZE_HiD), cudaMemcpyDeviceToHost);
			*/

			//hidden layer and output layer on device
			cudaMalloc((void**)&dev_hid, SIZE_HiD * sizeof(float));
			checkCUDAError("cudaMalloc dev_hid failed!");
			cudaMalloc((void**)&dev_output, SIZE_OUTPUT * sizeof(float));
			checkCUDAError("cudaMalloc dev_output failed!");

			//weights grads dev memory
			cudaMalloc((void**)&wgrad_i2h, (SIZE_INPUT * SIZE_HiD) * sizeof(float));
			checkCUDAError("cudaMalloc wgrad_i2h failed!");
			cudaMalloc((void**)&wgrad_h2o, (SIZE_HiD * SIZE_OUTPUT) * sizeof(float));
			checkCUDAError("cudaMalloc wgrad_h2o failed!");

			//mid-calculation data memory
			cudaMalloc((void**)&hid_sig, SIZE_HiD * sizeof(float));
			checkCUDAError("cudaMalloc hid_sig failed!");
			cudaMalloc((void**)&out_soft, SIZE_OUTPUT * sizeof(float));
			checkCUDAError("cudaMalloc out_soft failed!");
			std::cout << "successfully build the network!" << std::endl;
		}

		void train(float* input, float* real, int train_time) {
			float epi_loss = 0.0;	

			//cuBlas handle
			cublasHandle_t handle;
			cublasCreate(&handle);

			//malloc memory for input and real
			cudaMalloc((void**)&dev_input, (input_count * SIZE_INPUT) * sizeof(float));
			checkCUDAError("cudaMalloc dev_input failed!");
			cudaMalloc((void**)&dev_real, (input_count * SIZE_OUTPUT) * sizeof(float));
			checkCUDAError("cudaMalloc dev_real failed!");
			cudaMemcpy(dev_input, input, sizeof(float) * (input_count * SIZE_INPUT), cudaMemcpyHostToDevice);
			cudaMemcpy(dev_real, real, sizeof(float) * (input_count * SIZE_OUTPUT), cudaMemcpyHostToDevice);

			for (int i = 0; i < train_time; i++) {
				//for each data
				epi_loss = 0.0;
				for (int j = 0; j < input_count; j++) {
					//Forward_Pass
					forwardpass(j, dev_input, handle);

					////////back propa//////////////////
					//get Loss
					epi_loss += compute_loss(dev_real, out_soft, j);
					//Compute grads
					compute_grad(j, dev_input, dev_real, handle);
					
					//update weights
					dim3 fullBlocksPerGrid(((SIZE_INPUT * SIZE_HiD) + blockSize - 1) / blockSize);
					kernUp_Wei<< <fullBlocksPerGrid, blockSize >> > (SIZE_INPUT * SIZE_HiD, weights_inandhid, wgrad_i2h, lr);
					dim3 fullBlocksPerGrid2(((SIZE_HiD * SIZE_OUTPUT) + blockSize - 1) / blockSize);
					kernUp_Wei<< <fullBlocksPerGrid2, blockSize >> >(SIZE_HiD * SIZE_OUTPUT, weights_hidandoutput, wgrad_h2o, lr);
					
				}
				std::cout << "epoch " << i << " loss : " << epi_loss / (1.0 * input_count) << std::endl;
				if (epi_loss / (1.0 * input_count) < loss_threshold && !flag) {
					flag = true;
				}
			}
			if (flag) {
				float * wih = new float[SIZE_INPUT * SIZE_HiD];
				float * who = new float[SIZE_OUTPUT * SIZE_HiD];
				cudaMemcpy(wih, weights_inandhid, sizeof(float) * (SIZE_INPUT * SIZE_HiD), cudaMemcpyDeviceToHost);
				cudaMemcpy(who, weights_hidandoutput, sizeof(float) * (SIZE_OUTPUT * SIZE_HiD), cudaMemcpyDeviceToHost);

				Write_Weights2File("input_hid_w.txt", wih, input_count, SIZE_HiD);
				Write_Weights2File("hid_out_w.txt", who, SIZE_HiD, SIZE_OUTPUT);

				delete(wih);
				delete(who);
			}
			// Destroy the handle 
			cublasDestroy(handle);
		}

		//test
		void test(float* test_input, int test_times, int type) {
			int t = type == 1 ? test_times : input_count;
			if (test_times <= 0 || !flag) {
				std::cout << "error!" << std::endl;
				return;
			}

			//dev memory for test data
			float* test_in;
			cudaMalloc((void**)&test_in, (input_count * SIZE_INPUT) * sizeof(float));
			checkCUDAError("cudaMalloc test_in failed!");
			cudaMemcpy(test_in, test_input, sizeof(float) * (input_count * SIZE_INPUT), cudaMemcpyHostToDevice);

			//CUBLAS handle
			cublasHandle_t handle;
			cublasCreate(&handle);

			int correct_time = 0;
			float *re = new float[SIZE_OUTPUT];

			if (type == 1) {
				//test specific times
				while (test_times) {
					int ind = (10000 * (int)rand()) % input_count;
					forwardpass(ind, test_in, handle);
					std::cout << "pic: " << ind + 1 << "  pred: ";
					cudaMemcpy(re, out_soft, sizeof(float) * SIZE_OUTPUT, cudaMemcpyDeviceToHost);
					int result = GetfinalJudge(SIZE_OUTPUT, re);
					if (result == ind + 1) {
						correct_time++;
					}
					std::cout << "  " << result << std::endl;
					test_times--;
				}
			} 
			else if (type ==2) {
				for (int i = 0; i < input_count; i++) {
					forwardpass(i, test_in, handle);
					std::cout << "pic: " << i + 1 << "  pred: ";
					cudaMemcpy(re, out_soft, sizeof(float) * SIZE_OUTPUT, cudaMemcpyDeviceToHost);
					int result = GetfinalJudge(SIZE_OUTPUT, re);
					if (result == i + 1) {
						correct_time++;
					}
					std::cout << "  " << result << std::endl;
				}
			}
			std::cout << "test time: " << t 
				<< " ,correct probability: " 
				<< ((1.0 * correct_time) / t) * 100.0 <<"%"<<std::endl;

			delete(re);
			cublasDestroy(handle);
		}
}
