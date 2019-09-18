#pragma once

#include "common.h"
#include <cublas_v2.h>

constexpr int PIXEL_RES = 2;
constexpr int PIXELS = 10201 / PIXEL_RES;
constexpr int OUTPUTS = 52;

namespace CharacterRecognition {
    Common::PerformanceTimer& timer();

	class Matrix {
	public:
		Matrix(int colcnt, int rowcnt);
		~Matrix();

		int colcnt;
		int rowcnt;

		float* dev_data;
		float* cpu_data;

		void copyCpuToDev();
		void copyDevToCpu();

		void copyMatrix(Matrix* m);

		int getLen();

	private:
		// Memory-Management
		void cpuAlloc();
		void devAlloc();
		void cpuFree();
		void devFree();
	};

	class ImageFile {
	public:
		ImageFile(std::string filepath);
		~ImageFile();

		// Reads the data into a Matrix object
		void readImage(Matrix* m);

		int getExpectedNumber();

	private:
		std::FILE* fd;

		int expected_number; // TODO: This shouldn't be here. Rethink this...
		int pixels;
	};

	class Perceptron {
	public:
		Perceptron(int pixels, int outputs);
		~Perceptron();

		void randomizeWeights();
		void loadBrain(std::string brainfile);
		void saveBrain(std::string brainfile);

		void loadTrainingDataSet(ImageFile * input);
		void loadDataSet(ImageFile * input);

		void run();
		void train();

		int getLastResult();

		void updateBackprop();
		void applyBackprop();

		void updateCpu(); // For debugging

	private:
		Matrix inputLayer;
		Matrix hiddenLayer;
		Matrix outputLayer;
		Matrix expectedLayer;
		Matrix kjWeights;  // Input -> Hidden
		Matrix jiWeights;  // Hidden -> Output

		// Backprop data
		Matrix kjWeightsDelta;
		Matrix jiWeightsDelta;

		Matrix jiOmega;
		Matrix jiPsi;
		Matrix jiTheta;

		Matrix kjTheta;
		Matrix kjOmega;
		Matrix kjPsi;

		int result;
		float tr_runs;

		void impl_run(bool training);

		void calcPsi(const Matrix* omega, const Matrix* theta, Matrix* psi);
		void calcDeltaChange(const float lambda, const Matrix* leftLayer, const Matrix* psi, Matrix* weightDelta);
		void calcOmega(const Matrix* psi, const Matrix* weights, Matrix* omega);
	};

	static cublasHandle_t ch;

	void initCublas();
	void deleteCublas();

    // TODO: implement required elements for MLP sections 1 and 2 here
	void matrixMul(cublasHandle_t ch, const Matrix* A, const Matrix* B, Matrix* C);

	void sigmoid(Matrix* m);
	void reluActivate(Matrix* m);
	void softmaxActivate(Matrix* m);

	__device__ float devSigmoid(float n);
	__device__ float devInverseSigmoid(float n);
	__device__ int getGlobalIdx_3D_3D();
	__global__ void kernReluActivate(int n, float* in, float* out);
	__global__ void kernExponentiate(int n, float* in, float* out);

	void testMatrixMul();

}
