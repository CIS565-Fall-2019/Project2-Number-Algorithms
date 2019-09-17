#pragma once

#include "common.h"

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

		void loadTrainingDataSet(int expected_result, Matrix* input);
		void train();

		void loadDataSet(Matrix* input);
		void run();
		int getLastResult();

	private:
		Matrix inputData;
		Matrix hiddenLayer;
		Matrix outputLayer;
		Matrix kjWeights;  // Input -> Hidden
		Matrix jiWeights;  // Hidden -> Output

		// Backprop data
		Matrix kjWeightsDelta;
		Matrix jiWeightsDelta;

		int result;

		void backprop();
		void applyBackprop();

		void updateHiddenToOutputWeights();
	};
    // TODO: implement required elements for MLP sections 1 and 2 here
	void matrixMul(const Matrix* A, const Matrix* B, Matrix* C);
}
