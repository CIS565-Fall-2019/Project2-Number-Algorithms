/**
 * @file      main.cpp
 * @brief     Stream compaction test program
 * @authors   Kai Ninomiya
 * @date      2015
 * @copyright University of Pennsylvania
 */

#include <cstdio>
#include <character_recognition/mlp.h>
#include <character_recognition/common.h>
#include "testing_helpers.hpp"
#include <fstream>

#define XOR 1
#define MULT 0

#if XOR
	#define sizeData 2
	#define numLabels 1
	#define numData 4
	#define hiddenNodes 2
#else
	#define sizeData 10205
	#define numLabels 52
	#define numData 52
	#define hiddenNodes 100
#endif // XOR

#define index(i,j,ld) (((j)*(ld))+(i))

void loadXorExample(float *X, float *y) {
	for (int i = 0; i < sizeData; i++) {
		for (int j = 0; j < sizeData; j++) {
			X[sizeData*(2 * i + j)] = i;
			X[sizeData*(2 * i + j) + 1] = j;
			y[2*i + j] = i ^ j;
		}
	}

	for (int i = 0; i <= 1; i++) {
		for (int j = 0; j <= 1; j++) {
			std::cout << "data: " << X[sizeData*(2 * i + j)] << X[sizeData*(2 * i + j) + 1] << " label: " << y[2 * i + j] << "\n";
		}
	}
}


void readImageData(float *X, float *y) {
	int c = 0;
	for (int i = 1; i <= numLabels; i++) {
		float ascii = 65 + c;
		if (i % 2 == 0) { c++; }
		float yi = i % 2 == 0 ? ascii + 32 : ascii;
		y[(i - 1)*(numLabels + 1)] = yi;

		float xi;
		std::string n = i < 10 ? "0" + std::to_string(i) : std::to_string(i);
		std::string filePath = "../data-set/" + n + "info.txt";
		std::ifstream dataFile(filePath);
		int k = 0;
		while (!dataFile.fail() && !dataFile.eof())
		{
			dataFile >> xi;
			X[sizeData*(i - 1) + k++] = xi;
		};
	}
}


void readImageWeights(float *wI, float *wO) {
	printf("Doesn't currently load any weights \n");
	return;
}


void fixedInit(float *data, int size) {
	if (size == 4) {
		data[0] = 10.1f;
		data[1] = 0.9f;
		data[2] = 20.0f;
		data[3] = 0.87f;
	}
	else if (size == 2) {
		data[0] = 41.0f;
		data[1] = -54.0f;
	}
}


int main(int argc, char* argv[]) {
	printf("\n");
	printf("****************\n");
	printf("** MLP TESTS **\n");
	printf("****************\n\n");

	unsigned int size_X = sizeData * numData;
	unsigned int mem_size_X = sizeof(float) * size_X;
	float *X = (float *)malloc(mem_size_X);

	unsigned int size_y = numLabels * numData;
	unsigned int mem_size_y = sizeof(float) * size_y;
	float *y = (float *)malloc(mem_size_y);

#if XOR
	printDesc("loading XOR example");
	loadXorExample(X, y);
#else
	printDesc("reading image data");
	readImageData(X, y);
#endif // XOR

	std::string train = "y";
	std::string test = "y";
	std::string input;
	int iterations;

	printf("Would you like to train or test? (q) ");
	std::cin >> input;

	while (input != "q") {
		if (input == "train") {
			while (train == "y" | train == "Y" | train == "yes" | train == "Yes") {
				printf("How many iterations would you like to do? ");
				std::cin >> iterations;
				printDesc("training");
				CharacterRecognition::train(X, y, iterations, sizeData, hiddenNodes, numLabels, numData);
				printf("Would you like to continue training? (y, n, q) ");
				std::cin >> train;
			}
		}
		else if (input == "test") {
				while (test == "y" | test == "Y" | test == "yes" | test == "Yes") {
					unsigned int size_wI = hiddenNodes * sizeData;
					unsigned int mem_size_wI = sizeof(float) * size_wI;
					float *wI = (float *)malloc(mem_size_wI);

					unsigned int size_wO = numLabels * hiddenNodes;
					unsigned int mem_size_wO = sizeof(float) * size_wO;
					float *wO = (float *)malloc(mem_size_wO);

#if XOR
					printDesc("loading XOR weights");
					fixedInit(wI, size_wI);
					fixedInit(wO, size_wO);
#else
					printDesc("reading image weights");
					readImageWeights(wI, wO);
#endif // XOR

					printDesc("testing");
					CharacterRecognition::test(X, y, wI, wO, sizeData, hiddenNodes, numLabels, numData);
					printf("Would you like to continue testing? (y, n, q) ");
					std::cin >> test;
				}
		}

		if (test == "q" || train == "q") break;
		printf("Would you like to train or test? (q) ");
		std::cin >> input;
	}

#if MULT
	std::string mult = "n";
	printf("Would you like to test matrix multiplication? (y) ");
	std::cin >> mult;

	while (mult == "y" | mult == "Y" | mult == "mult") {
		int HA = 2, WA = 2, HB = 2, WB = 1;
		printf("How many rows for the first matrix? ");
		std::cin >> HA;
		printf("How many columns for the first matrix? ");
		std::cin >> WA;
		printf("How many rows for the second matrix? ");
		std::cin >> HB;
		printf("How many columns for the second matrix? ");
		std::cin >> WB;

		printDesc("test multiply");
		CharacterRecognition::testMatrixMultiply(HA,WA,HB,WB);

		printf("Would you like continue testing matrix multiplication? (y) ");
		std::cin >> mult;
	}
#endif

	free(X);
	free(y);

    system("pause"); // stop Win32 console from closing on exit
}
