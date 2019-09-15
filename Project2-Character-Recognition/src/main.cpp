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

#define sizeData 10205
#define numLabels 52

void readData(float *X, float *y) {
	int c = 0;
	for (int i = 1; i <= numLabels; i++) {
		float ascii = 65 + c;
		if (i % 2 == 0) { c++; }
		float yi = i % 2 == 0 ? ascii+32 : ascii;
		y[(i-1)*(numLabels+1)] = yi;

		float xi;
		std::string n = i < 10 ? "0" + std::to_string(i) : std::to_string(i);
		std::string filePath = "../data-set/" + n + "info.txt";
		std::ifstream dataFile(filePath);
		int k = 0;
		while (!dataFile.fail() && !dataFile.eof())
		{
			dataFile >> xi;
			X[sizeData*(i-1) + k++] = xi;
		};
	}
}

int main(int argc, char* argv[]) {
    printf("\n");
    printf("****************\n");
    printf("** MLP TESTS **\n");
    printf("****************\n");

	unsigned int size_X = sizeData*numLabels;
	unsigned int mem_size_X = sizeof(float) * size_X;
	float *X = (float *)malloc(mem_size_X);

	unsigned int size_y = numLabels*numLabels;
	unsigned int mem_size_y = sizeof(float) * size_y;
	float *y = (float *)malloc(mem_size_y);

	printDesc("reading data");
	readData(X,y);

	printDesc("test multiply");
	CharacterRecognition::testMatrixMultiply();

	printDesc("training");
	CharacterRecognition::train(X, y, sizeData, 10, numLabels);
	

	free(X);
	free(y);

    system("pause"); // stop Win32 console from closing on exit
}
