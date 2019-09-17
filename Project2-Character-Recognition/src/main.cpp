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

#define sizeData 2
#define numLabels 1
#define numData 4
#define hiddenNodes 2

#define index(i,j,ld) (((j)*(ld))+(i))

void readData(float *X, float *y) {
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

int main(int argc, char* argv[]) {
    printf("\n");
    printf("****************\n");
    printf("** MLP TESTS **\n");
    printf("****************\n");

	unsigned int size_X = sizeData* numData;
	unsigned int mem_size_X = sizeof(float) * size_X;
	float *X = (float *)malloc(mem_size_X);

	unsigned int size_y = numLabels*numData;
	unsigned int mem_size_y = sizeof(float) * size_y;
	float *y = (float *)malloc(mem_size_y);

	printDesc("reading data");
	readData(X,y);

	//printDesc("test multiply");
	//CharacterRecognition::testMatrixMultiply();

	printDesc("training");
	CharacterRecognition::train(X, y, sizeData, hiddenNodes, numLabels, numData);
	

	free(X);
	free(y);

    system("pause"); // stop Win32 console from closing on exit
}
