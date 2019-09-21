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

const int SIZE = 1 << 3; // feel free to change the size of array
const int NPOT = SIZE - 3; // Non-Power-Of-Two
int *a = new int[SIZE];
int *b = new int[SIZE];
int *c = new int[SIZE];

void printFloatArray(float *x, int n) {
    printf("    [ ");
    for (int i = 0; i < n; i++) {
        printf("%f ", x[i]);
	}
	printf("]\n");
}

int main(int argc, char* argv[]) {
	/*
	CHARACTER RECOGNITION TESTS
	*/
    printf("\n");
    printf("****************\n");
    printf("** CHARACTER RECOGNITION TESTS **\n");
    printf("****************\n");

	//XOR Input Array
	int numSamples = 4;
	int inputDim = 2;
	int outputDim = 3;
	float *x = new float[numSamples * inputDim];
	for (int i = 0; i < SIZE; ++i) { 
		if (i % 2 == 0) {
			x[i] = 1;
		}
		else {
			x[i] = 0;
		}
	}
	printFloatArray(x, numSamples * inputDim);

	//Build Layers
	float *out;
	CharacterRecognition::AffineLayer layer1(inputDim, outputDim);
	layer1.setSigmoid(false);

	/* FORWARD PROP */
	out = layer1.forward(x, numSamples);
	printFloatArray(out, numSamples * outputDim);

	/* BACKWARD PROP */


    system("pause"); // stop Win32 console from closing on exit
	delete[] a;
	delete[] b;
	delete[] c;
}
