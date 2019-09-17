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

//====CONFIG Neural Network for XOR ================
//==================================================

/*
// XOR
const int N = 4;     // Number of examples
const int D = 2;     //Feature length per example
const int H = 2;     // Number of Hidden unit
const int C = 2;     // NN number of classes
const double LR = 0.5;
const int epochs = 1000;
*/

// Char Recognition
const int N = 52;     // Number of examples
const int D = 10201;  // Feature length per example
const int H = 10;     // Number of Hidden unit
const int C = 52;     // NN number of classes
const double LR = 0.5;
const int epochs = 5000;

double *losses = new double[epochs];
double *idata = new double[N*D];
int * preds = new int[N];
int * gtruth = new int[N];


int main(int argc, char* argv[]) {
    // Scan tests

    printf("\n");
    printf("****************\n");
    printf("***MLP TESTS***\n");
    printf("****************\n");

	printf("Launch XOR Training\n");


	/*
	// XOR input dtat set 2 * 4
	idata[0] = 0;
	idata[1] = 0;
	idata[2] = 0;
	idata[3] = 1;
	idata[4] = 1;
	idata[5] = 0;
	idata[6] = 1;
	idata[7] = 1;

	// XOR input dtat set 2 * 4
	gtruth[0] = 0;
	gtruth[1] = 1;
	gtruth[2] = 1;
	gtruth[3] = 0;

	CharacterRecognition::trainMLP(N, D, H, C, idata, preds, gtruth, epochs, losses, LR);
	printf("\nCompleted XOR Training\n");
	*/

	// Data loading


	return 0;
}
