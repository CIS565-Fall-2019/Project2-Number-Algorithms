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

void createInputXor(float *input) {
	input[0] = 0.0;
	input[1] = 0.0;
	input[2] = 0.0;
	input[3] = 1.0;
	input[4] = 1.0;
	input[5] = 0.0;
	input[6] = 1.0;
	input[7] = 1.0;
}

void createOutputXor(float *output) {
	output[0] = 0.0;
	output[1] = 1.0;
	output[2] = 1.0;
	output[3] = 0.0;
}


int main(int argc, char* argv[]) {
	
	printf("\n");
	printf("****************\n");
	printf("** CREATING THE NEURAL NETWORK **\n");
	printf("****************\n");

	const int INPUT_SIZE = 4; // Input size
	const int HIDDENLAYER_SIZE = 2; // Output size
	const int OUTPUT_SIZE = 1; // Output size
	const int FEATURE_SIZE = 2;

	float *input = new float[INPUT_SIZE*FEATURE_SIZE];
	float *hidden = new float[HIDDENLAYER_SIZE];
	float *output = new float[OUTPUT_SIZE];
	float *weightsIH = new float[HIDDENLAYER_SIZE*FEATURE_SIZE];
	float *weightsHO = new float[HIDDENLAYER_SIZE*OUTPUT_SIZE];

	createInputXor(input);
	genArray(FEATURE_SIZE*HIDDENLAYER_SIZE, weightsIH, 100);
	genArray(HIDDENLAYER_SIZE*OUTPUT_SIZE, weightsHO, 100);

	CharacterRecognition::createNN(INPUT_SIZE, HIDDENLAYER_SIZE, OUTPUT_SIZE, input, output, weightsIH, weightsHO);

	return 0;
}
