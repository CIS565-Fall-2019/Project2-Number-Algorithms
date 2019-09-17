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
#include <iostream>
#include <fstream>
#include <string.h> 
using namespace std;

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
	output[0] = 1.0;
	output[1] = 0.0;
	output[2] = 0.0;
	output[3] = 1.0;
	output[4] = 0.0;
	output[5] = 1.0;
	output[6] = 1.0;
	output[7] = 0.0;
}

void xor2x2NN(int INPUT_SIZE, int HIDDENLAYER_SIZE, int OUTPUT_SIZE, int FEATURE_SIZE) {
	float *input = new float[INPUT_SIZE*FEATURE_SIZE];
	float *hidden = new float[INPUT_SIZE*HIDDENLAYER_SIZE];
	float *output = new float[INPUT_SIZE*OUTPUT_SIZE];
	float *weightsIH = new float[HIDDENLAYER_SIZE*FEATURE_SIZE];
	float *weightsHO = new float[HIDDENLAYER_SIZE*OUTPUT_SIZE];
	float *outputNN = new float[INPUT_SIZE*OUTPUT_SIZE];

	
	createInputXor(input);
	createOutputXor(output);
	srand(10);
	genArrayA(FEATURE_SIZE*HIDDENLAYER_SIZE, weightsIH, 0);
	genArrayB(HIDDENLAYER_SIZE*OUTPUT_SIZE, weightsHO, 1);

	//zeroArray(FEATURE_SIZE*HIDDENLAYER_SIZE, weightsIH);
	//zeroArray(HIDDENLAYER_SIZE*OUTPUT_SIZE, weightsHO);

	printf("Weights A array: \n");
	printArray(HIDDENLAYER_SIZE*FEATURE_SIZE, weightsIH, true);
	printf("Weights B array: \n");
	printArray(HIDDENLAYER_SIZE*OUTPUT_SIZE, weightsHO, true);

	CharacterRecognition::createAndTrainNN(INPUT_SIZE, HIDDENLAYER_SIZE, OUTPUT_SIZE, FEATURE_SIZE, input, hidden, outputNN, weightsIH, weightsHO, output);

	printf("After NN and Training \n");
	printf("Input Array: \n");
	printArray(INPUT_SIZE*FEATURE_SIZE, input, true);
	printf("hidden Layer Array: \n");
	printArray(INPUT_SIZE*HIDDENLAYER_SIZE, hidden, true);
	printf("Output Array: \n");
	printArray(INPUT_SIZE*OUTPUT_SIZE, outputNN, true);
	printf("Actual Output Array: \n");
	printArray(INPUT_SIZE*OUTPUT_SIZE, output, true);
	printf("Weights A array: \n");
	printArray(HIDDENLAYER_SIZE*FEATURE_SIZE, weightsIH, true);
	printf("Weights B array: \n");
	printArray(HIDDENLAYER_SIZE*OUTPUT_SIZE, weightsHO, true);
}

int main(int argc, char* argv[]) {
	
	printf("\n");
	printf("****************\n");
	printf("** CREATING THE NEURAL NETWORK **\n");
	printf("****************\n");

	/*
	const int INPUT_SIZE = 4; // Input size
	const int HIDDENLAYER_SIZE = 4; // Output size
	const int OUTPUT_SIZE = 2; // Output size
	const int FEATURE_SIZE = 2; // Feature Size

	xor2x2NN(INPUT_SIZE,HIDDENLAYER_SIZE,OUTPUT_SIZE,FEATURE_SIZE);
	*/

	const int INPUT_SIZE = 52; // Input size
	const int HIDDENLAYER_SIZE = 10; // Output size
	const int OUTPUT_SIZE = 52; // Output size
	const int FEATURE_SIZE = 10201; // Feature Size

	float *input = new float[INPUT_SIZE*FEATURE_SIZE];
	float *hidden = new float[INPUT_SIZE*HIDDENLAYER_SIZE];
	float *output = new float[INPUT_SIZE*OUTPUT_SIZE];
	float *weightsIH = new float[HIDDENLAYER_SIZE*FEATURE_SIZE];
	float *weightsHO = new float[HIDDENLAYER_SIZE*OUTPUT_SIZE];
	float *outputNN = new float[INPUT_SIZE*OUTPUT_SIZE];

	for (int i = 1; i <= INPUT_SIZE; i++) {
		string line;
		string filename;
		string temp;
		if (i < 10) 
			filename = "C:\\Users\\somanshu\\Desktop\\Project2-Number-Algorithms\\Project2-Character-Recognition\\data-set\\0" + std::to_string(i) + "info.txt";
		else
			filename = "C:\\Users\\somanshu\\Desktop\\Project2-Number-Algorithms\\Project2-Character-Recognition\\data-set\\" + std::to_string(i) + "info.txt";
		ifstream myfile(filename);
		if (myfile.is_open())
		{
			int count = 0;
			while (getline(myfile, line))
			{
				if (count == 2) {
					std::size_t foundFirst = 1;
					std::size_t foundLast = 1;
					int counter = 0;
					while (foundLast < line.length()) {
						foundLast = line.find(" ", foundFirst);
						temp = line.substr(foundFirst, foundLast - foundFirst);
						float val = (float) std::stoi(temp);
						//float val2 = (float)val;
						input[(i - 1)*FEATURE_SIZE + counter] = val/255;
						counter++;
						foundFirst = foundLast+1;
					}
				}
					
				count++;
			}
			myfile.close();
		}
	}


	
	/*for (int i = 0; i < 10; i++) {
		printf("Element is: %0.2f\n", input[i]);
	}
	printf("last Elements \n");
	for (int i = (10201*52)-10; i < 10201*52; i++) {
		printf("Element is: %0.2f\n", input[i]);
	}
	*/
	genArrayA(FEATURE_SIZE*HIDDENLAYER_SIZE, weightsIH, 0);
	genArrayB(HIDDENLAYER_SIZE*OUTPUT_SIZE, weightsHO, 1);

	zeroArray(INPUT_SIZE*OUTPUT_SIZE, output);
	for (int i = 0; i < OUTPUT_SIZE; i++)
		output[i*INPUT_SIZE+i] = 1.0;
	//float *inputNorm = new float[INPUT_SIZE*FEATURE_SIZE];
	CharacterRecognition::createAndTrainNN(INPUT_SIZE, HIDDENLAYER_SIZE, OUTPUT_SIZE, FEATURE_SIZE, input, hidden, outputNN, weightsIH, weightsHO, output);
	/*
	printf("Output Array: \n");
	printArray(INPUT_SIZE*OUTPUT_SIZE, outputNN, true);
	printf("Actual Output Array: \n");
	printArray(INPUT_SIZE*OUTPUT_SIZE, output, true);
	*/
}
