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
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>

#define TRAINING 0
#define CHAR_RECOGNITION 1

int main(int argc, char* argv[]) {

#if CHAR_RECOGNITION
	printf("\n");
	printf("*********************************\n");
	printf("** CHARACTER RECOGNITION TESTS **\n");
	printf("*********************************\n");

	int numTotalInput = 52;
	std::vector<float*> inputs;
	std::vector<float> expected;

	// collect character recognition data set
	int numInput;

	std::string prefix = "../data-set/";
	std::string suffix = "info.txt";
	for (int i = 1; i < numTotalInput + 1; i++) {
		std::stringstream buffer;
		buffer << prefix << std::setfill('0') << std::setw(2) << i << suffix;

		std::ifstream inputFile(buffer.str());
		if (inputFile.is_open()) {
			std::string line;
			// get expected character
			std::getline(inputFile, line);
			expected.push_back(atoi(line.c_str()) / 255.0);

			// get number of characters in input
			std::getline(inputFile, line);
			numInput = atoi(line.c_str());

			// get input
			std::getline(inputFile, line);

			std::istringstream iss(line);
			std::string oneInput;
			int counter = 0;
			float *input = new float[numInput];
			while (std::getline(iss, oneInput, ' ')) {
				if (!oneInput.empty()) {
					input[counter] = stof(oneInput) / 255.0;
					counter++;
				}
			}
			inputs.push_back(input);
		}
	}

#if TRAINING
	CharacterRecognition::train(numInput, inputs, expected, "char_recognition");
#else // #if TRAINING
	CharacterRecognition::run(numInput, inputs, expected, "char_recognition");
#endif // #else // #if TRAINING

	for (int i = 0; i < numTotalInput; i++) {
		delete[] inputs[i];
	}

#else // #if CHAR_RECOGNITION
	printf("\n");
	printf("*********************************\n");
	printf("*********** XOR TESTS ***********\n");
	printf("*********************************\n");

	int numTotalInput = 4;

	std::vector<float*> inputs;
	std::vector<float> expected;
	const int numInput = 2;

	float input1[numInput] = { 0, 0 };
	float input2[numInput] = { 0, 1 };
	float input3[numInput] = { 1, 0 };
	float input4[numInput] = { 1, 1 };
	inputs.push_back(input1);
	inputs.push_back(input2);
	inputs.push_back(input3);
	inputs.push_back(input4);

	expected.push_back(0);
	expected.push_back(1);
	expected.push_back(1);
	expected.push_back(0);

#if TRAINING
	CharacterRecognition::train(numInput, inputs, expected, "XOR");
#else // #if TRAINING
	CharacterRecognition::run(numInput, inputs, expected, "XOR");
#endif // #else // #if TRAINING

#endif // #else // #if CHAR_RECOGNITION

    system("pause"); // stop Win32 console from closing on exit
}
