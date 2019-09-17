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

const int SIZE = 1 << 8; // feel free to change the size of array
const int NPOT = SIZE - 3; // Non-Power-Of-Two
int *a = new int[SIZE];
int *b = new int[SIZE];
int *c = new int[SIZE];

#define training 1

int main(int argc, char* argv[]) {

	printf("\n");
	printf("*********************************\n");
	printf("*********** XOR TESTS ***********\n");
	printf("*********************************\n");
	
	int numTotalInput = 4;

	std::vector<float*> inputs;
	std::vector<int> expected;
	const int numInputXOR = 2;

	float input1[numInputXOR] = { 0, 0 };
	float input2[numInputXOR] = { 0, 1 };
	float input3[numInputXOR] = { 1, 0 };
	float input4[numInputXOR] = { 1, 1 };
	inputs.push_back(input1);
	inputs.push_back(input2);
	inputs.push_back(input3);
	inputs.push_back(input4);

	expected.push_back(0);
	expected.push_back(1);
	expected.push_back(1);
	expected.push_back(0);

	CharacterRecognition::train(numInputXOR, inputs, expected);


    printf("\n");
    printf("*********************************\n");
    printf("** CHARACTER RECOGNITION TESTS **\n");
    printf("*********************************\n");

	inputs.clear();
	expected.clear();

	// collect character recognition data set
	numTotalInput = 52;
	int numInputCharRecognition;

	std::string prefix = "../data-set/";
	std::string suffix = "info.txt";
	for (int i = 1; i < numTotalInput; i++) {
		std::stringstream buffer;
		buffer << prefix << std::setfill('0') << std::setw(2) << i << suffix;

		std::ifstream inputFile(buffer.str());
		if (inputFile.is_open()) {
			std::string line;
			std::getline(inputFile, line);
			expected.push_back(atoi(line.c_str()));

			std::getline(inputFile, line);
			numInputCharRecognition = atoi(line.c_str());

			std::getline(inputFile, line);
			std::string::iterator end = std::remove(line.begin(), line.end(), ' ');
			line.erase(end, line.end());

			float *input = new float[numInputCharRecognition];
			int length = line.size();
			for (int j = 0; j < numInputCharRecognition; j++) {
				input[j] = line.at(j);
			}
			inputs.push_back(input);
		}
	}

	CharacterRecognition::train(numInputCharRecognition, inputs, expected);

	for (int i = 0; i < numTotalInput; i++) {
		delete[] inputs[i];
	}

    system("pause"); // stop Win32 console from closing on exit
}
