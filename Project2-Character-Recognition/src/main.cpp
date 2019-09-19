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
#include <vector>


const int SIZE = 1 << 8; // feel free to change the size of array
const int NPOT = SIZE - 3; // Non-Power-Of-Two
int *a = new int[SIZE];
int *b = new int[SIZE];
int *c = new int[SIZE];

using namespace std;

void loadTrainingData(const string& dir, vector<vector<double> >& input, vector<vector<double> >& output)
{
	const int trainingSize = 52;	// How many samples to train
	const int numVals = 10201;

	input.resize(trainingSize); // 52 x 10201
	output.resize(trainingSize); // 52 x 52

	for (size_t i = 1; i <= trainingSize; i++) {
		string filename = to_string(i) + "info.txt";
		if (i < 10) {
			filename = "0" + filename;
		}
		string filePath = dir + filename;

		ifstream file(filePath);

		if (file.is_open()) {
			int tmp;
			for (size_t c = 0; c < 2; c++) file >> tmp;
			input[i-1].resize(numVals);
			for (size_t c = 0; c < numVals; c++) {
				file >> input[i-1][c];
			}
			file.close();
		}

		output[i - 1].resize(trainingSize, 0);
		output[i - 1][i - 1] = 1;
	}

}

int main(int argc, char* argv[]) {

	const string dir = "..\\data-set\\";
	vector<vector<double>> input;
	vector<vector<double>> output;

//	loadTrainingData(dir, input, output);

	// compute output
	
	CharacterRecognition::init(32, 8, 1, 0.5);


    system("pause"); // stop Win32 console from closing on exit
	delete[] a;
	delete[] b;
	delete[] c;
}
