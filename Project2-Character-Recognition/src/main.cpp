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


const int INPUT_N = 10201;
const int trainingSize = 52;	// How many samples to train

using namespace std;

void loadTrainingData(const string& dir, vector<vector<float> >& input, vector<vector<float> >& output)
{

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
			input[i-1].resize(INPUT_N);
			for (size_t c = 0; c < INPUT_N; c++) {
				file >> input[i-1][c];
			}
			file.close();
		}
		output[i - 1].resize(trainingSize, 0);
		output[i - 1][i - 1] = 1;
	}

	// Normalization
	for (size_t i = 0; i < trainingSize; i++) {
		float mean = 0.f;
		float variance = 0;
		for (int j = 0; j < INPUT_N; j++) {
			mean += input[i][j];
		}
		mean /= INPUT_N;
		for (int j = 0; j < INPUT_N; j++) {
			variance += (input[i][j] - mean) * (input[i][j] - mean);
		}
		float stdv = std::sqrt(variance / INPUT_N);
		for (int j = 0; j < INPUT_N; j++) {
			input[i][j] = (input[i][j]) / (stdv + 0.00001f);
		}
	}
}

int main(int argc, char* argv[]) {


	//CharacterRecognition::unitTest();
	//system("pause"); // stop Win32 console from closing on exit
	//return 1;


	const string dir = "..\\data-set\\";
	vector<vector<float>> input;
	vector<vector<float>> output;

	const int hidden_N = 64;
	loadTrainingData(dir, input, output);
	CharacterRecognition::init(INPUT_N, hidden_N, trainingSize, 0.1f);

	// compute output
	std::vector<float> inputArr = { 1, 2 };

	// train on 10 iterations
	for (int i = 0; i < 40; i++)
	{
		float cost;
		for (int j = 0; j < input.size(); j++) // train all 52 samples
		{
			CharacterRecognition::Matrix* m = CharacterRecognition::computeOutput(input[j]);
			cost = CharacterRecognition::learn(output[j]);
		}
		cout << "#" << i + 1 << "/10  Cost: " << cost <<  endl;
	}


	// test
	cout << "expected output : actual output" << endl;
	for (int i = 0; i < input.size(); i++) // testing on last 10 examples
	{
		for (int j = 0; j < trainingSize; j++)
		{
			cout << output[i][j] << " ";
		}
		cout << endl;

		CharacterRecognition::Matrix* result = CharacterRecognition::computeOutput(input[i]);
		result->copyToHost();
		result->print();
	}


    system("pause"); // stop Win32 console from closing on exit
}
