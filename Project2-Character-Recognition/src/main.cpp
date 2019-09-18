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
#include <string>
using namespace std;

const float alpha = 0.1;
const int batchSize = 52;

void GotoLine(std::ifstream& file, unsigned int num) {
	file.seekg(std::ios::beg);
	for (int i = 0; i < num - 1; ++i) {
		file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
	}
	return;
}

int main(int argc, char* argv[]) {
	
	float *data = new float[10201 * 52];
	float *label = new float[52 * 52];
	
	for (int i = 1; i < 53; i++) {
		string filename = i + "info.txt";
		ifstream file(filename);
		string line;
		if (file.is_open())
		{
			GotoLine(file, 3);
			for (int j = 0; j < 10201; j++) {
				file >> data[i - 1 + j * 52];
			}
			file.close();
		}
	}
	for (int i = 0; i < 52; i++) {
		for (int j = 0; j < 52; j++) {
			if(i != j)
				label[i * 52 + j] = 0;
			else 
				label[i * 52 + j] = 1;
		}
	}

	//normalization
	for (int i = 0; i < 52; i++) {
		float avg = 0;
		float sigma = 0;
		for (int j = 0; j < 10201; j++) {
			avg += data[i + j * 52];
		}
		avg /= 10201.0f;
		for (int j = 0; j < 10201; j++) {
			sigma += (data[i + j * 52] - avg) * (data[i + j * 52] - avg);
		}
		sigma = std::sqrt(sigma / 10201.0f);
		for (int j = 0; j < 10201; j++) {
			data[i + j * 52] = (data[i + j * 52] - avg) / (sigma + 0.000001f);
		}
	}
	
	int layerNum = 5;
	int *layerSizes = new int[layerNum];
	layerSizes[0] = 10201;
	layerSizes[1] = 30;
	layerSizes[2] = 10;
	layerSizes[3] = 30;
	layerSizes[4] = 52;

	int totalWNum = 0;
	for (int i = 1; i < layerNum; i++) {
		totalWNum += layerSizes[i] * (1 + layerSizes[i - 1]);
	}
	float *weights = new float[totalWNum];
	float *grad = new float[totalWNum];

	CharacterRecognition::initializeW(weights, layerSizes, layerNum);

	float cost;
	for (int i = 0; i < 100; i++) {
		cost = CharacterRecognition::computeCostGrad(layerSizes, layerNum, batchSize, weights, grad, data, label);
		CharacterRecognition::updateWeights(totalWNum, weights, grad, alpha);
		printf("Epoch: %d Cost: %f \n", i, cost);
	}

	delete[] layerSizes;
	delete[] data;
	delete[] label;
	delete[] weights;
	delete[] grad;
}
