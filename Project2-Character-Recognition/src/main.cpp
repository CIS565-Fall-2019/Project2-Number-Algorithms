/**
 * @file      main.cpp
 * @brief     Character Recognition
 * @authors   Disha Jindal
 * @date      2019
 * @copyright University of Pennsylvania
 */
#include <iostream>
#include <cstdio>
#include <character_recognition/mlp.h>
#include <character_recognition/common.h>
#include "testing_helpers.hpp"
#include <fstream>
#include <iostream>
#include <string.h>
#include <sstream>
#include <iomanip>
#include <fstream>

using namespace std;

void loadCharacterDataset(int num_instances, int input_layer, int output_layer, float* idata, float* ilabel) {
	for (int i = 0; i < num_instances * output_layer; i++)
		ilabel[i] = 0;

	for (int i = 1; i <= num_instances; i++) {
		ifstream infile;
		std::string  file_name;
		std::stringstream ss;
		ss << setw(2) << std::setfill('0') << i;
		file_name = "C:\\Users\\djjindal\\Project2-Number-Algorithms\\Project2-Character-Recognition\\data-set\\" + ss.str() + "info.txt";
		infile.open(file_name);
		int line_num = 0;
		string line;
		std::string::size_type sz;
		while (getline(infile, line)){
			if (line_num == 0) {
				ilabel[output_layer*(i - 1) + stoi(line) - 1] = 1;
			}
			if (line_num == 2) {
				stringstream ssin(line);
				int k = 0;
				while (ssin.good() && k < input_layer) {
					std::string temp;
					ssin >> temp;
					idata[output_layer*(i - 1) + k] = stof(temp, &sz) / 255;
					++k;
				}
			}
			line_num++;
		}
		infile.close();
	}
}

void loadXORDataset(float* idata, float* ilabel) {
	idata[0] = 0;
	idata[1] = 0;
	idata[2] = 1;
	idata[3] = 0;
	idata[4] = 0;
	idata[5] = 1;
	idata[6] = 1;
	idata[7] = 1;

	ilabel[0] = 1;
	ilabel[1] = 0;
	ilabel[2] = 0;
	ilabel[3] = 1;
	ilabel[4] = 0;
	ilabel[5] = 1;
	ilabel[6] = 1;
	ilabel[7] = 0;
}

int main(int argc, char* argv[]) {
	// Model Architecture and Num Instances
	int input_layer = 10201;
	int hidden_layer = 50;
	int output_layer = 52;
	int num_instances = 52;

	//int input_layer = 2;
	//int hidden_layer = 4;
	//int output_layer = 2;
	//int num_instances = 4;

	// Initialize Network
	CharacterRecognition::init(input_layer, hidden_layer, output_layer);

	// Load Dataset 
	float *idata = new float[num_instances * 10201];
	float *ilabel = new float[num_instances * output_layer];
	loadCharacterDataset(num_instances, input_layer, output_layer, idata, ilabel);

	//float *idata = new float[8];
	//float *ilabel = new float[8];
	//loadXORDataset(idata, ilabel);

	// Parameters
	int epochs = 2000;
	float learning_rate = 0.1f;

	// Train
	CharacterRecognition::train(idata, ilabel, num_instances, epochs, learning_rate);


	// Test

	// Free Resources
	CharacterRecognition::free();
	
}
