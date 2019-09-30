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

	std::string  file_name;
	for (int i = 1; i <= num_instances; i++) {
		std::stringstream ss;
		ss << setw(2) << std::setfill('0') << i;
		file_name = "C:\\Users\\djjindal\\Project2-Number-Algorithms\\Project2-Character-Recognition\\data-set\\" + ss.str() + "info.txt";
		ifstream infile(file_name);
		string line;
		if (infile.is_open()) {
			int line_num = 0;
			while (getline(infile, line)) {
				std::string::size_type sz;
				if (line_num == 0) {
					float label = std::stof(line, &sz);
					int idx = output_layer * (i - 1) + (label - 1);
					ilabel[idx] = 1;				
				}
				if (line_num == 2) {
					stringstream ssin(line);
					int k = 0;
					while (ssin.good() && k < input_layer) {
						std::string temp;
						ssin >> temp;
						idata[input_layer*(i - 1) + k] = stof(temp, &sz) / 255.0;
						++k;
					}
				}
				line_num++;
			}
			infile.close();
		}
	}
}

void loadXORDataset(float* idata, float* ilabel) {
	idata[0] = 0; idata[1] = 0;	ilabel[0] = 1; ilabel[1] = 0;
	idata[2] = 1; idata[3] = 0; ilabel[2] = 0; ilabel[3] = 1;
	idata[4] = 0; idata[5] = 1; ilabel[4] = 0; ilabel[5] = 1;
	idata[6] = 1; idata[7] = 1; ilabel[6] = 1; ilabel[7] = 0;	
}

void loadXORTestDataset(float* idata) {
	idata[0] = 0;
	idata[1] = 1;
}

/*
	Driver Function for XOR Model
*/
void XORModel() {
	// Model Architecture and Num Instances
	int input_layer = 2;
	int hidden_layer = 4;
	int output_layer = 2;
	int num_instances = 4;

	// Initialize Network
	CharacterRecognition::init(input_layer, hidden_layer, output_layer);

	// Load Dataset 
	float *idata = new float[8];
	float *ilabel = new float[8];
	loadXORDataset(idata, ilabel);

	// Parameters
	int epochs = 2000;
	float learning_rate = 0.1f;

	// Train
	CharacterRecognition::train(idata, ilabel, num_instances, epochs, learning_rate, "xor_model");

	// Test
	CharacterRecognition::test(idata, ilabel, num_instances);

	// Free Resources
	CharacterRecognition::free();
}
/*
	Driver Function for Character Recognition Model
*/
void CharacterRecognitionModel(int hidden_layer, float learning_rate = 0.1f) {
	// Model Architecture and Num Instances
	int input_layer = 10201;
	int output_layer = 52;
	int num_instances = 52;

	// Initialize Network
	CharacterRecognition::init(input_layer, hidden_layer, output_layer);

	// Load Dataset 
	float *idata = new float[num_instances * input_layer];
	float *ilabel = new float[num_instances * output_layer];
	loadCharacterDataset(num_instances, input_layer, output_layer, idata, ilabel);

	// Parameters
	int epochs = 500;

	// Train
	CharacterRecognition::train(idata, ilabel, num_instances, epochs, learning_rate, "cr_model");

	// Test
	CharacterRecognition::test(idata, ilabel, num_instances);

	// Free Resources
	CharacterRecognition::free();
}

int main(int argc, char* argv[]) {
	//XORModel();
	CharacterRecognitionModel(20, 0.1);
}
