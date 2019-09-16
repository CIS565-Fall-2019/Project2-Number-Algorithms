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
#include <map>


 // Define mode to training or running
 // Define acceptedError 0.0001
#define training 1 // If set to 1, indicates we are in training mode
#define acceptedError 0.001
#define inputSize 2
#define numInputs 4


int main(int argc, char* argv[]) {
	
	if (training) {
		// Fill in all data:
		//std::vector<int*> inputArrays = std::vector<int*>();
		std::map<float*, float> inputOutputMap = std::map<float*, float>();
		float *data1 = new float[2];
		float *data2 = new float[2];
		float *data3 = new float[2];
		float *data4 = new float[2];

		data1[0] = 0;
		data1[1] = 0;
		data2[0] = 0;
		data3[1] = 0;
		data2[1] = 1;
		data3[0] = 1;
		data4[0] = 1;
		data4[1] = 1;

		inputOutputMap.insert(std::pair<float*, float>(data1, 0));
		inputOutputMap.insert(std::pair<float*, float>(data2, 1));
		inputOutputMap.insert(std::pair<float*, float>(data3, 1));
		inputOutputMap.insert(std::pair<float*, float>(data4, 0));




		/*for (std::pair<int*, int> i : inputOutputMap) {
			std::cout << "(" << i.first[0] << ", " << i.first[1] << "): " << i.second << std::endl;
		}*/

		// Setup weights arrays:
		int numHiddenLayers = ceil((inputSize + 1) / 2.0);

		int layer1_numWeights = inputSize * numHiddenLayers;
		int layer2_numWeights = numHiddenLayers;

		float *layer1_weights = new float[layer1_numWeights];
		float *layer2_weights = new float[layer2_numWeights];

		float *layer1_adjustedWeights = new float[layer1_numWeights];
		float *layer2_adjustedWeights = new float[layer2_numWeights];

		CharacterRecognition::fillRandomWeights(layer1_numWeights, layer1_weights);
		CharacterRecognition::fillRandomWeights(layer2_numWeights, layer2_weights);

		layer1_weights[0] = 10.1;
		layer1_weights[1] = 0.9;
		layer1_weights[2] = 20;
		layer1_weights[3] = 0.87;

		layer2_weights[0] = 41;
		layer2_weights[1] = -54;

		for (int i = 0; i < layer1_numWeights; ++i) {
			std::cout << layer1_weights[i] << std::endl;
		}
		for (int i = 0; i < layer2_numWeights; ++i) {
			std::cout << layer2_weights[i] << std::endl;
		}

		float accumulatedError = 3.0;
		while (accumulatedError > acceptedError) {
			accumulatedError = 0.0;
			for (std::pair<float*, float> i : inputOutputMap) {
				float currExpected = i.second;
				std::cout << i.first[0] << ", " << i.first[1] << std::endl;

				float output = CharacterRecognition::mlp(inputSize, numHiddenLayers, currExpected, layer1_weights, layer2_weights, i.first, layer1_adjustedWeights, layer2_adjustedWeights);
				float currError = (output - currExpected) * (output - currExpected);
				accumulatedError += currError;
			}
			accumulatedError /= 2.0;
			for (int i = 0; i < layer1_numWeights; ++i) {
				layer1_weights[i] = layer1_adjustedWeights[i];
			}
			for (int i = 0; i < layer2_numWeights; ++i) {
				layer2_weights[i] = layer2_adjustedWeights[i];
			}
			accumulatedError = 0.0;
		}

		/*std::cout << "layer1 weights:" << std::endl;
		for (int i = 0; i < layer1_numWeights; ++i) {
			std::cout << layer1_weights[i] << std::endl;
		}
		std::cout << "layer2 weights:" << std::endl;
		for (int i = 0; i < layer2_numWeights; ++i) {
			std::cout << layer2_weights[i] << std::endl;
		}*/


	}
	/*
	// if training	

	randomize weights (kernel function)
	for 1 through 52, get the text numbers values into an array:
	Read file data into an array of arrays for all inputs

	for every array in array of inputs, pass in to mlp as the input data
	Accumulate all the error values

	while error is not within accepted error range.  

		adjust the weights (adjust weights)
		rerun mlp
		get new error

	keep final weights
	print out output weights, or write out to a txt file

	if not training
		read weights from text file
		read the input from the text file
		run the mlp just to get the final result, no error calculation
	*/

}
