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
#include <time.h>
#include <chrono>
#include <string>
#include <fstream>
#include <sstream>



 // Define mode to training or running
 // Define acceptedError 0.0001
#define training 1 // If set to 1, indicates we are in training mode
#define acceptedError 0.01
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

		// Read in training data:
		//for (int i = 1; i <= numInputs; ++i) {
			/*std::string filename = std::to_string(i) + "info.txt";
			if (i < 10) {
				filename = "0" + filename;
			}
			filename = "../data-set/" + filename;

			std::ifstream inputFile;
			inputFile.open(filename);
			if (inputFile.is_open()) {
				std::string firstLine;
				getline(inputFile, firstLine);
				int expectedVal = std::stoi(firstLine);

				std::string secondLine;
				getline(inputFile, secondLine);
				int inputLength = std::stoi(secondLine);

				float *currData = new float[inputLength];
				int counter = 0;

				std::string dataLine;
				getline(inputFile, dataLine);

				std::stringstream stream(dataLine);
				while (1) {
					int n;
					stream >> n;
					if (!stream) {
						break;
					}
					currData[counter] = n;
					counter++;
				}
				inputOutputMap.insert(std::pair<float*, float>(currData, expectedVal));
			}
			inputFile.close();
		}*/
	


		/*for (std::pair<float*, float> i : inputOutputMap) {
			std::cout << "(" << i.first[4] << ", " << i.first[5] << "): " << i.second << std::endl;
		}*/


		// Setup weights arrays:
		int numHiddenLayers = ceil((inputSize + 1) / 2.0);

		int layer1_numWeights = inputSize * numHiddenLayers;
		int layer2_numWeights = numHiddenLayers;

		float *layer1_weights = new float[layer1_numWeights];
		float *layer2_weights = new float[layer2_numWeights];

		float *layer1_adjustedWeights = new float[layer1_numWeights];
		float *layer2_adjustedWeights = new float[layer2_numWeights];

		
		layer1_weights[0] = 10.1;
		layer1_weights[1] = 0.9;
		layer1_weights[2] = 20;
		layer1_weights[3] = 0.87;

		layer2_weights[0] = 41;
		layer2_weights[1] = -54;

		float *partialDerivatives1 = new float[layer1_numWeights * numInputs];
		float *partialDerivatives2 = new float[layer2_numWeights * numInputs];

		auto start = std::chrono::steady_clock::now();
		int numRandIters = 0;
		float accumulatedError = 3.0; // Larger than accepted error
		while (accumulatedError > acceptedError && numRandIters < 1) {
			// Fill new random weights
			/*auto end1 = std::chrono::steady_clock::now();
			CharacterRecognition::fillRandomWeights(layer1_numWeights, layer1_weights, std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - start).count());
			auto end2 = std::chrono::steady_clock::now();
			CharacterRecognition::fillRandomWeights(layer2_numWeights, layer2_weights, std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start).count());
			std::cout << "NEW WEIGHTS" << std::endl;*/


			// Copy new weights to adjusted weights array
			for (int i = 0; i < layer1_numWeights; ++i) {
				layer1_adjustedWeights[i] = layer1_weights[i];
			}
			for (int i = 0; i < layer2_numWeights; ++i) {
				layer2_adjustedWeights[i] = layer2_weights[i];
			}
			
			int numInnerIters = 0.0;
			// Try refining weights iteratively
			while (accumulatedError > acceptedError && numInnerIters < 1) {
				accumulatedError = 0.0;

				for (std::pair<float*, float> i : inputOutputMap) {
					float currExpected = i.second;
					float output = CharacterRecognition::mlp(inputSize, numHiddenLayers, currExpected, layer1_weights, layer2_weights, i.first, layer1_adjustedWeights, layer2_adjustedWeights, partialDerivatives1, partialDerivatives2);


					float currError = (output - currExpected) * (output - currExpected);
					std::cout << "expected output: " << currExpected << "  Result: " << output << std::endl;
					accumulatedError += currError;
				}
				accumulatedError /= 2.0;
				// If error is low enough, print out weights and break
				if (accumulatedError < acceptedError) {
					/*std::cout << "WEIGHTS:" << std::endl;
					for (int i = 0; i < layer1_numWeights; ++i) {
						std::cout << "layer 1 weight " << i << ": " << layer1_weights[i] << std::endl;
					}
					for (int i = 0; i < layer2_numWeights; ++i) {
						std::cout << "layer 2 weight " << i << ": " << layer2_weights[i] << std::endl;
					}*/
					//break;
				}

				// Copy adjusted weights into actual weights for next iteration
				for (int i = 0; i < layer1_numWeights; ++i) {
					layer1_weights[i] = layer1_adjustedWeights[i];
					std::cout << layer1_weights[i] << std::endl;
				}
				for (int i = 0; i < layer2_numWeights; ++i) {
					layer2_weights[i] = layer2_adjustedWeights[i];
					std::cout << layer2_weights[i] << std::endl;
				}
				numInnerIters++;
			}
			// If error is low enough, print weights and break
			if (accumulatedError < acceptedError) {
				/*std::cout << "WEIGHTS:" << std::endl;
				for (int i = 0; i < layer1_numWeights; ++i) {
					std::cout << "layer 1 weight " << i << ": " << layer1_weights[i] << std::endl;
				}
				for (int i = 0; i < layer2_numWeights; ++i) {
					std::cout << "layer 2 weight " << i << ": " << layer2_weights[i] << std::endl;
				}*/
				break;
			}

			numRandIters++;
		}

		std::cout << "FINAL ERROR: " << accumulatedError << std::endl;

		// Delete data arrays stored in map
		for (std::pair<float*, float> i : inputOutputMap) {
			delete[] i.first;
		}
		delete[] partialDerivatives;
	}
	else {

		std::map<float*, float> inputOutputMap = std::map<float*, float>();
			// Read in data:
		for (int i = 1; i <= numInputs; ++i) {
			std::string filename = std::to_string(i) + "info.txt";
			if (i < 10) {
				filename = "0" + filename;
			}
			filename = "../data-set/" + filename;

			std::ifstream inputFile;
			inputFile.open(filename);
			if (inputFile.is_open()) {
				std::string firstLine;
				getline(inputFile, firstLine);
				int expectedVal = std::stoi(firstLine);

				std::string secondLine;
				getline(inputFile, secondLine);
				int inputLength = std::stoi(secondLine);

				float *currData = new float[inputLength];
				int counter = 0;

				std::string dataLine;
				getline(inputFile, dataLine);

				std::stringstream stream(dataLine);
				while (1) {
					int n;
					stream >> n;
					if (!stream) {
						break;
					}
					currData[counter] = n;
					counter++;
				}
				inputOutputMap.insert(std::pair<float*, float>(currData, expectedVal));
			}
			inputFile.close();
		}
		// Setup weights arrays:
		int numHiddenLayers = ceil((inputSize + 1) / 2.0);

		int layer1_numWeights = inputSize * numHiddenLayers;
		int layer2_numWeights = numHiddenLayers;

		float *layer1_weights = new float[layer1_numWeights];
		float *layer2_weights = new float[layer2_numWeights];

		float *layer1_adjustedWeights = new float[layer1_numWeights];
		float *layer2_adjustedWeights = new float[layer2_numWeights];

		auto start = std::chrono::steady_clock::now();

		// Fill weights
		auto end1 = std::chrono::steady_clock::now();
		CharacterRecognition::fillRandomWeights(layer1_numWeights, layer1_weights, std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - start).count());
		auto end2 = std::chrono::steady_clock::now();
		CharacterRecognition::fillRandomWeights(layer2_numWeights, layer2_weights, std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start).count());

		for (std::pair<float*, float> i : inputOutputMap) {
			float currExpected = i.second;
			float output = CharacterRecognition::mlp(inputSize, numHiddenLayers, currExpected, layer1_weights, layer2_weights, i.first, layer1_adjustedWeights, layer2_adjustedWeights);
			std::cout << "expected output: " << currExpected << "  Result: " << output << std::endl;
		}

		// Delete data arrays stored in map
		for (std::pair<float*, float> i : inputOutputMap) {
			delete[] i.first;
		}
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
