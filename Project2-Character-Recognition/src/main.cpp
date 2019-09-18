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



// Define run presets
#define training 0 // If set to 1, indicates we are in training mode
#define acceptedError 0.001
#define numRandomWeightAttempts 10
#define numConvergeIterations 1000
#define useXor 1


int main(int argc, char* argv[]) {
	// Setup input data depending on xor or character data
	int inputSize = 10201;
	int numInputs = 52;
	if (useXor) {
		inputSize = 2;
		numInputs = 4;
	}

	// Fill in all data:
	std::vector<float*> inputData = std::vector<float*>();
	std::vector<float> expectedData = std::vector<float>();

	if (useXor) {
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

		inputData.push_back(data1);
		inputData.push_back(data2);
		inputData.push_back(data3);
		inputData.push_back(data4);
		expectedData.push_back(0);
		expectedData.push_back(1);
		expectedData.push_back(1);
		expectedData.push_back(0);
	}

	else {
		// Read in character training data:
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
				inputData.push_back(currData);
				expectedData.push_back(expectedVal);
			}
			inputFile.close();
		}
	}

	
	if (training) {
		// Setup weights arrays:
		int numHiddenLayers = ceil((inputSize + 1) / 2.0);

		int layer1_numWeights = inputSize * numHiddenLayers;
		int layer2_numWeights = numHiddenLayers;

		float *layer1_weights = new float[layer1_numWeights];
		float *layer2_weights = new float[layer2_numWeights];
		

		if (useXor) {
			layer1_weights[0] = 10.1;
			layer1_weights[1] = 0.9;
			layer1_weights[2] = 20;
			layer1_weights[3] = 0.87;

			layer2_weights[0] = 41;
			layer2_weights[1] = -54;
		}

		std::vector<float*> partials1 = std::vector<float*>();
		std::vector<float*> partials2 = std::vector<float*>();

		for (int i = 0; i < numInputs; ++i) {
			partials1.push_back(new float[layer1_numWeights]);
			partials2.push_back(new float[layer2_numWeights]);
		}

		// Begin loop iteration
		auto start = std::chrono::steady_clock::now();
		int numRandIters = 0;
		float accumulatedError = 3.0; // Larger than accepted error
		bool done = false;
		int numRandAttempts = useXor ? 1 : numRandomWeightAttempts;
		while (!done && numRandIters < numRandAttempts) {
			// Fill new random weights (if xor, use preset weights)
			if (!useXor) {
				auto end1 = std::chrono::steady_clock::now();
				CharacterRecognition::fillRandomWeights(layer1_numWeights, layer1_weights, std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - start).count());
				auto end2 = std::chrono::steady_clock::now();
				CharacterRecognition::fillRandomWeights(layer2_numWeights, layer2_weights, std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start).count());
				std::cout << "NEW WEIGHTS" << std::endl;
			}
			
			int numInnerIters = 0.0;
			// Try refining weights iteratively
			while (!done && numInnerIters < numConvergeIterations) {
				accumulatedError = 0.0;
				bool resultAll1 = true;

				// Run each input through mlp, accumulating error
				for (int k = 0; k < numInputs; ++k) {
					float currExpected = expectedData.at(k);
					float output = CharacterRecognition::mlp(inputSize, numHiddenLayers, currExpected, 
						layer1_weights, layer2_weights, inputData.at(k), partials1.at(k), partials2.at(k));
					if (output != 1) {
						resultAll1 = false;
					}

					float currError = (output - currExpected) * (output - currExpected);
					std::cout << "expected output: " << currExpected << "  Result: " << output << std::endl;
					accumulatedError += currError;
				}
				if (resultAll1) {
					break;
				}
				accumulatedError /= 2.0;
				std::cout << "Accumulated error: " << accumulatedError << std::endl;
				if (accumulatedError < acceptedError) {
					done = true;
				}
				// Adjust weights based on accumulated error
				if (!done) {
					for (int k = 0; k < numInputs; ++k) {
						float* partialValues1 = partials1.at(k);
						float* partialValues2 = partials2.at(k);

						CharacterRecognition::updateWeights(layer1_numWeights,
							accumulatedError, partialValues1, layer1_weights);

						CharacterRecognition::updateWeights(layer2_numWeights,
							accumulatedError, partialValues2, layer2_weights);
					}
				}
				if (done) {
					std::cout << "DONE" << std::endl;
				}
				numInnerIters++;
			}
			numRandIters++;
		}
		// Print out final weights and error after either converging to good weights or failing to converge
		std::cout << "FINAL ERROR: " << accumulatedError << std::endl;
		std::cout << "WEIGHTS:" << std::endl;
		for (int i = 0; i < layer1_numWeights; ++i) {
			std::cout << "layer 1 weight " << i << ": " << layer1_weights[i] << std::endl;
		}
		for (int i = 0; i < layer2_numWeights; ++i) {
			std::cout << "layer 2 weight " << i << ": " << layer2_weights[i] << std::endl;
		}

		// Delete data arrays stored in map
		for (float* i : inputData) {
			delete[] i;
		}
		for (float* i : partials1) {
			delete[] i;
		}
		for (float* i : partials2) {
			delete[] i;
		}
		delete[] layer1_weights;
		delete[] layer2_weights;
	}
	else {
		// Setup weights arrays:
		int numHiddenLayers = ceil((inputSize + 1) / 2.0);

		int layer1_numWeights = inputSize * numHiddenLayers;
		int layer2_numWeights = numHiddenLayers;

		float *layer1_weights = new float[layer1_numWeights];
		float *layer2_weights = new float[layer2_numWeights];


		if (useXor) {
			layer1_weights[0] = 10.1;
			layer1_weights[1] = 0.9;
			layer1_weights[2] = 20;
			layer1_weights[3] = 0.87;

			layer2_weights[0] = 41;
			layer2_weights[1] = -54;
		}

		auto start = std::chrono::steady_clock::now();
		// if not xor, use random weights because never found good weights
		if (!useXor) {
			auto end1 = std::chrono::steady_clock::now();
			CharacterRecognition::fillRandomWeights(layer1_numWeights, layer1_weights, std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - start).count());
			auto end2 = std::chrono::steady_clock::now();
			CharacterRecognition::fillRandomWeights(layer2_numWeights, layer2_weights, std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start).count());
			std::cout << "NEW WEIGHTS" << std::endl;
		}

		// Run input data through mlp to get output with given weights
		for (int k = 0; k < numInputs; ++k) {
			float currExpected = expectedData.at(k);
			float output = CharacterRecognition::mlpNoError(inputSize, numHiddenLayers, currExpected,
				layer1_weights, layer2_weights, inputData.at(k));
			// Print out results, rounding output to integer to match input type
			std::cout << "expected output: " << currExpected << "  Result: " << round(output) << std::endl;
		}
	
		// Delete data arrays stored in map
		for (float* i : inputData) {
			delete[] i;
		}
		delete[] layer1_weights;
		delete[] layer2_weights;
	}
	
}
