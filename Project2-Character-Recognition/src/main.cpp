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
#include <vector>
#include <sstream>
#include <iomanip>

#define NUMBER_OF_INSTANCES 52
#define NUMBER_OF_FEATURES 10201
#define HIDDEN_LAYER_SIZE 50
#define NUMBER_OF_CLASSES 52

#define LEARNING_RATE 0.01
#define NUMBER_OF_EPOCHS 1000

int main(int argc, char* argv[]) {
	//float input[NUMBER_OF_INSTANCES * NUMBER_OF_FEATURES] = { 0,0, 0,1, 1,0, 1,1 };
	//float true_labels[NUMBER_OF_INSTANCES * NUMBER_OF_CLASSES] = { 1,0, 0,1 ,0,1, 1,0 };
	//float input = (float *)malloc(NUMBER_OF_INSTANCES * NUMBER_OF_FEATURES * sizeof(float));
	//float true_labels = (float *)malloc(NUMBER_OF_INSTANCES * NUMBER_OF_CLASSES * sizeof(float));
	float input[NUMBER_OF_INSTANCES * NUMBER_OF_FEATURES];
	float true_labels[NUMBER_OF_INSTANCES * NUMBER_OF_CLASSES] = {0};
	//Load data to arrays
	std::string filename;
	for (int i = 1; i <= NUMBER_OF_INSTANCES; i++) {
		std::stringstream ss;
		ss << std::setw(2) << std::setfill('0') << i;
		std::string s = ss.str();
		filename = s + "info.txt";
		
		std::ifstream f("C:\\Users\\saketk\\Project2-Number-Algorithms\\Project2-Character-Recognition\\data-set\\" + filename);
		std::string line;
		if (f.is_open()) {
			int line_counter = 1;
			while (std::getline(f, line))
			{
				std::string::size_type sz;
				if (line_counter == 1) {
					float label = std::stof(line, &sz);
					int index = (i - 1)*NUMBER_OF_CLASSES + (label - 1);
					true_labels[index] = 1;
				}else if (line_counter == 3) {
					int j = 0;
					std::stringstream ssin(line);
					while (ssin.good() && j < NUMBER_OF_FEATURES) {
						std::string temp;
						ssin >> temp;
						input[((i - 1) * NUMBER_OF_FEATURES) + j] = std::stof(temp, &sz);
						++j;
					}
				}
				line_counter++;
			}
			f.close();
		}
		else {
			std::cout << "Could not open file \n";
		}

	}
	/*for (int i = 0; i < NUMBER_OF_INSTANCES * NUMBER_OF_CLASSES; i++) {
		if (i % NUMBER_OF_CLASSES == 0) std::cout << "NEW\n";
		std::cout << true_labels[i] << std::endl;
		
	}*/


	printf("MAIN \n");
	//Initialize network
	CharacterRecognition::initialize_network(NUMBER_OF_INSTANCES, NUMBER_OF_FEATURES, NUMBER_OF_CLASSES, HIDDEN_LAYER_SIZE, 0.4f);

	//float* input = (float *)malloc(NUMBER_OF_INSTANCES * NUMBER_OF_FEATURES * sizeof(float));
	//float* true_labels = (float *)malloc(NUMBER_OF_INSTANCES * sizeof(float));
	
	

	//Call training loop
	CharacterRecognition::train(input, true_labels, NUMBER_OF_EPOCHS);

	

	//Test

	//1. Forward Pass

	//2. Find the element with maximum value in output and output the result
}
