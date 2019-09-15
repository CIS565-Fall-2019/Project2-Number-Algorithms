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

#define NUMBER_OF_INSTANCES 4
#define NUMBER_OF_FEATURES 2
#define HIDDEN_LAYER_SIZE 5
#define NUMBER_OF_CLASSES 2

#define LEARNING_RATE 0.01
#define NUMBER_OF_EPOCHS 1

int main(int argc, char* argv[]) {
	//Load data to arrays
	
	printf("MAIN \n");
	//Initialize network
	CharacterRecognition::initialize_network(NUMBER_OF_INSTANCES, NUMBER_OF_FEATURES, NUMBER_OF_CLASSES, HIDDEN_LAYER_SIZE, LEARNING_RATE);

	//float* input = (float *)malloc(NUMBER_OF_INSTANCES * NUMBER_OF_FEATURES * sizeof(float));
	//float* true_labels = (float *)malloc(NUMBER_OF_INSTANCES * sizeof(float));
	
	float input[NUMBER_OF_INSTANCES * NUMBER_OF_FEATURES] = { 0,0, 0,1, 1,0, 1,1 };
	float true_labels[NUMBER_OF_INSTANCES * NUMBER_OF_CLASSES] = { 1,0, 0,1 ,0,1, 1,0 };

	//Call training loop
	CharacterRecognition::train(input, true_labels, NUMBER_OF_EPOCHS);

	

	//Test

	//1. Forward Pass

	//2. Find the element with maximum value in output and output the result
}
