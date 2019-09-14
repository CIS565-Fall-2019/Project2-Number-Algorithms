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
#include <cublas.h>

#define HIDDEN_LAYER_SIZE 100
#define NUMBER_OF_CLASSES 52

#define LEARNING_RATE 0.01
#define NUMBER_OF_EPOCHS 100

int main(int argc, char* argv[]) {
	//Load data to arrays
	float* input;
	int* true_labels;

	//Initialize network
	CharacterRecognition::initialize_network(NUMBER_OF_CLASSES, HIDDEN_LAYER_SIZE, LEARNING_RATE);

	//Call training loop
	CharacterRecognition::train(input, true_labels, NUMBER_OF_EPOCHS);

	

	//Test

	//1. Forward Pass

	//2. Find the element with maximum value in output and output the result
}
