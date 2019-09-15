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

int main(int argc, char* argv[]) {
	// Initialize Network
	CharacterRecognition::init(2, 4, 2);

	// Load Dataset 
	int num_instances = 4;
	float idata[8] = { 0, 0, 1, 0, 0, 1, 1, 1};
	//float ilabel[4] = { 0, 1, 1, 0 };
	float ilabel[8] = { 1, 0, 0, 1, 0, 1, 1, 0};

	//Parameters
	int epochs = 1500;
	float learning_rate = 0.1;

	// Train
	CharacterRecognition::train(idata, ilabel, num_instances, epochs, learning_rate);
	// Test
	
}
