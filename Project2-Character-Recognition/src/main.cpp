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
	CharacterRecognition::init();
	//CharacterRecognition::train();

	FILE * image = std::fopen("../data-set/02info.txt", "r");
	int label;
	int dimensions;
	fscanf(image, "%d", &label);
	fscanf(image, "%d", &dimensions);
	float *colors = new float[dimensions];
	for (int i = 0; i < dimensions; i++) {
		int color;
		fscanf(image, "%d", &color);
		colors[i] = color;
	}
	CharacterRecognition::evaluate(colors);
	fclose(image);
	delete[] colors;

	CharacterRecognition::end();
}
