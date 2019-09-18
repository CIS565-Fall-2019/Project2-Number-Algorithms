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

void printArray(const float *array, int n) {
	printf("[");
	for (int i = 0; i < n; i++) {
		printf("%f, ", array[i]);
	}
	printf("]\n");
}

void testImage(int i, float *output) {
	std::string filename = "../data-set/";
	std::string number = std::to_string(i + 1);
	if (number.length() == 1) {
		number = std::string("0").append(number);
	}
	filename.append(number);
	filename.append("info.txt");
	FILE * image = std::fopen(filename.c_str(), "r");
	int label;
	int dimensions;
	fscanf(image, "%d", &label);
	fscanf(image, "%d", &dimensions);
	float *colors = new float[dimensions];
	for (int j = 0; j < dimensions; j++) {
		int color;
		fscanf(image, "%d", &color);
		colors[j] = color;
	}

	CharacterRecognition::evaluate(colors, output);
	fclose(image);
	delete[] colors;
}

int main(int argc, char* argv[]) {
	float *output = new float[52];

	CharacterRecognition::init();

	testImage(0, output);
	//printArray(output, 52);

	CharacterRecognition::train(0.2f);

	testImage(0, output);
	//printArray(output, 52);

	CharacterRecognition::train(0.2f);

	//printArray(output, 52);

	delete[] output;

	CharacterRecognition::end();
}
