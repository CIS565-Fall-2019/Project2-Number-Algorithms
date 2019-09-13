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
#include <filesystem>
#include <stdio.h>
#include <math.h>

namespace fs = std::experimental::filesystem;

//###############################
// FILE READING STUFF
//###############################

const fs::path rootPath = fs::path("../data-set");

class InputData {
public:
	InputData();//constructor
	int value;
	int numElements;
	int width;
	int height;
	uint8_v data;

public:
	int fillArray(uint8_t* dest);
};//InputData

InputData::InputData(void) {
	data = uint8_v();
}//empty constructor

int InputData::fillArray(uint8_t* dest) {
	for (int i = 0; i < this->numElements; i++) {
		dest[i] = this->data[i];
	}//for
	return this->numElements;
}//fillArray

InputData readFile(std::string filename) {
	std::FILE* infile = std::fopen(filename.c_str(), "r");
	if (!infile) {
		printf("Error opening file %s!\nErrno: %d\n", filename.c_str(), errno);
		exit(1);
	}//if

	InputData* retval = new InputData();
	
	int numRead = std::fscanf(infile, "%d\n%d\n", &retval->value, &retval->numElements);
	if (numRead != 2) {
		printf("Error reading file %s!\nNumread %d\nErrno: %d\n", filename.c_str(), numRead, errno);
		exit(1);
	}//if

	retval->width	= (int) sqrt(retval->numElements);//assuming square images
	retval->height	= (int) sqrt(retval->numElements);
	
	int nextval = 0;

	for (int i = 0; i < retval->numElements; i++) {
		numRead = std::fscanf(infile, "%i", &nextval);
		if (numRead < 1) break;

		retval->data.push_back((uint8_t) nextval);
	}//for

	std::fclose(infile);
	InputData realValue = InputData(*retval);
	delete(retval);

	return realValue;
}//readFile

//###############################
// TESTING HELPERS
//###############################


//###############################
// MAIN
//###############################
int main(int argc, char* argv[]) {

	fs::path trialPath = rootPath / fs::path("01info.txt");

	printf("Path exists: %d\n", fs::exists(trialPath));

	InputData testData = readFile(trialPath.string());



}//main
