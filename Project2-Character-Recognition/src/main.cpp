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

InputData readFile(std::string filename) {
	std::FILE* infile = std::fopen(filename.c_str(), "r");
	if (!infile) {
		printf("Error opening file %s!\nErrno: %d\n", filename.c_str(), errno);
		exit(1);
	}//if

	InputData retval =  InputData();
	
	int numRead = std::fscanf(infile, "%d\n%d\n", &retval.value, &retval.numElements);
	if (numRead != 2) {
		printf("Error reading file %s!\nNumread %d\nErrno: %d\n", filename.c_str(), numRead, errno);
		exit(1);
	}//if

	retval.value--;//0-index it

	retval.width	= (int) sqrt(retval.numElements);//assuming square images
	retval.height	= (int) sqrt(retval.numElements);
	
	int nextval = 0;

	for (int i = 0; i < retval.numElements; i++) {
		numRead = std::fscanf(infile, "%i", &nextval);
		if (numRead < 1) break;

		retval.data.push_back((uint8_t)nextval);
	}//for
	for (int i = 0; i < retval.numElements; i++) {
		retval.fData.push_back((float)(retval.data[i] / 255.0));
	}//for

	retval.fillActivationArray();

	std::fclose(infile);

	return retval;
}//readFile


//###############################
// MAIN
//###############################
int main(int argc, char* argv[]) {

	InputData_v allRecords = InputData_v();

	fs::path trialPath;
	InputData testData;
	for (int i = 1; i <= RSIZE; i++) {
		char numBuffer[11] = {};
		std::sprintf(numBuffer, "%02dinfo.txt", i);
		trialPath = rootPath / fs::path(numBuffer);
		testData = readFile(trialPath.string());
		allRecords.push_back(testData);
	}//for
	//trialPath = rootPath / fs::path("05info.txt");
	//testData = readFile(trialPath.string());
	//allRecords.push_back(testData);

#ifdef NUMTRAINING
	allRecords.resize(NUMTRAINING);
#endif

	float resultArray[RSIZE] = {};

	CharacterRecognition::kmallocBuffers();
	//CharacterRecognition::testMatMul();
	
	CharacterRecognition::trainWeights(allRecords, 5000);

	//Print how we're doing, results-wise
	CharacterRecognition::printForwardResults(allRecords);

	//CharacterRecognition::printWeights();

	CharacterRecognition::kfreeBuffers();

}//main
