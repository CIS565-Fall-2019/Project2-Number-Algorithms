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

#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

unsigned long seed = time(0);

 //====CONFIG Neural Network for XOR ================
 //==================================================

/*
 // XOR
 const int N = 4;     // Number of examples
 const int D = 2;     //Feature length per example
 const int H = 4;     // Number of Hidden unit
 const int C = 2;     // NN number of classes
 const double LR = 0.5;
 const int epochs = 1000;
 */

 
 // Char Recognition
const int N = 52;     // Number of examples
const int D = 10201;  // Feature length per example
const int H = 10;     // Number of Hidden unit
const int C = 52;     // NN number of classes
const double LR = 0.5;
const int epochs = 5000;
 


double *losses = new double[epochs];
double *idata = new double[N*D];
int * preds = new int[N];
int * gtruth = new int[N];


int main(int argc, char* argv[]) {
	// Scan tests

	printf("\n");
	printf("****************\n");
	printf("***MLP TESTS***\n");
	printf("****************\n");
	/*
	printf("Launch XOR Training\n");

	// XOR input data set 2*4
 	idata[0] = 0.0;
	idata[1] = 0.0;
	idata[2] = 0.0;
	idata[3] = 1.0;
	idata[4] = 1.0;
	idata[5] = 0.0;
	idata[6] = 1.0;
	idata[7] = 1.0;

	// XOR ground truth data set 4
	gtruth[0] = 0;
	gtruth[1] = 1;
	gtruth[2] = 1;
	gtruth[3] = 0;

	CharacterRecognition::trainMLP(N, D, H, C, idata, preds, gtruth, epochs, losses, LR, seed);
	printf("\nCompleted XOR Training\n");
	// STORE LOSSES
	std::ofstream myfile("xor_losses.txt");
	if (myfile.is_open())
	{
		for (int i = 0; i < epochs; i++) {
			myfile << std::fixed << std::setprecision(8) << losses[i]<<'\n';
		}
		myfile.close();
	}

	*/
	//==========================================================================================
	//==========================================================================================

	printf("Launch CharRec Training\n");
	// Data loading
	printf("Loading data...\n");
	int data_sz = 0;
	int x = 0;

	std::string line;
	int *id = new int[N*D];
	for (int i = 1; i <= 52; i++) {
		std::string fname;
		if (i < 10) {
			fname = "C:\\Users\\chhavis\\cis565\\Project2-Number-Algorithms\\Project2-Character-Recognition\\data-set\\0" + std::to_string(i) + "info.txt";
		}
		else {
			fname = "C:\\Users\\chhavis\\cis565\\Project2-Number-Algorithms\\Project2-Character-Recognition\\data-set\\" + std::to_string(i) + "info.txt";
		}
		std::ifstream myfile(fname);
		std::stringstream sstream;
		std::stringstream sstream2;
		std::stringstream sstream3;

		//std::cout<<fname<<std::endl;
		// Reading file
		if (myfile.is_open())
		{

			// Read line 1
			getline(myfile, line);
			sstream << line;
			sstream >> gtruth[i - 1];
			gtruth[i - 1] -= 1;
			printf(" gtruth = %d |", gtruth[i - 1]);

			// Read line 2 // Data Size
			getline(myfile, line);
			sstream2 << line;
			sstream2 >> data_sz;
			//printf("data_sz = %d \n", data_sz);

			// Read line 3 Pixel values
			getline(myfile, line);
			sstream3 << line;
			for (int j = 0; j < data_sz; j++) {
				sstream3 >> id[(i - 1) * 10201 + j];
			}

			myfile.close();
		}
		else {
			printf("Unable to open file.\n");;
		}
	}

	// Normalize Data
	for (int i = 0; i < N*D; i++) {
		idata[i] = id[i] / 255.0;
		//printf("\t %lf ", idata[i]);
	}
	delete(id);

	CharacterRecognition::trainMLP(N, D, H, C, idata, preds, gtruth, epochs, losses, LR, seed);
	printf("\nCompleted CharRec Training\n");

	// STORE LOSSES
	std::ofstream myfile("cr_losses.txt");
	if (myfile.is_open())
	{
		for (int i = 0; i < epochs; i++) {
			myfile << std::fixed << std::setprecision(8) << losses[i] << '\n';
		}
		myfile.close();
	}


	return 0;
}
