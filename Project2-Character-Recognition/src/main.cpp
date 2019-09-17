/**
 * @file      main.cpp
 * @brief     Stream compaction test program
 * @authors   Chhavi Sharma
 * @date      2019
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
#define XOR 1
#define CR 1

int N ;  
int D ;    
int H ;     
int C ;     
double LR;
int epochs;

int main(int argc, char* argv[]) {

	printf("****************\n");
	printf("***MLP TESTS***\n");
	printf("****************\n");

	//==========================================================================================
	//================================= XOR GATE NEURAL NET ====================================
	//==========================================================================================
	if (XOR == true) {
		N = 4;     // Number of examples
		D = 2;     //Feature length per example
		H = 4;     // Number of Hidden unit
		C = 2;     // NN number of classes
		LR = 0.5;
		epochs = 1001;

		double	*losses = new double[epochs];
		double	*idata = new double[N*D];
		int		* preds = new int[N];
		int		* gtruth = new int[N];
		
		double	*w1 = new double[D*H];
		double	*w2 = new double[H*C];

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

		CharacterRecognition::trainMLP(N, D, H, C, idata, preds, gtruth, epochs, losses, LR, w1, w2, seed);
		printf("\nCompleted XOR Training\n");
		// STORE LOSSES
		std::ofstream myfile("xor_losses.txt");
		if (myfile.is_open())
		{
			for (int i = 0; i < epochs; i++) {
				myfile << std::fixed << std::setprecision(8) << losses[i] << '\n';
			}
			myfile.close();
			printf("Wrote loss to file\n");
		}

		// STORE WEIGHTS
		std::ofstream weight1_file("xor_W1_DxH_"+std::to_string(D)+"_x_"+std::to_string(H)+".txt");
		if (weight1_file.is_open())
		{
			for (int i = 0; i < D*H; i++) {
				weight1_file << std::fixed << std::setprecision(8) << w1[i] << '\n';
			}
			weight1_file.close();
			printf("Wrote w1 to file\n");
		}
		std::ofstream weight2_file("xor_W2_HxC_"+std::to_string(H)+"_x_"+std::to_string(C)+".txt");
		if (weight2_file.is_open())
		{
			for (int i = 0; i < C*H; i++) {
				weight2_file << std::fixed << std::setprecision(8) << w2[i] << '\n';
			}
			weight2_file.close();
			printf("Wrote w2 to file\n");
		}

		delete(losses);
		delete(idata);
		delete(preds);
		delete(gtruth);
		delete(w1);
		delete(w2);
	}
	//==========================================================================================
	//================================= CHARACTER RECOGNITION ==================================
	//==========================================================================================
	if (CR == true) {
		// Char Recognition
		N = 52;			// Number of examples
		D = 10201;		// Feature length per example
		H = 10;			// Number of Hidden unit
		C = 52;			// NN number of classes
		LR = 0.5;		// Learning Rate
		epochs = 5001;  // Epochs

		double *CR_losses = new double[epochs];
		double *CR_idata = new double[N*D];
		int * CR_preds = new int[N];
		int * CR_gtruth = new int[N];

		double	*w1 = new double[D*H];
		double	*w2 = new double[H*C];

		printf("\n\nLaunch CharRec Training\n");
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
				sstream >> CR_gtruth[i - 1];
				CR_gtruth[i - 1] -= 1;
				//printf(" gtruth = %d |", CR_gtruth[i - 1]);

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
			CR_idata[i] = id[i] / 255.0;
			//printf("\t %lf ", CR_idata[i]);
		}
		delete(id);

		CharacterRecognition::trainMLP(N, D, H, C, CR_idata, CR_preds, CR_gtruth, epochs, CR_losses, LR, w1, w2, seed);
		printf("\nCompleted CharRec Training\n");

		// STORE LOSSES
		std::ofstream myfile1("cr_losses.txt");
		if (myfile1.is_open())
		{
			for (int i = 0; i < epochs; i++) {
				myfile1 << std::fixed << std::setprecision(8) << CR_losses[i] << '\n';
			}
			myfile1.close();
			printf("Wrote loss to file\n");
		}

		// STORE WEIGHTS
		std::ofstream weight1file("cr_W1_DxH_"+std::to_string(D)+"_x_" + std::to_string(H)+".txt");
		if (myfile1.is_open())
		{
			for (int i = 0; i < D*H; i++) {
				weight1file << std::fixed << std::setprecision(8) << w1[i] << '\n';
			}
			weight1file.close();
			printf("Wrote w1 to file\n");
		}
		std::ofstream weight2file("cr_W2_HxC_" + std::to_string(H) + "_x_" + std::to_string(C)+".txt");
		if (weight2file.is_open())
		{
			for (int i = 0; i < C*H; i++) {
				weight2file << std::fixed << std::setprecision(8) << w2[i] << '\n';
			}
			weight2file.close();
			printf("Wrote w2 to file\n");
		}

		delete(CR_losses);
		delete(CR_idata);
		delete(CR_preds);
		delete(CR_gtruth);
		delete(w1);
		delete(w2);
	}

	return 0;
}
