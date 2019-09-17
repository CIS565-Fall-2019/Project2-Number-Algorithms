/**
 * @file      main.cpp
 * @brief     MLP Driver
 * @authors   Kushagra Goel
 * @date      2019
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

using namespace::std;


#define NUMBER_OF_INSTANCES 52
#define NUMBER_OF_FEATURES 10201
#define HIDDEN_LAYER_SIZE 200
#define NUMBER_OF_CLASSES 52


int main(int argc, char* argv[]) {
	/*int numEpochs = 10000;
	int hiddenDimensions[] = { 5 };
	CharacterRecognition::MultiLayerPerceptron *mlp = new CharacterRecognition::MultiLayerPerceptron(3, 1, hiddenDimensions, 2, 4);

	float inputs[] = { 0, 0, 1, 
						0, 1, 1, 
						1, 0, 1, 
						1, 1, 1};

	float labels[] = { 1, 0,
						0, 1,
						0, 1,
						1, 0 };

	float *predicted = new float[8];


	for (int i = 0; i < numEpochs; i++) {
		mlp->forward(inputs, predicted);
		cout << mlp->loss(labels, predicted) << endl;
		mlp->backward(labels, predicted, 0.1);
	}
	mlp->forward(inputs, predicted);
	for (int i = 0; i < 8; i++) {
		if (i % 2 == 0) {
			cout << endl;
		}
		cout << predicted[i] << "\t";
	}
	cout<<"Final Loss : "<< mlp->loss(labels, predicted) << endl;*/
	

	int numEpochs = 5000;
	float *input = new float[NUMBER_OF_INSTANCES * NUMBER_OF_FEATURES];
	float *true_labels = new float[NUMBER_OF_INSTANCES * NUMBER_OF_CLASSES];
	memset(true_labels, 0, NUMBER_OF_INSTANCES * NUMBER_OF_CLASSES * sizeof(float));
	for (int i = 0; i < NUMBER_OF_INSTANCES; i++) {
		ifstream file("S:\\CIS 565\\Project_2\\Project2-Number-Algorithms\\Project2-Character-Recognition\\data-set\\" + ((i + 1 < 10) ? to_string(0) : "") + to_string(i + 1) + "info.txt");
		if (!file.is_open()) {
			exit(-1);
		}
		int count = 0;
		string line;
		while (getline(file, line))
		{
			count++;
			if (count == 1) {
				int index = i*NUMBER_OF_CLASSES + (stof(line) - 1);
				true_labels[index] = 1;
			}
			if (count == 3) {
				stringstream ssin(line);
				for (int k = 0; ssin.good() && k < NUMBER_OF_FEATURES; k++) {
					string temp;
					ssin >> temp;
					input[(i * NUMBER_OF_FEATURES) + k] = stof(temp) / 255;
				}
			}
		}
		file.close();
	}


	printf("MAIN \n");
	//Initialize network
	int hiddenDimensions[] = { 200 };
	CharacterRecognition::MultiLayerPerceptron *mlp = new CharacterRecognition::MultiLayerPerceptron(NUMBER_OF_FEATURES, 1, hiddenDimensions, NUMBER_OF_CLASSES, NUMBER_OF_INSTANCES);
	//CharacterRecognition::MultiLayerPerceptron *mlp = new CharacterRecognition::MultiLayerPerceptron(NUMBER_OF_FEATURES, 200, NUMBER_OF_CLASSES, NUMBER_OF_INSTANCES);
	float *predicted = new float[NUMBER_OF_INSTANCES * NUMBER_OF_CLASSES];



	for (int i = 0; i < numEpochs; i++) {
		mlp->forward(input, predicted);
		cout << mlp->loss(true_labels, predicted) << endl;
		mlp->backward(true_labels, predicted, 0.01);
	}

		cout <<endl<<endl<< mlp->loss(true_labels, predicted) << endl;

}
