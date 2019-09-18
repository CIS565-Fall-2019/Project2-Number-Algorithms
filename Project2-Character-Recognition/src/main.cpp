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
#include <Windows.h>
#include <vector>
#include <sstream>
#include <iostream>
#include <math.h>

using namespace::std;


#define NUMBER_OF_INSTANCES 52
#define NUMBER_OF_FEATURES 10201
#define HIDDEN_LAYER_SIZE 200
#define NUMBER_OF_CLASSES 52
// read MNIST data into double vector, OpenCV Mat, or Armadillo mat
// free to use this code for any purpose
// author : Eric Yuan 
// my blog: http://eric-yuan.me/
// part of this code is stolen from http://compvisionlab.wordpress.com/

#include <math.h>
#include <iostream>

using namespace std;


int ReverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void read_Mnist(string filename, vector<vector<double> > &vec)
{
	ifstream file(filename, ios::binary);
	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);
		file.read((char*)&n_rows, sizeof(n_rows));
		n_rows = ReverseInt(n_rows);
		file.read((char*)&n_cols, sizeof(n_cols));
		n_cols = ReverseInt(n_cols);
		for (int i = 0; i < number_of_images; ++i)
		{
			vector<double> tp;
			for (int r = 0; r < n_rows; ++r)
			{
				for (int c = 0; c < n_cols; ++c)
				{
					unsigned char temp = 0;
					file.read((char*)&temp, sizeof(temp));
					tp.push_back((double)temp);
				}
			}
			vec.push_back(tp);
		}
	}
	else {
		cout << "Couldn't Open"<<endl;
	}
}

void read_Mnist_Label(string filename, vector<double> &vec)
{
	ifstream file(filename, ios::binary);
	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);
		for (int i = 0; i < number_of_images; ++i)
		{
			unsigned char temp = 0;
			file.read((char*)&temp, sizeof(temp));
			vec[i] = (double)temp;
		}
	}
}

int main(int argc, char* argv[]) {
	int numEpochs = 10000;
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
	cout<<"Final Loss : "<< mlp->loss(labels, predicted) << endl;
	

	numEpochs = 1000;
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


	int hiddenDimensionsAlpha[] = { 200 };
	CharacterRecognition::MultiLayerPerceptron *mlpAlpha = new CharacterRecognition::MultiLayerPerceptron(NUMBER_OF_FEATURES, 1, hiddenDimensionsAlpha, NUMBER_OF_CLASSES, NUMBER_OF_INSTANCES);
	delete(predicted);
	predicted = new float[NUMBER_OF_INSTANCES * NUMBER_OF_CLASSES];



	for (int i = 0; i < numEpochs; i++) {
		mlpAlpha->forward(input, predicted);
		cout << mlpAlpha->loss(true_labels, predicted) << endl;
		mlpAlpha->backward(true_labels, predicted, 0.01);
	}

		cout <<endl<<endl<< mlp->loss(true_labels, predicted) << endl;





	
	string filename = R"(..\data-set\mnist\t10k-images-idx3-ubyte)";
	int number_of_images = 10000;
	int image_size = 28 * 28;
	int number_of_classes = 10;
	numEpochs = 100;
	int hiddenDimensionsMnist[] = {64 , 32};
	CharacterRecognition::MultiLayerPerceptron *mnistMlp = new CharacterRecognition::MultiLayerPerceptron(image_size, 2, hiddenDimensionsMnist, number_of_classes, 100);

	float *inputMnist;
	inputMnist = new float[number_of_images * image_size];

	float *labelMnist;
	labelMnist = new float[number_of_images*number_of_classes];

	float *predictedMnist;
	predictedMnist = new float[number_of_images*number_of_classes];
		
	//read MNIST image into double vector
	vector<vector<double> > vec;
	read_Mnist(filename, vec);

	for (int i = 0; i < number_of_images; i++) {
		for (int j = 0; j < image_size; j++) {
			inputMnist[i * image_size + j] = (vec[i][j] - 0.5) * 2;
		}
	}

    filename = R"(..\data-set\mnist\t10k-labels-idx1-ubyte)";
	//read MNIST label into double vector
	vector<double> vecLabel(number_of_images);
	read_Mnist_Label(filename, vecLabel);


	for (int i = 0; i < number_of_images; i++) {
		for (int j = 0; j < number_of_classes; j++) {
			labelMnist[i * number_of_classes + j] = (j == vecLabel[i])? 1.0 : 0.0;
		}
	}

	for (int i = 0; i < numEpochs; i++) {
		mnistMlp->forward(inputMnist, predictedMnist);
		cout << mnistMlp->loss(labelMnist, predictedMnist) << endl;
		mnistMlp->backward(labelMnist, predictedMnist, 0.0001);
	}


}
