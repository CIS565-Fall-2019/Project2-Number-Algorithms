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
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>

const int SIZE = 1 << 8; // feel free to change the size of array
const int NPOT = SIZE - 3; // Non-Power-Of-Two
//int *a = new int[SIZE];
//int *b = new int[SIZE];
//int *c = new int[SIZE];

//size of the three buffer
const int input_count = 2;
const int size_INPUT = 4;
const int size_HiD = 2;
const int size_OUTPUT = 2;
const int size_WEI1 = size_INPUT * size_HiD;
const int size_WEI2 = size_HiD * size_OUTPUT;

//create three buffers
float *input_ = new float[input_count * size_INPUT];
float *hidden_ = new float[size_HiD];
float *output_ = new float[size_OUTPUT];
float *real_ = new float[input_count * size_OUTPUT];

//weights buffer
float *w1= new float[size_WEI1];
float *w2 = new float[size_WEI2];

///real number read
const int character_num = 52;
const int feature_num = 10201;
const int class_num = 52;


int main(int argc, char* argv[]) {
    printf("\n");
    printf("****************\n");
    printf("** Recognize TESTS **\n");
    printf("****************\n");

	/////////////////simple test///////////////////////////
	//printDesc("input");
    //genArrayf(size_INPUT * input_count, input_, 9);  // Leave a 0 at the end to test that edge case
	zeroArrayf(size_INPUT * input_count, input_);
	input_[0] = 6.0;
	input_[1] = 1.0;
	input_[2] = 2.0;
	input_[3] = 5.0;
	input_[4] = 4.0;
	input_[5] = 0.0;
	input_[6] = 3.0;
	input_[7] = 0.0;
	//printArrayf(size_INPUT * input_count, input_, true);
	//printDesc("real");
	zeroArrayf(input_count * size_OUTPUT, real_);
	real_[1] = 1.0;
	real_[3] = 1.0;
	//printArrayf(input_count * size_OUTPUT, real_, true);

    zeroArrayf(size_HiD, hidden_);
	zeroArrayf(size_OUTPUT, output_);
    //printDesc("initial hidden");
	//printArrayf(size_HiD, hidden_, true);
	//printDesc("initial output");
	//printArrayf(size_OUTPUT, output_, true);
	
	//CharacterRecognition::build_network(2, 4, 2, 2, 0.1, 0.5);
	//CharacterRecognition::train(input_, real_, 10);

	/////////////////test with image///////////////////////////
	//load the img data info
	float *input = new float [character_num * feature_num];
	float *real = new float[character_num * class_num]();

	std::string filename;
	std::string pre;
	std::string path = "D:\\study\\2019fall\\cis565\\Project2-Number-Algorithms\\Project2-Character-Recognition\\data-set\\";
	//std::string p;
	//read character_num times
	int real_ind = 0;
	int input_ind = 0;
	for (int i = 1; i <= character_num; i++) {
		//set the whole path name
		if (i >= 10) {
			pre = std::to_string(i);
		} else {
			pre = std::to_string(0) + std::to_string(i);
		}
		filename = pre + "info.txt";
		std::ifstream f(path + filename, std::ios::in);
		std::string line;
		if (!f) {//false open
			std::cout << "error opening source file." << std::endl;
			return 0;
		}
		
		//class index
		std::getline(f, line);
		int cl = std::stoi(line);
		//std::cout << "class: " << cl << std::endl;
		real[real_ind * class_num + cl - 1] = 1.0;
		//std::cout << "ind: " << real_ind * class_num + cl - 1 << std::endl;
		real_ind++;

		//10201
		std::getline(f, line);

		int index = 0;
		while (index < feature_num) {
			std::string x;
			f >> x;
			input[input_ind++] = std::stof(x);
			index++;
			//std::cout << "data: " << x << std::endl;
		}
		f.close();
	}

	CharacterRecognition::build_network(52, 10201, 52, 25, 0.1, 0.5);
	CharacterRecognition::train(input, real, 40);
	CharacterRecognition::test(input, 30, 2);
}
