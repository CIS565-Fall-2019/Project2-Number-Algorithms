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


// CONFIG ITEMS

const int N = 52;    // Number of examples
const int P = 10201; // Feature length

int inputLayerSize	= 2;  //NN input layer size 
int hiddenLayerSize = 2;  //NN hidden layer size 
int outputLayerSize = 1;  //NN output layer size
int nClasses		= 2;  //NN number of classes

int *a = new int[SIZE];
int *b = new int[SIZE];
int *c = new int[SIZE];

int main(int argc, char* argv[]) {
    // Scan tests

    printf("\n");
    printf("****************\n");
    printf("***MLP TESTS***\n");
    printf("****************\n");



	CharacterRecognition::trainMLP(SIZE, b, a);
	CharacterRecognition::trainMLP(SIZE, b, a);

	return 0;
}
