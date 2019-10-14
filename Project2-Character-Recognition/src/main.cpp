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

const int SIZE = 1 << 3; // feel free to change the size of array
const int NPOT = SIZE - 3; // Non-Power-Of-Two
int *a = new int[SIZE];
int *b = new int[SIZE];
int *c = new int[SIZE];

void printFloatArray(float *x, int n) {
    printf("    [ ");
    for (int i = 0; i < n; i++) {
        printf("%f ", x[i]);
	}
	printf("]\n");
}

int main(int argc, char* argv[]) {
	/*
	CHARACTER RECOGNITION TESTS
	*/
    printf("\n");
    printf("****************\n");
    printf("** CHARACTER RECOGNITION TESTS **\n");
    printf("****************\n");

	CharacterRecognition::XORTest();

    system("pause"); // stop Win32 console from closing on exit
	delete[] a;
	delete[] b;
	delete[] c;
}
