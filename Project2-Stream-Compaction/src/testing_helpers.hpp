#pragma once

#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <string>
#include <ctime>

template<typename T>
int cmpArrays(unsigned long int n, T *a, T *b) {
    for (unsigned long int i = 0; i < n; i++) {
        if (a[i] != b[i]) {
            printf("    a[%lld] = %lld, b[%lld] = %lld\n", i, a[i], i, b[i]);
            return 1;
        }
    }
    return 0;
}

void printDesc(const char *desc) {
    printf("==== %s ====\n", desc);
}

template<typename T>
void printCmpResult(unsigned long int n, T *a, T *b) {
    printf("    %s \n",
            cmpArrays(n, a, b) ? "FAIL VALUE" : "passed");
}

template<typename T>
void printCmpLenResult(unsigned long int n, unsigned long int expN, T *a, T *b) {
    if (n != expN) {
        printf("    expected %lld elements, got %lld\n", expN, n);
    }
    printf("    %s \n",
            (n == -1 || n != expN) ? "FAIL COUNT" :
            cmpArrays(n, a, b) ? "FAIL VALUE" : "passed");
}


void zeroArray(unsigned long int n, long long *a) {
    for (unsigned long int i = 0; i < n; i++) {
        a[i] = 0;
    }
}

void onesArray(unsigned long int n, long long *a) {
	for (unsigned long int i = 0; i < n; i++) {
		a[i] = 1;
	}
}

void genArray(unsigned long int n, long long *a, int maxval) {
    srand(time(nullptr));

    for (unsigned long int i = 0; i < n; i++) {
        a[i] = rand() % maxval;
    }
}

void printArray(unsigned long int n, long long *a, bool abridged = false) {
    printf("    [ ");
    for (unsigned long int i = 0; i < n; i++) {
        if (abridged && i + 2 == 15 && n > 16) {
            i = n - 2;
            printf("... ");
        }
        printf("%3d ", a[i]);
    }
    printf("]\n");
}

template<typename T>
void printElapsedTime(T time, std::string note = "")
{
	std::cout << "   elapsed time: " << time << "ms    " << note << std::endl;
}