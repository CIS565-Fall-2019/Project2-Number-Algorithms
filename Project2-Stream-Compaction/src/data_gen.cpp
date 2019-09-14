/**
 * @file      data_gen.cpp
 * @brief     Stream compaction data gen program
 * @authors   Vaibhav Arcot
 * @date      2019
 * @copyright University of Pennsylvania
 */

#include <cstdio>
#include <stream_compaction/cpu.h>
#include <stream_compaction/naive.h>
#include <stream_compaction/efficient.h>
#include <stream_compaction/thrust.h>
#include <stream_compaction/radix.h>
#include <thrust/sort.h>
#include "testing_helpers.hpp"
#define max_value_scan 50
#define max_value_compaction 4
#define max_value_sorting 500
const unsigned long int SIZE = 1 << 29; // feel free to change the size of array
const unsigned long unsigned long int NPOT = SIZE - 3; // Non-Power-Of-Two
int *a = new int[SIZE];
int *b = new int[SIZE];
int *c = new int[SIZE];

int main(int argc, char* argv[]) {
	// Scan tests

	printf("\n");
	printf("****************\n");
	printf("** SCAN TESTS **\n");
	printf("****************\n");

	genArray(SIZE - 1, a, 50);  // Leave a 0 at the end to test that edge case
	a[SIZE - 1] = 0;
	printArray(SIZE, a, true);

	// initialize b using StreamCompaction::CPU::compactWithoutScan you implement
   // We use b for further comparison. Make sure your StreamCompaction::CPU::compactWithoutScan is correct.
	zeroArray(SIZE, b);
	printDesc("cpu scan, power-of-two");
	StreamCompaction::CPU::scan(SIZE, b, a);
	printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
	printArray(SIZE, b, true);
	
	try {
		zeroArray(SIZE, c);
		printDesc("naive scan, power-of-two");
		StreamCompaction::Efficient::scan(SIZE, c, a);
		printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
		//printArray(SIZE, c, true);
		printCmpResult(SIZE, b, c);
	}
	catch (...) {
		std::cout << "Failed to compute, but not dead yet"<<std::endl;
	}

	zeroArray(SIZE, c);
	printDesc("Shared memory scan, power-of-two");
	StreamCompaction::SharedMemory::scan(SIZE, c, a);
	printElapsedTime(StreamCompaction::SharedMemory::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
	//printArray(SIZE, c, true);
	printCmpResult(SIZE, b, c);

	

	system("pause"); // stop Win32 console from closing on exit
	delete[] a;
	delete[] b;
	delete[] c;
}
