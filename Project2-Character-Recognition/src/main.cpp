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
//#include <../Project2-Stream-Compaction/stream_compaction/cpu.h>
//#include <../Project2-Stream-Compaction/stream_compaction/naive.h>
//#include <../Project2-Stream-Compaction/stream_compaction/efficient.h>
//#include <../Project2-Stream-Compaction/stream_compaction/thrust.h>
#include "testing_helpers.hpp"

const int SIZE = 1 << 8; // feel free to change the size of array
const int NPOT = SIZE - 3; // Non-Power-Of-Two
int *a = new int[SIZE];
int *b = new int[SIZE];
int *c = new int[SIZE];

//size of the three buffer
const int SIZE_INPUT = 4;
const int SIZE_HiD = 2;
const int SIZE_OUTPUT = 1;
const int SIZE_WEI = SIZE_INPUT * SIZE_HiD + SIZE_HiD * SIZE_OUTPUT;

//create three buffers
int *input = new int[SIZE_INPUT];
int *hidden = new int[SIZE_HiD];
int *output = new int[SIZE_OUTPUT];
//weights buffer
float *weights = new float[SIZE_WEI];

int main(int argc, char* argv[]) {
    // Scan tests

    printf("\n");
    printf("****************\n");
    printf("** SCAN TESTS **\n");
    printf("****************\n");
	
    genArray(SIZE_INPUT, input, 5);  // Leave a 0 at the end to test that edge case
	printArray(SIZE_INPUT, input, true);

    zeroArray(SIZE_HiD, hidden);
	zeroArray(SIZE_OUTPUT, output);
    printDesc("initial hidden");
	printArray(SIZE_HiD, hidden, true);
	printDesc("initial output");
	printArray(SIZE_OUTPUT, output, true);
 //   StreamCompaction::CPU::scan(SIZE, b, a);
 //   printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
 //   printArray(SIZE, b, true);

 //   zeroArray(SIZE, c);
 //   printDesc("cpu scan, non-power-of-two");
 //   StreamCompaction::CPU::scan(NPOT, c, a);
 //   printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
 //   printArray(NPOT, b, true);
 //   printCmpResult(NPOT, b, c);

 //   zeroArray(SIZE, c);
 //   printDesc("naive scan, power-of-two");
 //   StreamCompaction::Naive::scan(SIZE, c, a);
 //   printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
 //   //printArray(SIZE, c, true);
 //   printCmpResult(SIZE, b, c);

	///* For bug-finding only: Array of 1s to help find bugs in stream compaction or scan
	//onesArray(SIZE, c);
	//printDesc("1s array for finding bugs");
	//StreamCompaction::Naive::scan(SIZE, c, a);
	//printArray(SIZE, c, true); */

 //   zeroArray(SIZE, c);
 //   printDesc("naive scan, non-power-of-two");
 //   StreamCompaction::Naive::scan(NPOT, c, a);
 //   printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
 //   //printArray(SIZE, c, true);
 //   printCmpResult(NPOT, b, c);

 //   zeroArray(SIZE, c);
 //   printDesc("work-efficient scan, power-of-two");
 //   StreamCompaction::Efficient::scan(SIZE, c, a);
 //   printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
 //   //printArray(SIZE, c, true);
 //   printCmpResult(SIZE, b, c);

 //   zeroArray(SIZE, c);
 //   printDesc("work-efficient scan, non-power-of-two");
 //   StreamCompaction::Efficient::scan(NPOT, c, a);
 //   printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
 //   //printArray(NPOT, c, true);
 //   printCmpResult(NPOT, b, c);

 //   zeroArray(SIZE, c);
 //   printDesc("thrust scan, power-of-two");
 //   StreamCompaction::Thrust::scan(SIZE, c, a);
 //   printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
 //   //printArray(SIZE, c, true);
 //   printCmpResult(SIZE, b, c);

 //   zeroArray(SIZE, c);
 //   printDesc("thrust scan, non-power-of-two");
 //   StreamCompaction::Thrust::scan(NPOT, c, a);
 //   printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
 //   //printArray(NPOT, c, true);
 //   printCmpResult(NPOT, b, c);

 //   printf("\n");
 //   printf("*****************************\n");
 //   printf("** STREAM COMPACTION TESTS **\n");
 //   printf("*****************************\n");

 //   // Compaction tests

 //   genArray(SIZE - 1, a, 4);  // Leave a 0 at the end to test that edge case
 //   a[SIZE - 1] = 0;
 //   printArray(SIZE, a, true);

 //   int count, expectedCount, expectedNPOT;

 //   // initialize b using StreamCompaction::CPU::compactWithoutScan you implement
 //   // We use b for further comparison. Make sure your StreamCompaction::CPU::compactWithoutScan is correct.
 //   zeroArray(SIZE, b);
 //   printDesc("cpu compact without scan, power-of-two");
 //   count = StreamCompaction::CPU::compactWithoutScan(SIZE, b, a);
 //   printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
 //   expectedCount = count;
 //   printArray(count, b, true);
 //   printCmpLenResult(count, expectedCount, b, b);

 //   zeroArray(SIZE, c);
 //   printDesc("cpu compact without scan, non-power-of-two");
 //   count = StreamCompaction::CPU::compactWithoutScan(NPOT, c, a);
 //   printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
 //   expectedNPOT = count;
 //   printArray(count, c, true);
 //   printCmpLenResult(count, expectedNPOT, b, c);

 //   zeroArray(SIZE, c);
 //   printDesc("cpu compact with scan");
 //   count = StreamCompaction::CPU::compactWithScan(SIZE, c, a);
 //   printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
 //   printArray(count, c, true);
 //   printCmpLenResult(count, expectedCount, b, c);

 //   zeroArray(SIZE, c);
 //   printDesc("work-efficient compact, power-of-two");
 //   count = StreamCompaction::Efficient::compact(SIZE, c, a);
 //   printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
 //   //printArray(count, c, true);
 //   printCmpLenResult(count, expectedCount, b, c);

 //   zeroArray(SIZE, c);
 //   printDesc("work-efficient compact, non-power-of-two");
 //   count = StreamCompaction::Efficient::compact(NPOT, c, a);
 //   printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
 //   //printArray(count, c, true);
 //   printCmpLenResult(count, expectedNPOT, b, c);

 //   system("pause"); // stop Win32 console from closing on exit
	//delete[] a;
	//delete[] b;
	//delete[] c;
}
