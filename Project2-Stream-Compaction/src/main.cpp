/**
 * @file      main.cpp
 * @brief     Stream compaction test program
 * @authors   Kai Ninomiya
 * @date      2015
 * @copyright University of Pennsylvania
 */

#include <cstdio>
#include <stream_compaction/cpu.h>
#include <stream_compaction/naive.h>
#include <stream_compaction/efficient.h>
#include <stream_compaction/thrust.h>
#include <stream_compaction/radix.h>
#include "testing_helpers.hpp"


//0001->  10000000
const int SIZE = 1 << 8; // feel free to change the size of array
const int NPOT = SIZE - 3; // Non-Power-Of-Two
const int Sort_Size = 20;
//three int array
int *a = new int[SIZE];
int *b = new int[SIZE];
int *c = new int[SIZE];
//test radix
int *ra = new int[Sort_Size];
int *raout = new int[Sort_Size];

int main(int argc, char* argv[]) {
    // Scan tests

    printf("\n");
    printf("****************\n");
    printf("** SCAN TESTS **\n");
    printf("****************\n");

	//size-1 -> generate number, a->array pointer, 50->max value of each number
    genArray(SIZE - 1, a, 50);  // Leave a 0 at the end to test that edge case
    a[SIZE - 1] = 0;//the last one is 0
    printArray(SIZE, a, true);
	//std::cout << "16: "<< ilog2ceil(16) << std::endl;
	//std::cout << "13: "<< ilog2ceil(13) << std::endl;
	//std::cout << "14: " << ilog2ceil(14) << std::endl;
	//int aa = (5 >> 0) & 1;
	//int aa2 = (5 >> 1) & 1;
	//int aa3 = (5 >> 2) & 1;
	//int aa4 = (5 >> 3) & 1;
	//std::cout << "5<<0: " << aa << std::endl;
	//std::cout << "5<<1: " << aa2 << std::endl;
	//std::cout << "5<<2: "<< aa3  << std::endl;
	//std::cout << "5<<3: " << aa4 << std::endl;
	std::cout << "SIZE: " << SIZE << ", NPOT:" << NPOT << std::endl;

    // initialize b using StreamCompaction::CPU::scan you implement
    // We use b for further comparison. Make sure your StreamCompaction::CPU::scan is correct.
    // At first all cases passed because b && c are all zeroes.
    zeroArray(SIZE, b);//initial as zero 
    printDesc("cpu scan, power-of-two");
	//scan
    StreamCompaction::CPU::scan(SIZE, b, a);
	//print the execute time
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
	//print array b
    printArray(SIZE, b, true);

	//zero initial array c
    zeroArray(SIZE, c);
    printDesc("cpu scan, non-power-of-two");
	//NPOT = SIZE - 3
    StreamCompaction::CPU::scan(NPOT, c, a);
	//print execute time
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(NPOT, b, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("naive scan, power-of-two");
    StreamCompaction::Naive::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
	printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

	/* For bug-finding only: Array of 1s to help find bugs in stream compaction or scan
	onesArray(SIZE, c);
	printDesc("1s array for finding bugs");
	StreamCompaction::Naive::scan(SIZE, c, a);
	printArray(SIZE, c, true); */

    zeroArray(SIZE, c);
    printDesc("naive scan, non-power-of-two");
    StreamCompaction::Naive::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printArray(SIZE, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, power-of-two");
    StreamCompaction::Efficient::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, non-power-of-two");
    StreamCompaction::Efficient::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("thrust scan, power-of-two");
    StreamCompaction::Thrust::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("thrust scan, non-power-of-two");
    StreamCompaction::Thrust::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    printf("\n");
    printf("*****************************\n");
    printf("** STREAM COMPACTION TESTS **\n");
    printf("*****************************\n");

    // Compaction tests

    genArray(SIZE - 1, a, 4);  // Leave a 0 at the end to test that edge case
    a[SIZE - 1] = 0;
    printArray(SIZE, a, true);

    int count, expectedCount, expectedNPOT;

    // initialize b using StreamCompaction::CPU::compactWithoutScan you implement
    // We use b for further comparison. Make sure your StreamCompaction::CPU::compactWithoutScan is correct.
    zeroArray(SIZE, b);
    printDesc("cpu compact without scan, power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(SIZE, b, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    expectedCount = count;
    printArray(count, b, true);
    printCmpLenResult(count, expectedCount, b, b);

    zeroArray(SIZE, c);
    printDesc("cpu compact without scan, non-power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(NPOT, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    expectedNPOT = count;
    printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("cpu compact with scan");
    count = StreamCompaction::CPU::compactWithScan(SIZE, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, power-of-two");
    count = StreamCompaction::Efficient::compact(SIZE, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, non-power-of-two");
    count = StreamCompaction::Efficient::compact(NPOT, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);

    //test radix sort
	//ra[0] = 4;//100
	//ra[1] = 7;//111
	//ra[2] = 2;//010
	//ra[3] = 6;//110
	//ra[4] = 3;//011
	//ra[5] = 5;//101
	//ra[6] = 1;//001
	//ra[7] = 0;//000
	//ra[8] = 7;//111
	//ra[9] = 1;//001
	genArray(Sort_Size, ra, 64);
	zeroArray(Sort_Size, raout);
	StreamCompaction::Radix::radix_sort(Sort_Size, 6, raout, ra);
	printArray(Sort_Size, ra, true);
	printArray(Sort_Size, raout, true);

	system("pause"); // stop Win32 console from closing on exit
	delete[] a;
	delete[] b;
	delete[] c;
	delete[] ra;
	delete[] raout;
}
