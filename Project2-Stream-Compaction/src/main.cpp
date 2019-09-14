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
#include <thrust/sort.h>
#include "testing_helpers.hpp"
#define max_value_scan 50
#define max_value_compaction 4
#define max_value_sorting 500
const unsigned long int SIZE = 1<<15; // feel free to change the size of array
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

    // initialize b using StreamCompaction::CPU::scan you implement
    // We use b for further comparison. Make sure your StreamCompaction::CPU::scan is correct.
    // At first all cases passed because b && c are all zeroes.
    zeroArray(SIZE, b);
    printDesc("cpu scan, power-of-two");
    StreamCompaction::CPU::scan(SIZE, b, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(SIZE, b, true);

    zeroArray(SIZE, c);
    printDesc("cpu scan, non-power-of-two");
    StreamCompaction::CPU::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(NPOT, b, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("naive scan, power-of-two");
    StreamCompaction::Naive::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("naive scan, non-power-of-two");
    StreamCompaction::Naive::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, power-of-two");
    StreamCompaction::Efficient::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, non-power-of-two");
    StreamCompaction::Efficient::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    // printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

	zeroArray(SIZE, c);
	printDesc("Shared memory scan, power-of-two");
	StreamCompaction::SharedMemory::scan(SIZE, c, a);
	printElapsedTime(StreamCompaction::SharedMemory::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
	//printArray(SIZE, c, true);
	printCmpResult(SIZE, b, c);

	zeroArray(SIZE, c);
	printDesc("Shared memory scan, non power-of-two");
	StreamCompaction::SharedMemory::scan(NPOT, c, a);
	printElapsedTime(StreamCompaction::SharedMemory::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
	//printArray(SIZE, c, true);
	printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("thrust scan, power-of-two");
    StreamCompaction::Thrust::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("thrust scan, non-power-of-two");
    StreamCompaction::Thrust::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    printf("\n");
    printf("*****************************\n");
    printf("** STREAM COMPACTION TESTS **\n");
    printf("*****************************\n");

    // Compaction tests

    genArray(SIZE - 1, a, max_value_compaction);  // Leave a 0 at the end to test that edge case
    a[SIZE - 1] = 0;
    printArray(SIZE, a, true);

	unsigned long int count, expectedCount, expectedNPOT;

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

	printf("\n");
	printf("*****************************\n");
	printf("** RADIX SORT TESTS **\n");
	printf("*****************************\n");
	// radix sort tests
	genArray(SIZE - 1, a, max_value_sorting);  // Leave a 0 at the end to test that edge case
	a[SIZE - 1] = 0;
	printArray(SIZE, a, true);
	printf("Data generation and results computed using thrust\n");
	// generate 2 ground truths using thrust sort (one for powers of 2 and the other one for non powers of 2)
	int *gt_pot = new int[SIZE]();
	std::memcpy(gt_pot, a, SIZE * sizeof(int));
	thrust::sort(gt_pot, gt_pot + SIZE);
	int *gt_npot = new int[NPOT]();
	std::memcpy(gt_npot, a, NPOT * sizeof(int));
	thrust::sort(gt_npot, gt_npot + NPOT);

	zeroArray(SIZE, c);
	printDesc("radix sort, power-of-two");
	Sorting::Radix::sort(SIZE, c, a, max_value_sorting);
	printArray(SIZE, c, true);
	printCmpResult(SIZE, c, gt_pot);
	printElapsedTime(Sorting::Radix::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");

	zeroArray(SIZE, c);
	printDesc("radix sort, power-of-two");
	Sorting::Radix::sort(NPOT, c, a, max_value_sorting);
	printArray(NPOT, c, true);
	printCmpResult(NPOT, c, gt_npot);
	printElapsedTime(Sorting::Radix::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");

    system("pause"); // stop Win32 console from closing on exit
	delete[] a;
	delete[] b;
	delete[] c;
	delete[] gt_pot;
	delete[] gt_npot;
}
