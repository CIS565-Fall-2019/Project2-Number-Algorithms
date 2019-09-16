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
#include "testing_helpers.hpp"

void scanTests(const int SIZE, const int NPOT, int *a, int *b , int *c, int blockSize) {
	genArray(SIZE - 1, a, 50);  // Leave a 0 at the end to test that edge case
	a[SIZE - 1] = 0;

	//zeroArray(SIZE, b);
	//StreamCompaction::CPU::scan(SIZE, b, a);
	//std::cout << StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation() << "  ";

	zeroArray(SIZE, c);
	StreamCompaction::Naive::scan(SIZE, c, a, blockSize);
	std::cout << StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation() << "  ";

	zeroArray(SIZE, c);
	StreamCompaction::Efficient::scan(SIZE, c, a, blockSize);
	std::cout << StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation() << "  ";

	zeroArray(SIZE, c);
	StreamCompaction::Efficient::scanEfficient(SIZE, c, a, blockSize);
	std::cout << StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation() << "  ";

	//zeroArray(SIZE, c);
	//StreamCompaction::Thrust::scan(SIZE, c, a);
	//std::cout << StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation() << "  ";

	//zeroArray(SIZE, c);
	//StreamCompaction::CPU::scan(NPOT, c, a);
	//std::cout << StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation() << "  ";

	zeroArray(SIZE, c);
	StreamCompaction::Naive::scan(NPOT, c, a, blockSize);
	std::cout  << StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation() << "  ";

	zeroArray(SIZE, c);
	StreamCompaction::Efficient::scan(NPOT, c, a, blockSize);
	std::cout  << StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation() << "  ";

	zeroArray(SIZE, c);
	StreamCompaction::Efficient::scanEfficient(NPOT, c, a, blockSize);
	std::cout << StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation() << std::endl;

	//zeroArray(SIZE, c);
	//StreamCompaction::Thrust::scan(NPOT, c, a);
	//std::cout << StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation() << std::endl;
}

void streamCompactionTests(const int SIZE, const int NPOT, int *a, int *b, int *c, int blockSize) {
	genArray(SIZE - 1, a, 4);  // Leave a 0 at the end to test that edge case
	a[SIZE - 1] = 0;

	int count, expectedCount, expectedNPOT;

	//zeroArray(SIZE, b);
	//count = StreamCompaction::CPU::compactWithoutScan(SIZE, b, a);
	//std::cout << StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation() << "  ";
	//expectedCount = count;
	////printCmpLenResult(count, expectedCount, b, b);

	//zeroArray(SIZE, c);
	//count = StreamCompaction::CPU::compactWithScan(SIZE, c, a);
	//std::cout << StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation() << "  ";
	////printCmpLenResult(count, expectedCount, b, c);

	zeroArray(SIZE, c);
	count = StreamCompaction::Efficient::compact(SIZE, c, a, false, blockSize);
	std::cout << StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation() << "  ";
	//printCmpLenResult(count, expectedCount, b, c);

	zeroArray(SIZE, c);
	count = StreamCompaction::Efficient::compact(SIZE, c, a, true, blockSize);
	std::cout << StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation() << "  ";
	//printCmpLenResult(count, expectedCount, b, c);

	//zeroArray(SIZE, c);
	//count = StreamCompaction::CPU::compactWithoutScan(NPOT, c, a);
	//std::cout << StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation() << "  ";
	//expectedNPOT = count;
	////printCmpLenResult(count, expectedNPOT, b, c);

	//zeroArray(SIZE, c);
	//count = StreamCompaction::CPU::compactWithScan(NPOT, c, a);
	//std::cout << StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation() << "  ";
	////printCmpLenResult(count, expectedCount, b, c);

	zeroArray(SIZE, c);
	count = StreamCompaction::Efficient::compact(NPOT, c, a, false, blockSize);
	std::cout << StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation() << "  ";
	//printCmpLenResult(count, expectedNPOT, b, c);

	zeroArray(SIZE, c);
	count = StreamCompaction::Efficient::compact(NPOT, c, a, true, blockSize);
	std::cout << StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation() << std::endl;
	//printCmpLenResult(count, expectedNPOT, b, c);
}

void metricsForDifferentN() {
	printf("\n");
	printf("****************\n");
	printf("** SCAN TESTS **\n");
	printf("****************\n");
	std::cout << "PowerOfTwo  " << "N  "
		<< "CPU:POT  " << "GPU:Naive:POT  " << "GPU:WorkEfficient:POT  " << "GPU:Optimized:POT  " << "GPU:Thrust:POT  "
		<< "CPU:NPOT  " << "GPU:Naive:NPOT  " << "GPU:WorkEfficient:NPOT  " << "GPU:Optimized:NPOT  " << "GPU:Thrust:NPOT" << std::endl;

	for (int s = 3; s <= 30; s++) {
		const int SIZE = 1 << s; // feel free to change the size of array
		const int NPOT = SIZE - 3; // Non-Power-Of-Two
		int *a = new int[SIZE];
		int *b = new int[SIZE];
		int *c = new int[SIZE];
		std::cout << s << "  " << SIZE << "  ";
		scanTests(SIZE, NPOT, a, b, c, 128);
		delete[] a;
		delete[] b;
		delete[] c;
	}

	printf("\n");
	printf("*****************************\n");
	printf("** STREAM COMPACTION TESTS **\n");
	printf("*****************************\n");
	std::cout << "PowerOfTwo  " << "N  "
		<< "CPU:WithoutScan:POT  " << "CPU:WithScan:POT  " << "GPU:WorkEfficient:POT  " << "GPU:Optimized:POT  "
		<< "CPU:NPOT  " << "CPU:WithScan:NPOT  " << "GPU:WorkEfficient:NPOT  " << "GPU:Optimized:NPOT  " << std::endl;
	for (int s = 3; s <= 30; s++) {
		const int SIZE = 1 << s; // feel free to change the size of array
		const int NPOT = SIZE - 3; // Non-Power-Of-Two
		int *a = new int[SIZE];
		int *b = new int[SIZE];
		int *c = new int[SIZE];
		std::cout << s << "  " << SIZE << "  ";
		streamCompactionTests(SIZE, NPOT, a, b, c, 128);
		delete[] a;
		delete[] b;
		delete[] c;
	}
}

void metricsForDifferentBlockSize() {
	printf("\n");
	printf("****************\n");
	printf("** SCAN TESTS **\n");
	printf("****************\n");
	std::cout << "BlockSize  "
		<< "GPU:Naive:POT  " << "GPU:WorkEfficient:POT  " << "GPU:Optimized:POT  "
		<< "GPU:Naive:NPOT  " << "GPU:WorkEfficient:NPOT  " << "GPU:Optimized:NPOT  "<< std::endl;

	int blockSizes[11] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
	for (int i = 0; i < 11; i++) {
		const int SIZE = 1 << 20; // feel free to change the size of array
		const int NPOT = SIZE - 3; // Non-Power-Of-Two
		int *a = new int[SIZE];
		int *b = new int[SIZE];
		int *c = new int[SIZE];
		std::cout << blockSizes[i] << "  ";
		scanTests(SIZE, NPOT, a, b, c, blockSizes[i]);
		delete[] a;
		delete[] b;
		delete[] c;
	}

	printf("\n");
	printf("*****************************\n");
	printf("** STREAM COMPACTION TESTS **\n");
	printf("*****************************\n");
	std::cout << "BlockSize  "
		<< "GPU:WorkEfficient:POT  " << "GPU:Optimized:POT  "
		<< "GPU:WorkEfficient:NPOT  " << "GPU:Optimized:NPOT  " << std::endl;
	for (int i = 0; i < 11; i++) {
		const int SIZE = 1 << 20; // feel free to change the size of array
		const int NPOT = SIZE - 3; // Non-Power-Of-Two
		int *a = new int[SIZE];
		int *b = new int[SIZE];
		int *c = new int[SIZE];
		std::cout << blockSizes[i] << "  ";
		streamCompactionTests(SIZE, NPOT, a, b, c, blockSizes[i]);
		delete[] a;
		delete[] b;
		delete[] c;
	}
}
int main(int argc, char* argv[]) {
	//metricsForDifferentN();
	metricsForDifferentBlockSize();
	system("pause"); // stop Win32 console from closing on exit
}
