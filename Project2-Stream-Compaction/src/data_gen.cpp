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
#include <string>
#include <fstream>
using namespace std;

#define max_value_scan 50
#define max_value_compaction 4
#define max_value_sorting 500
#define cpu_scan true
#define naive_scan true
#define efficient_scan true
#define shared_mem_scan true
#define thrust_scan true
#define pow_2 true

void csv_write(float time, unsigned long long size, string file_name){
	std::ofstream outfile;
	outfile.open(file_name, std::ios_base::app);
	outfile <<size<<","<< time<<endl;
}


int main(int argc, char* argv[]) {
	printf("\n");
	printf("****************\n");
	printf("** SCAN Data Gen **\n");
	printf("****************\n");
	int pow = 4;
	while (true) {
		unsigned long long int SIZE;
		cout << pow << endl;
		if (pow >= 31)
			pow = 4;
		if (pow_2)
			 SIZE = std::pow(2,pow); // Power of 2
		else
			SIZE = std::pow(2, pow) - 3; // Non-Power-Of-Two
		int *a = new int[SIZE];
		int *b = new int[SIZE];
		int *c = new int[SIZE];
		genArray(SIZE - 1, a, 50);  // Leave a 0 at the end to test that edge case
		a[SIZE - 1] = 0;
		// get time
		float time = -1;
		// set solution
		if (cpu_scan) {
			zeroArray(SIZE, b);
			StreamCompaction::CPU::scan(SIZE, b, a);
			time = StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation();
			csv_write(time, SIZE, "cpu.csv");
			cout << "CPU:"; printArray(SIZE, b, true);
		}
		if (naive_scan && pow < 30) {
			zeroArray(SIZE, c);
			StreamCompaction::Naive::scan(SIZE, c, a);
			cout << "Nai:"; printCmpResult(SIZE, b, c);
			time = StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation();
			csv_write(time, SIZE, "naive.csv");
		}
		else if (efficient_scan)
			cout << "Nai: Failed" << endl;
		if (efficient_scan && pow < 30){
			zeroArray(SIZE, c);
			StreamCompaction::Efficient::scan(SIZE, c, a);
			cout << "Eff:"; printCmpResult(SIZE, b, c);
			time = StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation();
			csv_write(time, SIZE, "eff.csv");
		}
		else if (efficient_scan)
			cout << "Eff: Failed" << endl;
		if (shared_mem_scan) {
			zeroArray(SIZE, c);
			StreamCompaction::SharedMemory::scan(SIZE, c, a);
			time = StreamCompaction::SharedMemory::timer().getGpuElapsedTimeForPreviousOperation();
			cout << "ShM:"; printCmpResult(SIZE, b, c);
			csv_write(time, SIZE, "shared_mem.csv");
		}
		if (thrust_scan) {
			zeroArray(SIZE, c);
			StreamCompaction::Thrust::scan(SIZE, c, a);
			time = StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation();
			cout << "Thr:"; printCmpResult(SIZE, b, c);
			csv_write(time, SIZE, "thrust.csv");
		}
		pow++;
		delete[] a;
		delete[] b;
		delete[] c;
	}
}
