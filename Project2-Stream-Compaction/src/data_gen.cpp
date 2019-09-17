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
#define do_scan true
#define do_sort false // true skips scan (so set to false if we want scan or stream compaction data)
#define cpu_scan true // always keep this true to get metrics (pass/fail)
#define naive_scan true
#define efficient_scan true
#define shared_mem_scan true
#define thrust_scan true
#define pow_2 true
#define max_pow 26
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
	while (true && !do_sort) {
		unsigned long long int SIZE;
		if (pow > max_pow)// 28 if stream_compaction
			pow = 4;
		cout << pow << endl;
		if (pow_2)
			 SIZE = 1ULL << pow; // Power of 2
		else
			SIZE = (1ULL << pow) - 3; // Non-Power-Of-Two
		long long *a = new long long[SIZE];
		long long *b = new long long[SIZE];
		long long *c = new long long[SIZE];
		genArray(SIZE - 1, a, (do_scan)? max_value_scan: max_value_compaction);  // Leave a 0 at the end to test that edge case
		a[SIZE - 1] = 0;
		// get time
		float time = -1;
		float count, expectedCount;
		// set solution
		if (cpu_scan) {
			zeroArray(SIZE, b); // reset ans
			zeroArray(SIZE, c);
			if (do_scan) {
				StreamCompaction::CPU::scan(SIZE, b, a);
				time = StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation();
				csv_write(time, SIZE, "cpu.csv");
				cout << "CPU:"; printArray(SIZE, b, true);
			}
			else {
				count = StreamCompaction::CPU::compactWithoutScan(SIZE, b, a); // baseline
				expectedCount = count;
				time = StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation();
				csv_write(time, SIZE, "cpu_without.csv");
				cout << "BAS (" << count << "):"; printArray(count, b, true); // baseline
				count = StreamCompaction::CPU::compactWithoutScan(SIZE, c, a); // without
				time = StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation();
				csv_write(time, SIZE, "cpu_with.csv");
				cout << "CPU (" <<count<<"):"; printCmpLenResult(count, expectedCount, b, c);
			}
		}
		if (naive_scan && pow < 30) {
			if (do_scan) {
				zeroArray(SIZE, c);
				StreamCompaction::Naive::scan(SIZE, c, a);
				cout << "Nai:"; printCmpResult(SIZE, b, c);
				time = StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation();
				csv_write(time, SIZE, "naive.csv");
			}
		}
		else if (efficient_scan)
			cout << "Nai: Failed" << endl;
		if (efficient_scan && pow < 30){
			zeroArray(SIZE, c);
			if (do_scan) {
				StreamCompaction::Efficient::scan(SIZE, c, a);
				cout << "Eff:"; printCmpResult(SIZE, b, c);
				time = StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation();
			}
			else {
				count = StreamCompaction::Efficient::compact(SIZE, c, a);
				time = StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation();
				cout << "Eff (" << count << "):"; printCmpLenResult(count, expectedCount, b, c);
			}
			csv_write(time, SIZE, "eff.csv");
		}
		else if (efficient_scan)
			cout << "Eff: Failed" << endl;
		if (shared_mem_scan) {
			zeroArray(SIZE, c);
			if (do_scan) {
				StreamCompaction::SharedMemory::scan(SIZE, c, a);
				time = StreamCompaction::SharedMemory::timer().getGpuElapsedTimeForPreviousOperation();
				cout << "ShM:"; printCmpResult(SIZE, b, c);
			}
			else {
				count = StreamCompaction::SharedMemory::compact(SIZE, c, a);
				time = StreamCompaction::SharedMemory::timer().getGpuElapsedTimeForPreviousOperation();
				cout << "ShM (" << count << "):"; printCmpLenResult(count, expectedCount, b, c);
			}
			csv_write(time, SIZE, "shared_mem.csv");
		}
		if (thrust_scan) {
			zeroArray(SIZE, c);
			if (do_scan) {
				StreamCompaction::Thrust::scan(SIZE, c, a);
				time = StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation();
				cout << "Thr:"; printCmpResult(SIZE, b, c);
				csv_write(time, SIZE, "thrust.csv");
			}
		}
		pow++;
		delete[] a;
		delete[] b;
		delete[] c;
	}
	pow = 4;
	while (true && do_sort) {
		unsigned long long int SIZE;
		if (pow > 27)
			pow = 4;
		cout << pow << endl;
		if (pow_2)
			SIZE = 1ULL<<pow; // Power of 2
		else
			SIZE = (1ULL << pow) - 3; // Non-Power-Of-Two
		long long *a = new long long[SIZE];
		long long *b = new long long[SIZE];
		long long *c = new long long[SIZE];
		genArray(SIZE - 1, a, (do_scan) ? max_value_scan : max_value_compaction);  // Leave a 0 at the end to test that edge case
		a[SIZE - 1] = 0;
		// get time
		float time = -1;
		float count, expectedCount;
		// copy a to b
		std::memcpy(b, a, SIZE * sizeof(long long));
		// sort using thrust + store time
		StreamCompaction::CPU::timer().startCpuTimer();
		thrust::sort(b, b + SIZE);
		StreamCompaction::CPU::timer().endCpuTimer();
		time = StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation();
		csv_write(time, SIZE, "thrust.csv");
		zeroArray(SIZE, c);
		Sorting::Radix::sort(SIZE, c, a, max_value_sorting);
		cout << "Radix: "; printCmpResult(SIZE, c, b);
		time = Sorting::Radix::timer().getGpuElapsedTimeForPreviousOperation();
		csv_write(time, SIZE, "radix.csv");
		pow++;
		delete[] a;
		delete[] b;
		delete[] c;
	}
}
