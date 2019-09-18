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
#include <fstream>
#include <string>
using namespace std;

const float alpha = 0.1;
const int batchSize = 52;

void GotoLine(std::ifstream& file, unsigned int num) {
	file.seekg(std::ios::beg);
	for (int i = 0; i < num - 1; ++i) {
		file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
	}
	return;
}

using time_point_t = std::chrono::high_resolution_clock::time_point;
time_point_t time_start_cpu;
time_point_t time_end_cpu;

bool cpu_timer_started = false;

float prev_elapsed_time_cpu_milliseconds = 0.f;

void startCpuTimer()
{
	if (cpu_timer_started) { throw std::runtime_error("CPU timer already started"); }
	cpu_timer_started = true;

	time_start_cpu = std::chrono::high_resolution_clock::now();
}

void endCpuTimer()
{
	time_end_cpu = std::chrono::high_resolution_clock::now();

	if (!cpu_timer_started) { throw std::runtime_error("CPU timer not started"); }

	std::chrono::duration<double, std::milli> duro = time_end_cpu - time_start_cpu;
	prev_elapsed_time_cpu_milliseconds =
		static_cast<decltype(prev_elapsed_time_cpu_milliseconds)>(duro.count());

	cpu_timer_started = false;
}

int main(int argc, char* argv[]) {
	
	float *data = new float[10201 * 52];
	float *label = new float[52 * 52];
	
	for (int i = 1; i < 53; i++) {
		string filename = i + "info.txt";
		ifstream file(filename);
		string line;
		if (file.is_open())
		{
			GotoLine(file, 3);
			for (int j = 0; j < 10201; j++) {
				file >> data[i - 1 + j * 52];
			}
			file.close();
		}
	}
	for (int i = 0; i < 52; i++) {
		for (int j = 0; j < 52; j++) {
			if(i != j)
				label[i * 52 + j] = 0;
			else 
				label[i * 52 + j] = 1;
		}
	}

	//normalization
	for (int i = 0; i < 52; i++) {
		float avg = 0;
		float sigma = 0;
		for (int j = 0; j < 10201; j++) {
			avg += data[i + j * 52];
		}
		avg /= 10201.0f;
		for (int j = 0; j < 10201; j++) {
			sigma += (data[i + j * 52] - avg) * (data[i + j * 52] - avg);
		}
		sigma = std::sqrt(sigma / 10201.0f);
		for (int j = 0; j < 10201; j++) {
			data[i + j * 52] = (data[i + j * 52] - avg) / (sigma + 0.000001f);
		}
	}
	
	int layerNum = 3;
	int *layerSizes = new int[layerNum];
	layerSizes[0] = 10201;
	layerSizes[1] = 70;
	layerSizes[2] = 52;

	int totalWNum = 0;
	for (int i = 1; i < layerNum; i++) {
		totalWNum += layerSizes[i] * (1 + layerSizes[i - 1]);
	}
	float *weights = new float[totalWNum];
	float *grad = new float[totalWNum];

	CharacterRecognition::initializeW(weights, layerSizes, layerNum);

	startCpuTimer();
	float cost;
	for (int i = 0; i < 10; i++) {
		cost = CharacterRecognition::computeCostGrad(layerSizes, layerNum, batchSize, weights, grad, data, label);
		CharacterRecognition::updateWeights(totalWNum, weights, grad, alpha);
		printf("Epoch: %d Cost: %f \n", i, cost);
	}
	endCpuTimer();
	printf("Time used : %f ms", prev_elapsed_time_cpu_milliseconds);

	delete[] layerSizes;
	delete[] data;
	delete[] label;
	delete[] weights;
	delete[] grad;
}
