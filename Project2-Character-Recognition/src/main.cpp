/**
 * @file      main.cpp
 * @brief     Stream compaction test program
 * @authors   Weiqi Chen
 * @date      2019
 * @copyright University of Pennsylvania
 */

#include <cstdio>
#include <character_recognition/mlp.h>
#include <character_recognition/common.h>
#include "testing_helpers.hpp"
#include <fstream>
#include <string>

using namespace std;
bool test(CharacterRecognition::mlp &mlp, int label);
int decode(int n, float* array);


void xor_test() {
	const int input_size = 2;
	const int hidden_size = 2;
	const int output_size = 1;

	CharacterRecognition::mlp xor(input_size, hidden_size, output_size);
	float *wkj = new float[input_size * hidden_size];
	float *wji = new float[hidden_size * output_size];
	
	wkj[0] = 10.1;
	wkj[1] = 20;
	wkj[2] = 0.9;
	wkj[3] = 0.87;
	
	wji[0] = 41;
	wji[1] = -54;
	xor.initWeights(wkj, wji);

	const int combo = 4;
	float *x = new float[combo * input_size];
	float *y = new float[combo * output_size];

	x[0] = 0;
	x[4] = 0;
	y[0] = 0;

	x[1] = 0;
	x[5] = 1;
	y[1] = 1;

	x[2] = 1;
	x[6] = 0;
	y[2] = 1;

	x[3] = 1;
	x[7] = 1;
	y[3] = 0;

	cout << "--- XOR ---" << endl;
	for (int e = 0; e < 14; e++) {
		// Train
		xor.train(x, y, combo, 10);
		float err = xor.getError();

		std::cout << "epoch: " << e * 10 << " | error: " << err << std::endl;
	}
}

void char_reg() {

	const int num_data = 52;
	const int input_size = 10201;
	const int hidden_size = 256;
	const int output_size = 52;
	CharacterRecognition::mlp characterRec(input_size, hidden_size, output_size);
	characterRec.initRandom();
	
	float *x = new float[num_data * input_size];
	float *y = new float[num_data * output_size];
	memset(y, 0, num_data * output_size * sizeof(float));

	for (int i = 0; i < 52; i++) {
		string id = to_string(i + 1);
		if (id.length() == 1) {
			id = "0" + id;
		}
		string name = "../data-set/" + id + "info.txt";
		ifstream input_stream(name);

		int a, b;
		input_stream >> a;
		input_stream >> b;
		for (int j = 0; j < input_size; j++) {
			input_stream >> x[i+ j * num_data];
		}
		y[i * num_data + i] = 1.0;
	}

	cout << "--- Character Recognition ---" << endl;
	for (int t = 0; t < 14; t++) {

		float err = characterRec.getError();
		int correct_cnt = 0;
		for (int i = 1; i <= 52; i++) {
			if (test(characterRec, i)) {
				correct_cnt++;
			}
		}
		std::cout << "epoch: " << t * 10 << " | error: " << err << " | accuracy: " << correct_cnt / 52.0 * 100 << "%" << std::endl;
	}
	cout << "---" << endl;
	for (int i = 1; i <= 52; i++) {
		if (test(characterRec, i)) {
		}
	}
}

bool test(CharacterRecognition::mlp &mlp, int label) {

	float *x = new float[10201];
	float *y = new float[52];

	string id = to_string(label);
	if (id.length() == 1) {
		id = "0" + id;
	}
	string name = "../data-set/" + id + "info.txt";
	ifstream input_stream(name);
	int a, b;
	input_stream >> a;
	input_stream >> b;
	
	for (int i = 0; i < 10201; i++) {
		input_stream >> x[i];
	}
	mlp.predict(x, y, 1);
	int pred = decode(52, y) + 1;
	cout << "Target:" << label << ", Predicted:" << pred << endl;
	

	delete[] x;
	delete[] y;
	if (pred == label) {
		return true;
	}
	else {
		return false;
	}

}

int decode(int n, float* array) {
	float temp = 10000;
	int index = 0;
	for (int i = 0; i < n; i++) {
		if (array[i] > temp) {
			index = i;
			temp = array[i];
		}
	}
	return index;
}

int main(int argc, char* argv[]) {
	xor_test();
	char_reg();
}