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

#include <string>
#include <fstream>
using namespace std;

void test_XOR() {
  const int input_size = 2;
  const int hidden_size = 4;
  const int output_size = 1;

  CharacterRecognition::MLP3 xor_mlp(input_size, hidden_size, output_size);

  // initial weights (from Hannah excel)
  float *wkj = new float[input_size * hidden_size];
  wkj[0] = 10.1;
  wkj[1] = 20;
  wkj[2] = 0.9;
  wkj[3] = 0.87;
  float *wji = new float[hidden_size * output_size];
  wji[0] = 41;
  wji[1] = -54;
  xor_mlp.init_weights(wkj, wji);


  // test input (row major)
  const int num_inputs = 4;
  float *xor_input = new float[num_inputs * input_size];
  float *xor_label = new float[num_inputs * output_size];

  xor_input[0] = 0; xor_input[4] = 0;
  xor_label[0] = 0;

  xor_input[1] = 0; xor_input[5] = 1;
  xor_label[1] = 1;

  xor_input[2] = 1; xor_input[6] = 0;
  xor_label[2] = 1;

  xor_input[3] = 1; xor_input[7] = 1;
  xor_label[3] = 0;

  xor_mlp.train(xor_input, xor_label, num_inputs, 150);
}

void test_image() {
  const int input_size = 10201;
  const int hidden_size = 128;
  const int output_size = 52;

  CharacterRecognition::MLP3 image_mlp(input_size, hidden_size, output_size);
  image_mlp.init_weights();

  // test input (row major)
  const int num_inputs = 52;
  float *image_input = new float[num_inputs * input_size];
  float *image_label = new float[num_inputs * output_size];
  memset(image_label, 0, sizeof(float) * num_inputs * output_size);

  for (int i = 0; i < 52; i++) {
    string file_idx = to_string(i+1);
    if (file_idx.length() == 1) {
      file_idx = "0" + file_idx;
    }
    string file_name = "../data-set/" + file_idx + "info.txt";
    ifstream input_stream(file_name);
    if (!input_stream) cerr << "Can't open input file:" << file_name << endl;
    int first, second;
    input_stream >> first;
    input_stream >> second;
    for (int j = 0; j < input_size; j++) {
      input_stream >> image_input[j * num_inputs + i];
    }
    image_label[i * num_inputs + i] = 1.0;
  }

  image_mlp.train(image_input, image_label, num_inputs, 200);
}

int main(int argc, char* argv[]) {
  test_XOR();
  //test_image();
}
