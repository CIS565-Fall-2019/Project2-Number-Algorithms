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
#include "testing_helpers.hpp"''

#include <string>
#include <fstream>
using namespace std;

// find the index of the maximum element in an array.
int find_max_idx(int n, float* array);
bool test_image(CharacterRecognition::MLP3 &mlp, int label, bool verbose);

void train_XOR() {
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


  // training input (row major)
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

  cout << "--- XOR Example ---" << endl;
  for (int t = 0; t < 20; t++) {
    // Train
    xor_mlp.train(xor_input, xor_label, num_inputs, 10, false);
    float cost = xor_mlp.total_error();

    std::cout << "epoch: " << t * 10 << " | cost: " << cost << std::endl;
  }
}

void train_image() {
  const int input_size = 10201;
  const int hidden_size = 128;
  const int output_size = 52;

  CharacterRecognition::MLP3 image_mlp(input_size, hidden_size, output_size);
  image_mlp.init_weights();

  // training input (row major)
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

  cout << "--- Character Recognition ---" << endl;
  for (int t = 0; t < 20; t++) {
    // Train
    CharacterRecognition::timer().startCpuTimer();
    image_mlp.train(image_input, image_label, num_inputs, 10, false);
    CharacterRecognition::timer().endCpuTimer();
    float time_elapsed = CharacterRecognition::timer().getCpuElapsedTimeForPreviousOperation();
    float cost = image_mlp.total_error();

    // Test
    int correct_cnt = 0;
    for (int i = 1; i <= 52; i++) {
      if (test_image(image_mlp, i, false)) {
        correct_cnt++;
      }
    }
    std::cout << "epoch: " << t * 10 << " | cost: " << cost << " | correct: " << correct_cnt / 52.0 * 100 << "% | time elapsed: " << time_elapsed << " ms" << std::endl;
  }
  cout << "---" << endl;
  for (int i = 1; i <= 52; i++) {
    if (test_image(image_mlp, i, true)) {
    }
  }
}

bool test_image(CharacterRecognition::MLP3 &mlp, int label, bool verbose) {
  string file_idx = to_string(label);
  if (file_idx.length() == 1) {
    file_idx = "0" + file_idx;
  }
  string file_name = "../data-set/" + file_idx + "info.txt";
  ifstream input_stream(file_name);
  if (!input_stream) cerr << "Can't open input file:" << file_name << endl;
  int first, second;
  input_stream >> first;
  input_stream >> second;
  float *test_input = new float[10201];
  float *test_output = new float[52];
  for (int j = 0; j < 10201; j++) {
    input_stream >> test_input[j];
  }
  mlp.predict(test_input, test_output, 1);
  int label_output = find_max_idx(52, test_output) + 1;
  if (verbose) {
    cout << "Target Label:" << label << ", Predicted Label:" << label_output << endl;
  }

  delete[] test_input;
  delete[] test_output;

  return label_output == label;
}

int find_max_idx(int n, float* array) {
  float max = FLT_MIN;
  int idx = 0;
  for (int i = 0; i < n; i++) {
    if (array[i] > max) {
      idx = i;
      max = array[i];
    }
  }
  return idx;
}

int main(int argc, char* argv[]) {
  train_XOR();
  cout << endl;
  train_image();
}
