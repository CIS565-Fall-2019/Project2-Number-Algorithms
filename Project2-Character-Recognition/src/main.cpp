#include <cstdio>
#include <character_recognition/mlp.h>
#include <character_recognition/common.h>
#include "testing_helpers.hpp"
#include "cublas_v2.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<Windows.h>
#include<vector>
#include <fstream>
#include<stdio.h>
#include<stdlib.h>
#include<regex>

std::vector<std::string> read_directory(const std::string& name)
{
	std::vector<std::string> v;
	std::string pattern(name);
	pattern.append("\\*");
	WIN32_FIND_DATA data;
	HANDLE hFind;
	if ((hFind = FindFirstFile(pattern.c_str(), &data)) != INVALID_HANDLE_VALUE) {
		do {
			if (string(data.cFileName).find("bmp") != std::string::npos)
				v.push_back(name + data.cFileName);
		} while (FindNextFile(hFind, &data) != 0);
		FindClose(hFind);
	}
	return v;
}

void csv_write(double loss, int iteration_count, string file_name) {
	std::ofstream outfile;
	outfile.open(file_name, std::ios_base::app);
	outfile << iteration_count << "," << loss << endl;
}

int main(int argc, char* argv[]) {


	std::string path = "..\\data-set\\";
	std::vector<std::string>v = read_directory(path);
	vector<double*>images;
	vector<double*>labels;
	const int classes = 52;
	for (auto x : v) {
		cv::Mat image = cv::imread(x, 0);
		cv::Size size(14, 14);
		std::string output = x.substr(12, 2);
		int label_index = stoi(output);
		cv::resize(image, image, size);
		if (image.isContinuous()) {
			images.push_back(new double[image.rows*image.cols]);
			for (int i = 0; i < image.rows*image.cols; i++) {
				images.back()[i] = (double)image.data[i];
			}
			labels.push_back(new double[classes]);
			memset(labels.back(), 0, classes * sizeof(double));
			labels.back()[label_index - 1] = 1;
		}
	}
	const int inp_dim = 14 * 14;

	vector<int>layers = { 100, 50, 25 };
	vector<double>losses;
	CharacterRecognition::NeuralNet nn(inp_dim, classes, layers);
	for (int i = 0; i < 100; i++) {
		double loss_epoch = 0.0;
		for (int j = 0; j < classes; j++) {
			double *output = nn.forward(images[j]);
			loss_epoch += nn.calculateLoss(output, labels[j], classes);
			nn.backward(labels[j]);
		}
		losses.push_back(loss_epoch / classes);
		//csv_write(loss_epoch / 4, i, R"(..\output.csv)");
	}

	for (auto x : losses) {
		std::cout << x << endl;
	}
	//
	int count = 0;
	for (int j = 0; j < classes; j++) {
		double *output = nn.forward(images[j]);
		count += std::distance(labels[j], std::max_element(labels[j], labels[j] + classes)) == std::distance(output, std::max_element(output, output + classes));

	}
	cout << count << endl;
	for (auto x : images)
		delete[] x;
	for (auto x : labels)
		delete[] x;
	return 0;
}
