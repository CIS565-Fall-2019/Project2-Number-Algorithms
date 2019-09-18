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
#include<random>
using namespace std;

#define mnist 0

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

// http://eric-yuan.me/cpp-read-mnist/
int ReverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}
// http://eric-yuan.me/cpp-read-mnist/
void read_Mnist_Label(string filename, vector<double> &vec)
{
	 ifstream file (filename, ios::binary);

	 if (file.is_open()){

     int magic_number = 0;
     int number_of_images = 0;
     int n_rows = 0;
     int n_cols = 0;
     file.read((char*) &magic_number, sizeof(magic_number));
     magic_number = ReverseInt(magic_number);
     file.read((char*) &number_of_images,sizeof(number_of_images));
     number_of_images = ReverseInt(number_of_images);
     for(int i = 0; i < number_of_images; ++i)
     {
		 unsigned char temp = 0;
         file.read((char*) &temp, sizeof(temp));
         vec[i]= (double)temp;
     }
	}
}

// http://eric-yuan.me/cpp-read-mnist/
void read_Mnist(string filename, vector<cv::Mat> &vec) {
	ifstream file(filename, ios::binary);
	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);
		file.read((char*)&n_rows, sizeof(n_rows));
		n_rows = ReverseInt(n_rows);
		file.read((char*)&n_cols, sizeof(n_cols));
		n_cols = ReverseInt(n_cols);
		for (int i = 0; i < number_of_images; ++i)
		{
			cv::Mat tp = cv::Mat::zeros(n_rows, n_cols, CV_8UC1);
			for (int r = 0; r < n_rows; ++r)
			{
				for (int c = 0; c < n_cols; ++c)
				{
					unsigned char temp = 0;
					file.read((char*)&temp, sizeof(temp));
					tp.at<uchar>(r, c) = (int)temp;
				}
			}
			vec.push_back(tp);
		}
	}
}


void csv_write(double loss, int iteration_count, string file_name) {
	std::ofstream outfile;
	outfile.open(file_name, std::ios_base::app);
	outfile << iteration_count << "," << loss <<endl;
}

double* mattodouble(cv::Mat image) {
	double* inp = new double[image.rows*image.cols];
	if (image.isContinuous()) {
		for (int i = 0; i < image.rows*image.cols; i++) {
			inp[i] = (double)image.data[i];
		}
	}
	return inp;
}
double* onehotmnist(double index) {
	double *lab = new double[9];
	memset(lab, 0, 10 * sizeof(double));
	lab[(int)(index)] = 1;
	return lab;
}

int main(int argc, char* argv[]) {

	vector<double*>images;
	vector<double*>labels;
	vector<double*>images_test;
	vector<double*>labels_test;
	int classes = 0, input_dim = 0;
	if (mnist) {
		string filename = R"(..\data-set\train-images.idx3-ubyte)";
		int number_of_images = 60000;
		int image_size = 28 * 28;
		classes = 10;
		input_dim = image_size;
		vector<cv::Mat>images_train;
		read_Mnist(filename, images_train);
		filename = R"(..\data-set\train-labels.idx1-ubyte)";
		number_of_images = 60000;
		vector<double> labels_train(number_of_images);
		read_Mnist_Label(filename, labels_train);

		for (auto train : images_train) {
			images.push_back(mattodouble(train));
		}
		for (auto label : labels_train) {
			labels.push_back(onehotmnist(label));
		}

		filename = R"(..\data-set\t10k-images.idx3-ubyte)";
		number_of_images = 10000;
		vector<cv::Mat>images_te;
		read_Mnist(filename, images_te);
		filename = R"(..\data-set\t10k-labels.idx1-ubyte)";
		vector<double> labels_te(number_of_images);
		read_Mnist_Label(filename, labels_te);
		for (auto img : images_te) {
			images_test.push_back(mattodouble(img));
		}
		for (auto label : labels_te) {
			labels_test.push_back(onehotmnist(label));
		}


	}

	else {
		classes = 52;
		input_dim = 14*14;
		std::string path = "..\\data-set\\";
		std::vector<std::string>v = read_directory(path);
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
		images_test = images;
		labels_test = labels;
	}
	vector<int>layers = { 100, 50, 25 };
	vector<double>losses;
	vector<double>accuracy;
	CharacterRecognition::NeuralNet nn(input_dim, classes, layers);
	double loss_epoch = 0;
	int cnt = 0;
	for (int i = 0; i < 120000; i++) {
		int j = rand() % images.size();

		double *output = nn.forward(images[j]);
		loss_epoch += nn.calculateLoss(output, labels[j], classes);
		if (i%500 == 0)
		{
			losses.push_back(loss_epoch/images.size());
			cnt++;
			//csv_write(loss_epoch, cnt , R"(..\output_mnist_loss_500.csv)");
			loss_epoch = 0.0;
		}
		nn.backward(labels[j]);
	}
	double train_correct = 0;
	double test_correct = 0;
	for (int j = 0; j < images.size(); j++) {
		double *output = nn.forward(images[j]);
		train_correct += std::distance(labels[j], std::max_element(labels[j], labels[j] + classes)) == std::distance(output, std::max_element(output, output + classes));
	}
	cout << "Train Accuracy: " << train_correct / images.size() << endl;

	for (int j = 0; j < images_test.size(); j++) {
		double *output = nn.forward(images_test[j]);
		test_correct += std::distance(labels_test[j], std::max_element(labels_test[j], labels_test[j] + classes)) == std::distance(output, std::max_element(output, output + classes));
	}
	cout <<"Test Accuracy: "<< test_correct/images_test.size() << endl;
	return 0;
}
