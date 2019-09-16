#include <cstdio>
#include <character_recognition/mlp.h>
#include <character_recognition/common.h>
#include "testing_helpers.hpp"
#include "cublas_v2.h"
//#include <opencv2/opencv.hpp>
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
#include<Windows.h>
#include<vector>

//std::vector<std::string> read_directory(const std::string& name)
//{
//	std::vector<std::string> v;
//	std::string pattern(name);
//	pattern.append("\\*");
//	WIN32_FIND_DATA data;
//	HANDLE hFind;
//	if ((hFind = FindFirstFile(pattern.c_str(), &data)) != INVALID_HANDLE_VALUE) {
//		do {
//			v.push_back(data.cFileName);
//		} while (FindNextFile(hFind, &data) != 0);
//		FindClose(hFind);
//	}
//	return v;
//}
int main(int argc, char* argv[]) {
		
		
		//std::string path = R"(..\data-set)";
		//std::vector<std::string>v =  read_directory(path);
		//for (auto x: v) {
		//	cv::Mat image = cv::imread(R"(..\data-set\01out.bmp)", 0);
		//}
//		for (const auto & entry : fs::directory_iterator(path))
//			
//		uchar * arr = image.isContinuous() ? image.data : image.clone().data;
//		uint length = image.total()*image.channels();
//		std::cout << length << std::endl;
	const int inp_dim = 196;
	float data[inp_dim] = {1};
	vector<int>layers = {98,65};
	int classes = 52;
	CharacterRecognition::NeuralNet nn(inp_dim, classes, layers);
	float *output = nn.forward(data);
	float sum = 0;
	for (int i = 0; i < classes; i++) {
		std::cout << output[i] << std::endl;
		sum += output[i];
	}
	std::cout << sum << std::endl;
	delete[] output;
	return 0;
	}
