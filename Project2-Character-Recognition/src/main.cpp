/**
 * Character Recognition
 * John Marcao, CIS565 2019
 */

#include <cstdio>
#include <vector>
#include <regex>
#include <windows.h>
#include <character_recognition/mlp.h>
#include <character_recognition/common.h>
#include "testing_helpers.hpp"

using CharacterRecognition::Matrix;
using CharacterRecognition::ImageFile;
using CharacterRecognition::Perceptron;

constexpr int PIXELS = 10201;
constexpr int OUTPUTS = 52;

std::vector<std::string> parseDirectory(const std::string path);
void testMatrixMul();

int main(int argc, char* argv[]) {
	/****************************
	* TODO: User Input for training/loading/saving
	*/
	const std::string IMAGE_PATH = "..\\data-set\\*";
	std::vector<std::string> files = parseDirectory(IMAGE_PATH);

	Perceptron p(PIXELS, OUTPUTS);

	// Begin With Random Values
	p.randomizeWeights();
	for (auto &fname : files) {
		ImageFile inputFile(fname);
		Matrix inputData(PIXELS, 1);
		inputFile.readImage(&inputData);

		p.loadTrainingDataSet(inputFile.getExpectedNumber(), &inputData);
		p.train(100);
	}

	// Now Run against data set
	std::vector<std::string> correct_guesses;
	std::vector<std::string> wrong_guesses;
	for (auto &fname : files) {
		ImageFile inputFile(fname);
		Matrix inputData(PIXELS, 1);
		inputFile.readImage(&inputData);

		p.loadDataSet(&inputData);
		p.run();

		if(inputFile.getExpectedNumber() == p.getLastResult()) {
			correct_guesses.push_back(fname);
		}
		else {
			wrong_guesses.push_back(fname);
		}
	}

	// Report Results
	int correct = correct_guesses.size();
	int wrong = wrong_guesses.size();
	float accuracy = (float)correct / (wrong + correct);
	std::cout << "Run complete with accuracy " << accuracy << std::endl;
	std::cout << "MLP was wrong about the following files: " << std::endl;
	for (auto &f : wrong_guesses) {
		std::cout << "\t" << f << std::endl;
	}

	return 0;
}

std::vector<std::string> parseDirectory(const std::string path) {
	std::vector<std::string> ret;
	std::regex fileMatch(".*\.txt$");

	// Directory walking adapted from https://www.bfilipek.com/2019/04/dir-iterate.html

	WIN32_FIND_DATA FindFileData;
	HANDLE hFind = FindFirstFile(path.c_str(), &FindFileData);
	if (hFind == INVALID_HANDLE_VALUE) {
		throw std::runtime_error("FindFirstFile failed!");
	}

	do {
		std::string file(FindFileData.cFileName);
		if (std::regex_match(file, fileMatch)) {
			ret.push_back(file);
		}
	} while (FindNextFile(hFind, &FindFileData) != 0);

	FindClose(hFind);

	return ret;
}

void testMatrixMul() {
	Matrix m_a(10201, 1);     // Input Values
	Matrix m_b(10201, 10201); // Weights
	Matrix m_c(10201, 1);     // Output Values

	// Init matrix
	for (int i = 0; i < m_a.getLen(); i++) {
		m_a.cpu_data[i] = i;
	}
	for (int i = 0; i < m_b.getLen(); i++) {
		m_b.cpu_data[i] = m_b.getLen() - i;
	}

	// Populate Device
	m_a.copyCpuToDev();
	m_b.copyCpuToDev();

	matrixMul(&m_a, &m_b, &m_c);

	m_c.copyDevToCpu();
}
