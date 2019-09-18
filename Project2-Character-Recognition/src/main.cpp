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
using CharacterRecognition::initCublas;
using CharacterRecognition::deleteCublas;

std::vector<std::string> parseDirectory(const std::string path);
void testMatrixMul();


int main(int argc, char* argv[]) {
	/****************************
	* TODO: User Input for training/loading/saving
	*/
	const std::string IMAGE_PATH = "..\\data-set\\";
	const std::string IMAGE_SEARCH_PATH = IMAGE_PATH + "*";
	std::vector<std::string> files = parseDirectory(IMAGE_SEARCH_PATH);
	initCublas();

	CharacterRecognition::testMatrixMul();

	Perceptron p(PIXELS, OUTPUTS);

	// Begin With Random Values
	p.randomizeWeights();
	p.updateCpu();

	// Load files and train on those files
	for (int i = 0; i < 10; i++) {
		for (auto &fname : files) {
			ImageFile inputFile(IMAGE_PATH + fname);

			p.loadTrainingDataSet(&inputFile);
			p.train();
			p.updateBackprop();
			p.updateCpu();
		}
		p.applyBackprop();
		p.updateCpu();
	}

	p.updateCpu();

	// Now Run against data set
	std::vector<std::string> correct_guesses;
	std::vector<std::string> wrong_guesses;
	for (auto &fname : files) {
		ImageFile inputFile(IMAGE_PATH + fname);

		p.loadDataSet(&inputFile);
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

	deleteCublas();
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
