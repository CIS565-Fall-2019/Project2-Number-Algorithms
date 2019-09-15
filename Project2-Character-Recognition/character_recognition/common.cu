#include "common.h"

void checkCUDAErrorFn(const char *msg, const char *file, int line) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file) {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %d: %s: %s\n", line, msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
}

/**
* Constructor (empty)
*/
InputData::InputData(void) {
	data = uint8_v();
	fData = float_v();
	resultArray = float_v();
}//empty constructor

void InputData::fillArray(uint8_t* dest) {
	memcpy(dest, this->data.data(), this->numElements * sizeof(uint8_t));
}//fillArray

void InputData::fillActivationArray(float_v* dest) {

	dest->clear();

	for (int i = 0; i < RSIZE; i++) {
		if (i == this->value) {
			dest->push_back(1.0);
		}//if
		else {
			dest->push_back(0.0);
		}//else
	}//for
}//fillActivationArray

char getShade(float input, float scale = 1.0) {
	std::string shadeString = std::string(" .:-=+*#%@");
	int numLevels = shadeString.size();
	float step = 1.0 / numLevels;

	float normInput = input / scale;
	if (normInput < 0.0) normInput = 0.0;
	if (normInput >= 1.0) normInput = 0.99;

	int inputStep = (int)(normInput / step);
	return shadeString.at(inputStep);

}//getShade

void printFloatPic(float* begin, int width, int height) {
	printf("\n");
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			float rawVal = begin[i * width + j];
			printf("%c", getShade(rawVal));
		}//for
		printf("\n");
	}//for
	printf("\n");
}//printFloatPic