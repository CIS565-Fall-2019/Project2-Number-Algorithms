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

short getShade(float input, float scale = 1.0) {
	if (input < 0) return '\u0020';
	else if (input < scale / 4) return '\u2591';
	else if (input < scale / 2) return '\u2592';
	else if (input < 3 * scale / 4) return '\u2593';
	else return '\u2588';
}//getShade

void printFloatPic(float* begin, int width, int height) {
	printf("\n");
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			float rawVal = begin[i * width + j];
			printf("%lc", getShade(rawVal));
		}//for
		printf("\n");
	}//for
	printf("\n");
}//printFloatPic