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
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
}

/**
* Constructor (empty)
*/
InputData::InputData(void) {
	data = uint8_v();
}//empty constructor

/**
* Puts the stored internal data into a regular array
* Assumes enough memory has been allocated
*/
void InputData::fillArray(uint8_t* dest) {
	memcpy(dest, this->data.data(), this->numElements * sizeof(uint8_t));
}//fillArray