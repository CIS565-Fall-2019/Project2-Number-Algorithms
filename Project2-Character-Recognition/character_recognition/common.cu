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

//void memory_debug_float(int elements, float* cuda_mem, float* cpu_mem)
//{
//	printf("elements %d\n ", elements);
//	cudaMemcpy(cpu_mem, cuda_mem, elements * sizeof(float), cudaMemcpyDeviceToHost);
//	checkCUDAErrorFn("debug failed!");
//	printf("=============================\n");
//	for (int i = 0; i < elements; i++)
//	{
//		printf("out[%d] %d \n", i, cpu_mem[i]);
//	}
//	printf("=============================\n");
//}