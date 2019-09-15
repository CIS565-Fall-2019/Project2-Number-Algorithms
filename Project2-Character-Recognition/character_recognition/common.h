#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <vector>
#include <wchar.h>

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

#define DEBUGGINGTRANSFERS 0

//Just guessing at what's appropriate here, will benchmark later (read: never)
#define BLOCKSIZE 512

//development defines
#define RSIZE 4
#define NUMTRAINING 2


//production defines
#ifndef RSIZE
#define RSIZE 52
#endif
#ifndef NUMTRAINING
#define NUMTRAINING (RSIZE)
#endif


typedef std::vector<uint8_t>	uint8_v;
typedef std::vector<float>		float_v;
typedef std::vector<float_v>	float_vv;

typedef struct filter3 {
	float kernel[9];
} filter3;

void printFloatPic(float* begin, int width, int height);

/**
 * Check for CUDA errors; print and exit if there was a problem.
 */
void checkCUDAErrorFn(const char *msg, const char *file = NULL, int line = -1);

inline int ilog2(int x) {
    int lg = 0;
    while (x >>= 1) {
        ++lg;
    }
    return lg;
}

inline int ilog2ceil(int x) {
    return x == 1 ? 0 : ilog2(x - 1) + 1;
}


namespace Common {
    /**
    * This class is used for timing the performance
    * Uncopyable and unmovable
    *
    * Adapted from WindyDarian(https://github.com/WindyDarian)
    */
    class PerformanceTimer
    {
    public:
	    PerformanceTimer()
	    {
		    cudaEventCreate(&event_start);
		    cudaEventCreate(&event_end);
	    }

	    ~PerformanceTimer()
	    {
		    cudaEventDestroy(event_start);
		    cudaEventDestroy(event_end);
	    }

	    void startCpuTimer()
	    {
		    if (cpu_timer_started) { throw std::runtime_error("CPU timer already started"); }
		    cpu_timer_started = true;

		    time_start_cpu = std::chrono::high_resolution_clock::now();
	    }

	    void endCpuTimer()
	    {
		    time_end_cpu = std::chrono::high_resolution_clock::now();

		    if (!cpu_timer_started) { throw std::runtime_error("CPU timer not started"); }

		    std::chrono::duration<double, std::milli> duro = time_end_cpu - time_start_cpu;
		    prev_elapsed_time_cpu_milliseconds =
			    static_cast<decltype(prev_elapsed_time_cpu_milliseconds)>(duro.count());

		    cpu_timer_started = false;
	    }

	    void startGpuTimer()
	    {
		    if (gpu_timer_started) { throw std::runtime_error("GPU timer already started"); }
		    gpu_timer_started = true;

		    cudaEventRecord(event_start);
	    }

	    void endGpuTimer()
	    {
		    cudaEventRecord(event_end);
		    cudaEventSynchronize(event_end);

		    if (!gpu_timer_started) { throw std::runtime_error("GPU timer not started"); }

		    cudaEventElapsedTime(&prev_elapsed_time_gpu_milliseconds, event_start, event_end);
		    gpu_timer_started = false;
	    }

	    float getCpuElapsedTimeForPreviousOperation() //noexcept //(damn I need VS 2015
	    {
		    return prev_elapsed_time_cpu_milliseconds;
	    }

	    float getGpuElapsedTimeForPreviousOperation() //noexcept
	    {
		    return prev_elapsed_time_gpu_milliseconds;
	    }

	    // remove copy and move functions
	    PerformanceTimer(const PerformanceTimer&) = delete;
	    PerformanceTimer(PerformanceTimer&&) = delete;
	    PerformanceTimer& operator=(const PerformanceTimer&) = delete;
	    PerformanceTimer& operator=(PerformanceTimer&&) = delete;

    private:
	    cudaEvent_t event_start = nullptr;
	    cudaEvent_t event_end = nullptr;

	    using time_point_t = std::chrono::high_resolution_clock::time_point;
	    time_point_t time_start_cpu;
	    time_point_t time_end_cpu;

	    bool cpu_timer_started = false;
	    bool gpu_timer_started = false;

	    float prev_elapsed_time_cpu_milliseconds = 0.f;
	    float prev_elapsed_time_gpu_milliseconds = 0.f;
    };
}

/**
* This class wraps up various data we have regarding our input into one package
*/
class InputData {
public:
	InputData();//constructor
	int value;//0-indexed "character value"
	int numElements;
	int width;
	int height;
	uint8_v data;
	float_v fData;
	float_v resultArray;

public:
	/**
	Puts the stored internal data into a regular array
	Assumes enough memory has been allocated
	*/
	void fillArray(uint8_t* dest);
	/**
	Creates a "correct" activation array,
	Which is all zeroes except where our feature is
	*/
	void fillActivationArray();
};//InputData

typedef std::vector<InputData> InputData_v;