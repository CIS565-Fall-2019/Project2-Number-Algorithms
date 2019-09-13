#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include "common.h"
#include "mlp.h"

#define ALLOWKERNEL5 1

//These are definitions for index math in the 1d-2d world
#define UL(idx, w) (idx - w - 1)
#define UC(idx, w) (idx - w)
#define UR(idx, w) (idx - w + 1)
#define CL(idx, w) (idx - 1)
#define CC(idx, w) (idx)
#define CR(idx, w) (idx + 1)
#define DL(idx, w) (idx + w - 1)
#define DC(idx, w) (idx + w)
#define DR(idx, w) (idx + w + 1)


namespace CharacterRecognition {
    using Common::PerformanceTimer;
    PerformanceTimer& timer()
    {
        static PerformanceTimer timer;
        return timer;
    }

	//##################################
	// FUNCTION DELCARATIONS
	//##################################

	/**
	* Gets the "index" for the thread
	* Currently, only supporting single-dimensional block indexes
	* Computes all relevant x, y, z transformations
	*/
	__device__ int getIndex();

        
	//##################################
	// DEVICE FUNCTIONS
	//##################################


	__device__ int getIndex() {
		int threadIndex = threadIdx.x + (blockDim.x) * threadIdx.y + (blockDim.y * blockDim.x) * threadIdx.z;
		int overallIndex = threadIndex + blockIdx.x * (blockDim.x * blockDim.y * blockDim.z);

		return overallIndex;
	}//getIndex


	//##################################
	// DEVICE GLOBAL FUNCTIONS
	//##################################

	/**
	* Does a convolution from one image to another
	* A few notes:
	* Takes char data in for the input
	* Assuming we're running one thread per output pixel, and that we've sized things correctly for our filter
	* filter, idata, and odata must all be square
	* Also, currently only accepting filter widths of 3 and 5
	*/
	__global__ void convolve(float* filter, int filterWidth, uint8_t* idata, float* odata, int odataWidth) {
		int index = getIndex();
		if (index >= odataWidth * odataWidth) return;
		int idataW = odataWidth + 2;

		//get ourselves an "idata" index
		int iindex = (index / odataWidth) * 2 + 1 + idataW;
		

		float sum = 0;

		if (filterWidth == 3) {
			uint8_t relData[9];
			//Flips the kernel here
			relData[0] = idata[DR(iindex, idataW)];
			relData[1] = idata[DC(iindex, idataW)];
			relData[2] = idata[DL(iindex, idataW)];
			relData[3] = idata[CR(iindex, idataW)];
			relData[4] = idata[CC(iindex, idataW)];
			relData[5] = idata[CL(iindex, idataW)];
			relData[6] = idata[UR(iindex, idataW)];
			relData[7] = idata[UC(iindex, idataW)];
			relData[8] = idata[UL(iindex, idataW)];
			for (int i = 0; i < 9; i++) {
				sum += relData[i] * filter[i];
			}//for 9
		}//if 3
#if ALLOWKERNEL5
		else if (filterWidth == 5) {
			uint8_t relData[25];
			//Flips the kernel here (without the macro stuff)
			for (int i = 0; i < 5; i++) {
				int iOffset = idataW * (i - 2);
				for (int j = 0; j < 5; j++) {
					relData[5 * i + j] = idata[iindex + (j - 2) + iOffset];
				}//for
			}//for
			for (int i = 0; i < 25; i++) {
				sum += relData[i] * filter[i];
			}//for 25
		}//elif 5
#endif
		else {
			return;//please don't get here
		}//else

		odata[index] = sum;
 
	}//convolve


	// TODO: implement required elements for MLP sections 1 and 2 here
}
