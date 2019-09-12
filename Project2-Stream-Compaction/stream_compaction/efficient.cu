#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define blockSize 128

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

		void memory_debug(int elements, int* cuda_mem, int* cpu_mem, const int* cpu_in)
		{
			cudaMemcpy(cpu_mem, cuda_mem, elements * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAErrorFn("copy out failed!");
			printf("=============================\n");
			for (int i = 0; i < elements; i++)
			{
				printf("out[%d] %d ", i, cpu_mem[i]);
				printf("in[%d] %d\n", i, cpu_in[i]);
			}
			printf("=============================\n");
		}

		__global__ void kernel_inclusive_to_exclusive(int bufflength, int* idata, int* odata, int* inc_byte)
		{
			int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

			if (tid >= bufflength)
				return;

			if (tid == 0) {
				odata[tid] = 0;
				return;
			}

			if (tid == bufflength - 1) {
				odata[tid] = idata[tid] + inc_byte[0];
				return;
			}
			// shift one
			odata[tid] = idata[tid];
			return;
		}

		// according to notes we need to padd with zeros to accomodate not 
		// perfect logs.
		__global__ void kernel_padd_0s(int* idata,int bufflength,int padded_length)
		{
			int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

			if (tid > bufflength && tid < padded_length)
				idata[tid] = 0;

			return;
		}

		/*
		* perform and upsweep
		* from the notes that means 
		* fir d=0 to log2n-1
		*	for k =0; to n-1 by 2^(d+1) in parallel
		*		x[k+(2^d+1) -1] += x[k+(2^d)-1] // so we need power and power plus one
		*/
		__global__ void kernel_upsweep(int bufflength, int* data, int power, int power_plus1)
		{
			int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
			
			if (tid >= bufflength)
				return;
			// for depth = 0
			//in[0,1,2,3,4,5,6,7]
			//sums[x,(0+1),x,(2+3),x(4+5),x,(6+7)]
			//move down [0,x,2,x,4,x,6,x]
			// want threads [1,3,5,6] to compute something
			if ( ( tid % power_plus1 ) == 0)
			{
				data[tid+power_plus1-1] += data[tid+power-1];
			}
		}

		/*
		* perform a downsweep after an upsweep
		* for[0,1,2,3,4,5,6,7]
		* after upsweep [0,1,2,6,4,9,6,28]
		* from notes
		* for d = log2n-1 to 0
		*	for all k = 0 to n-1 by 2^(d+1) in par
		*		t = x[k+(2^d)-1]
		*		x[k+(2^d)-1] = x[k+(2^d+1) -1]
		*		x[k+(2^d+1)-1] += t
		*/
		__global__ void kernel_downsweep(int bufflength, int* data, int power, int power_plus1)
		{
			int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

			if (tid >= bufflength)
				return;
			//for depth = 0 or last dpeth on upsweep
			// for[0,1,2,6,4,9,6,28]
			// first set last to 0
			//         [0,1,2,6,4,9,6,0] set initially last element to 0
			// now want[0,1,2,0,4,9,6,6] ( [7-4] + [7] ) then set 7-4 to 0
			if ((tid % power_plus1) == 0)
			{
				int old = data[tid + power - 1]; 
				data[tid + power - 1] = data[tid + power_plus1 - 1]; 
				data[tid + power_plus1 - 1] += old;
			}
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();

			int* dev_data;
			int* dev_out;
			int pow = 0;
			int byte[1] = { 0 };
			int* inc_byte;

			//const int log_n_ceil = ilog2ceil(n);

			//printf("log ceil %d\n", log_n_ceil);
			//const int pow2RoundedSize = 1 << log_n_ceil;
			//printf("log ceil %d\n", pow2RoundedSize);
			
			//const int numbytes_pow2roundedsize = pow2RoundedSize * sizeof(int);
			//const int numbytes_ForCopying = n * sizeof(int);

			// if we have 257 elements we need to account for element 257
			// so we have to do an extra loop log2size will be 512 in this case
			int rounded_depth = ilog2ceil(n);
			int rounded_elements = 1 << rounded_depth;
			dim3 fullBlocksPerGrid = ((rounded_elements + blockSize - 1) / blockSize);

			//int last_in = idata[]
            // need a slightly bigger buffer since if we have 257 elements well go up to 
			// iteration 512
			cudaMalloc((void**)&dev_data, rounded_elements * sizeof(int));
			checkCUDAErrorFn("malloc temp in failed!");

			// init to zero
			//kernel_calloc<< < fullBlocksPerGrid, blockSize >> > kernel_calloc(dev_data, rounded_depth);

			// copy data to device n or n*size? check
			cudaMemcpy(dev_data, idata, rounded_elements * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAErrorFn("copy failed!");
			
			// pad if we need to 
			kernel_padd_0s<< < fullBlocksPerGrid, blockSize >> > (dev_data, n, rounded_elements);

			//cudaMemcpy(inc_byte, &idata[n-1], sizeof(int), cudaMemcpyHostToDevice);
			//checkCUDAErrorFn("copy failed!");

		
			for (int i = 0; i <= rounded_depth-1; i++)
			{
				pow = (1 << i);
				int powplus1 = (1 << (i+1));
				//printf("i %d  -> depth %d \n ", other_pow, pow2);
				kernel_upsweep << < fullBlocksPerGrid, blockSize >> > (rounded_elements, dev_data, pow, powplus1);
				checkCUDAErrorFn("up sweep failed!");
			}

	
			//memory_debug(n, dev_data, odata, idata);

			// write one single byte to the last entry ... 
			cudaMemcpy(&dev_data[n-1], &byte[0], sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAErrorFn("copy last byte failed!");
			//printf("write last byte\n");
			//memory_debug(n, dev_data, odata, idata);
			//printf("starting downsweep\n");
			for (int i = rounded_depth-1; i >= 0; i--)
			{
				pow = (1 << (i));
				int powplus1 = (1 << (i + 1));
				kernel_downsweep << < fullBlocksPerGrid, blockSize >> > (rounded_elements,dev_data,pow,powplus1);
				checkCUDAErrorFn("down sweep failed!");
			}
			//printf("fin downsweep\n");
			//memory_debug(n, dev_data, odata, idata);
			// need to run inclusive to exclusive
			//kernel_inclusive_to_exclusive <<< fullBlocksPerGrid, blockSize >> > (n,dev_data,dev_out,inc_byte);

			printf("fin in to ex\n");
			//memory_debug(n, dev_out, odata, idata);
			cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAErrorFn("copy out failed!");

			cudaFree(dev_data);
			checkCUDAErrorFn("free input failed!");

			//cudaFree(dev_out);
			//checkCUDAErrorFn("free input failed!");

            timer().endGpuTimer();
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            timer().endGpuTimer();
            return -1;
        }
    }
}
