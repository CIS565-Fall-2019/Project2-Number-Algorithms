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

		void dev_scan(int elements,int rounded_depth,int rounded_elements,int* dev_data,dim3 blocks)
		{
			int pow = 0;
			int powplus1 = 0;
			int byte[1] = { 0 };
			
			for (int i = 0; i <= rounded_depth - 1; i++)
			{
				pow = (1 << i);
				powplus1 = (1 << (i + 1));
				//printf("i %d  -> depth %d \n ", other_pow, pow2);
				kernel_upsweep << < blocks, blockSize >> > (elements, dev_data, pow, powplus1);
				checkCUDAErrorFn("up sweep failed!");
			}


			//memory_debug(n, dev_data, odata, idata);

			// write one single byte to the LAST entry ... even if it was rounded and you just padded  
			cudaMemcpy(&dev_data[rounded_elements - 1], &byte[0], sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAErrorFn("copy last byte failed!");
			//printf("write last byte\n");
			//memory_debug(n, dev_data, odata, idata);
			//printf("starting downsweep\n");
			for (int i = rounded_depth - 1; i >= 0; i--)
			{
				pow = (1 << (i));
				powplus1 = (1 << (i + 1));
				kernel_downsweep << < blocks, blockSize >> > (rounded_elements, dev_data, pow, powplus1);
				checkCUDAErrorFn("down sweep failed!");
			}
		}
	
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
         
			int* dev_data;
			int pow = 0;
			int byte[1] = { 0 };

			// if we have 257 elements we need to account for element 257
			// so we have to do an extra loop log2size will be 512 in this case
			int rounded_depth = ilog2ceil(n);
			int rounded_elements = 1 << rounded_depth;
			dim3 fullBlocksPerGrid((rounded_elements + blockSize - 1) / blockSize);

			// need a slightly bigger buffer since if we have 257 elements well go up to 
			// iteration 512
			cudaMalloc((void**)&dev_data, rounded_elements * sizeof(int));
			checkCUDAErrorFn("malloc temp in failed!");

			cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAErrorFn("scan idata copy failed!");

			// pad if we need to 
			kernel_padd_0s << < fullBlocksPerGrid, blockSize >> > (dev_data, n, rounded_elements);

			timer().startGpuTimer();
			// run the actual work efficient algorithm
			dev_scan(n, rounded_depth, rounded_elements, dev_data, fullBlocksPerGrid);
			
			timer().endGpuTimer();

			cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAErrorFn("copy out failed!");

			cudaFree(dev_data);
			checkCUDAErrorFn("free input failed!");
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
            // TODO
			int* dev_boolbuff;
			int* dev_map;
			int* dev_out;
			int* dev_in;

			int rounded_depth = ilog2ceil(n);
			int rounded_elements = 1 << rounded_depth;
			dim3 fullBlocksPerGrid((rounded_elements + blockSize - 1) / blockSize);

			//int last_in = idata[]
			// need a slightly bigger buffer since if we have 257 elements well go up to 
			// iteration 512
			cudaMalloc((void**)&dev_boolbuff, rounded_elements * sizeof(int));
			checkCUDAErrorFn("malloc dev_boolbuff in failed!");
			cudaMalloc((void**)&dev_map, rounded_elements * sizeof(int));
			checkCUDAErrorFn("malloc dev_map in failed!");
			cudaMalloc((void**)&dev_out, n * sizeof(int));
			checkCUDAErrorFn("malloc dev_out in failed!");
			cudaMalloc((void**)&dev_in, rounded_elements * sizeof(int));
			checkCUDAErrorFn("malloc dev_in in failed!");

			cudaMemcpy(dev_in, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAErrorFn("dev_in copy failed!");

			// pad if we need to 
			kernel_padd_0s << < fullBlocksPerGrid, blockSize >> > (dev_in, n, rounded_elements);


			timer().startGpuTimer();
			// stores 1s and zeros in the boolbuffer
			// have [3,0,4,0,0,4,0,6,0]
			// create [1,0,1,0,0,1,0,1,0] 
			StreamCompaction::Common::kernMapToBoolean << < fullBlocksPerGrid, blockSize >> > (n, dev_boolbuff, dev_in);

			// need to retain this bool data for the kernscatter
			cudaMemcpy(dev_map, dev_boolbuff, n * sizeof(int), cudaMemcpyDeviceToDevice);
			checkCUDAErrorFn("dev_map copy failed!");

			// dev scan takes in device buffers create our map
			// have [1,0,1,0,0,1,0,1,0]
			// create scan[1,1,2,2,2,3,3,4,4] 
			dev_scan(n,rounded_depth,rounded_elements, dev_map, fullBlocksPerGrid);

			// have scan[1,1,2,2,2,3,3,4,4] index to where we should place output
			// have input[3,0,4,0,0,4,0,6,0]
			//API calls for:
			// device  output buffer
			// device original input
			// device array of bools
			// device map created by scan
			StreamCompaction::Common::kernScatter << < fullBlocksPerGrid, blockSize >> > (n, dev_out, dev_in, dev_boolbuff, dev_map);


			timer().endGpuTimer();
			// we need to read the last elements from our map and our the bool buff.
			// the map will tell us how many elements but is an exclusive scan so
			// we need to read the last element of the bool array to see if it contains a 1 or 0
			int last_bool;
			int last_map;
			int scatter_size = 0;

			cudaMemcpy(&last_bool, &dev_boolbuff[n-1], sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAErrorFn("copy last_bool failed!");
			cudaMemcpy(&last_map, &dev_map[n-1], sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAErrorFn("copy last_map failed!");

			scatter_size = last_bool + last_map;

			cudaMemcpy(odata, dev_out, scatter_size * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAErrorFn("copy out failed!");

			cudaFree(dev_boolbuff);
			cudaFree(dev_map);
			cudaFree(dev_out);
			cudaFree(dev_in);

            return scatter_size;
        }
    }
}
