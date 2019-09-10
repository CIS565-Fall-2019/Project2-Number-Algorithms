#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
	        static PerformanceTimer timer;
	        return timer;
        }
	
	// since used in compact scan
	static void scan( int n, int* odata, const int *idata )
	{
		//from notes:
		// in [3,1,4,7 ,0,4,1,6,3]
		// out[0,3,4,8,15,15,19,20,26] 
		// itr1 odata[1] = idata[0] + odata[0]; 3
		// itr2 odata[2] = idata[1] + odata[1]
		odata[0] = 0;
		for(int i = 1; i < n; i++)
		{
			odata[i] = idata[i-1] + odata[i-1];
		}
	}
        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
            	// TODO
		scan( n, odata, idata );
	        timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
            	// TODO
		//from notes:
		// in [3,0,4,0,0,4,0,6,0]
		// out[3,4,4,6] or [3,7,11,17]?
		odata[0] = 0;
		int writer = 0;
		for(int i = 0; i < n; i++)
		{
			if( idata[i] != 0 )
			{
				odata[writer] = idata[i];
				writer++;
			}
		}
	        timer().endCpuTimer();
            return writer;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
	// in [3,0,4,0,0,4,0,6,0]
	// create [1,0,1,0,0,1,0,1,0] 
	// after scan[1,1,2,2,2,3,3,4,4]  // 4 elements these are the indexes to where the data is
	// scatter?
	//Result of scan is index into final array
	//Only write an element if temporary array has a 1
	//Write index is given by scan
	//scatter out [3,4,4,6] return 4
        int compactWithScan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
	        // TODO
		// create the 1,0 buffer
		int* buff = new int[n];
		int i = 0;
		int rval = 0;
		
		// have [3,0,4,0,0,4,0,6,0]
		// create [1,0,1,0,0,1,0,1,0] 
		for(i = 0; i < n; i++)
		{
			if(idata != 0)
			{
				buff[i] = 1;
			}
			else
			{
				buff[i] = 0;
			}
		}
		// scan this buffer now to figure out the indexes
		// have [1,0,1,0,0,1,0,1,0]
		// create scan[1,1,2,2,2,3,3,4,4] 
		int* scan_buff = new int[n];
		scan( n, scan_buff, buff ); // #el, out, in
		
		// have scan[1,1,2,2,2,3,3,4,4] index to where we should place output
		// have input[3,0,4,0,0,4,0,6,0]
		//create[3,4,4,6]
		for(i = 0; i < n; i++)
		{
			if(buff[i] == 1) // marked as data
			{
				odata[scan_buff[i]] = idata[i]; 
				rval = scan_buff[i]; // how many elements do we have
			}
		}
		
	        timer().endCpuTimer();
	    //
	    delete scan_buff;
	    delete buff;
		
            return (rval +1);
        }
    }
}
