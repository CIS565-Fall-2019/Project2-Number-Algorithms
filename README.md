CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Eric Micallef
  * https://www.linkedin.com/in/eric-micallef-99291714b/
  
* Tested on: Windows 10, i5, Nvidia GTX1660 (Personal)

### Performance Analysis

Roughly optimize the block sizes of each of your implementations for minimal run time on your GPU.

Running the naive and work efficient scans and compact algorithms we can see that the best blocksize is 128 or 64. This data was collected using the same array size of 268,435,456 elements

two picture links

(You shouldn't compare unoptimized implementations to each other!)
Compare all of these GPU Scan implementations (Naive, Work-Efficient, and Thrust) to the serial CPU version of Scan. Plot a graph of the comparison (with array size on the independent axis).



We wrapped up both CPU and GPU timing functions as a performance timer class for you to conveniently measure the time cost.
We use std::chrono to provide CPU high-precision timing and CUDA event to measure the CUDA performance.
For CPU, put your CPU code between timer().startCpuTimer() and timer().endCpuTimer().
For GPU, put your CUDA code between timer().startGpuTimer() and timer().endGpuTimer(). Be sure not to include any initial/final memory operations (cudaMalloc, cudaMemcpy) in your performance measurements, for comparability.
Don't mix up CpuTimer and GpuTimer.
To guess at what might be happening inside the Thrust implementation (e.g. allocation, memory copy), take a look at the Nsight timeline for its execution. Your analysis here doesn't have to be detailed, since you aren't even looking at the code for the implementation.
Write a brief explanation of the phenomena you see here.

Can you find the performance bottlenecks? Is it memory I/O? Computation? Is it different for each implementation?
Paste the output of the test program into a triple-backtick block in your README.

-

If you add your own tests (e.g. for radix sort or to test additional corner cases), be sure to mention it explicitly.

![alt text](https://raw.github.com/micallef25/Project2-Number-Algorithms/master/images/boids.png)
