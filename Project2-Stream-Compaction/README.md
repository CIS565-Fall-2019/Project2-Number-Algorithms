CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Disha Jindal: [Linkedin](https://www.linkedin.com/in/disha-jindal/)
* Tested on: Windows 10 Education, Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz 16GB, GTX 222 222MB (Moore 100B Lab)

## Description
Stream Compaction is a very widely used algorithm with path tracer as one of the applications. This project implements GPU stream compaction in CUDA and compares the following algorithms for stream compaction along with analyzing the scan module:
1. CPU  
2. GPU: Naive
3. GPU: Work-Efficient
4. GPU: Optimized Work-Efficient
5. GPU: Thrust

## Performance Analysis
### Array Size
#### Scan
Scan is the most important module in stream compaction. The following plots show the performance of above mentioned five algorithms with varying array size. Two variations considered are number of elements in powers of two and not in powers of two.  

<p align="center"><img src="https://github.com/DishaJindal/Project2-Number-Algorithms/blob/master/Project2-Stream-Compaction/img/Scan_POT_ArraySize.png" width="430"/> <img src="https://github.com/DishaJindal/Project2-Number-Algorithms/blob/master/Project2-Stream-Compaction/img/Scan_NPOT_ArraySize.png" width="430"/></p>

**Analysis** - The difference between different implementations is clearly visible for large array size starting around two rasise to power 18. As we can see from the above plots that the scan on the CPU is performing the worst which is expected. The performance of Naive GPU approach and Work Efficient is almost the same with work efficient being slighly better which is because of the extra overhead in the work efficient approach. Thurst Implementation is better than these three. But the optimized version of work efficient scan in which I am omitting all unnecessary threads and keep only the ones required to do the computation in each iteration of Up Sweep and Down Sweep. Due to these optimizations, this approaching is taking the least amount of time of all. A similar trend can be seen in non power of two. 
#### Stream Compaction
The following plots show the performance of different stream compaction algorithms with array size. Two variations considered are number of elements in powers of two and not in powers of two. 
<p align="center"><img src="https://github.com/DishaJindal/Project2-Number-Algorithms/blob/master/Project2-Stream-Compaction/img/Compaction_ArraySize.png" width="430"/> <img src="https://github.com/DishaJindal/Project2-Number-Algorithms/blob/master/Project2-Stream-Compaction/img/Compaction_NPOT_ArraySize.png" width="430"/></p>

**Analysis** - CPU with scan is performing the worst because of no parrallism. Followed by this is the Work efficient approach. Naive Implementation with GPU is omiitted beacuse it performed poorly in scan. It is interesting to see the sudden performance drop of Work Efficient GPU algorithm on non power of two variant after a certain point. But it performs better than the CPU approach for the most part because of the parallelism and runtime of log n as compared to n with the CPU approach. Optimized approach further improves the performance. The CPU implementation without scan is performing the best. One of the benefits of the GPU approach is that it compacts in place.

### Block Size
Next, I did the performance analysis of both scan and stream compaction for different block sizes. 
#### Scan
The following plot shows the performance of three GPU implementation with varying block sizes on the SCAN module.
<p align="center"><img src="https://github.com/DishaJindal/Project2-Number-Algorithms/blob/master/Project2-Stream-Compaction/img/Scan_BlockSize.png" width="800"/></p>

**Analysis** - The trend is same for all implementations. As we increase the block size till 32, the performance increases because the warp size is 32 so keep the block size less than 32 is not optimal. The performance does not change on increasing the blocksize after 32. 

#### Stream Compaction
The following plot shows the performance of two GPU implementation with varying block sizes on the stream compaction.

<p align="center"><img src="https://github.com/DishaJindal/Project2-Number-Algorithms/blob/master/Project2-Stream-Compaction/img/Compaction_BlockSize.png" width="800"/> </p>

**Analysis** - The impact of the block size on the stream compaction performance is same as that the one on scan. The performance increases on increasing the blocksize till 32 and stays almost constant after that.
## Output
### Scan
Following is the output of the main program for SCAN module showing the arrays and runtime of various algorthims (including power and two and not power of two variants) already given with an addition of optimized work efficient approach.
```
****************
** SCAN TESTS **
****************
    [   2  20  35  39  13   6  16   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 0.0005ms    (std::chrono Measured)
    [   0   2  22  57  96 109 115 131 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 0.0002ms    (std::chrono Measured)
    [   0   2  22  57  96 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 0.01856ms    (CUDA Measured)
    [   0   2  22  57  96 109 115 131 ]
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 0.017408ms    (CUDA Measured)
    [   0   2  22  57  96   0   0   0 ]
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 0.072192ms    (CUDA Measured)
    [   0   2  22  57  96 109 115 131 ]
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 0.057344ms    (CUDA Measured)
    [   0   2  22  57  96 ]
    passed
==== optimized work-efficient scan, power-of-two ====
   elapsed time: 0.070112ms    (CUDA Measured)
    [   0   2  22  57  96 109 115 131 ]
    passed
==== optimized work-efficient scan, non-power-of-two ====
   elapsed time: 0.036864ms    (CUDA Measured)
    [   0   2  22  57  96 ]
    passed
==== thrust scan, power-of-two ====
   elapsed time: 5.54496ms    (CUDA Measured)
    [   0   2  22  57  96 109 115 131 ]
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 1.39277ms    (CUDA Measured)
    [   0   2  22  57  96 ]
    passed
```

### Stream Compaction
Following is the output of the main program for Stream Compaction showing the arrays and runtime of various algorthims (including power and two and not power of two variants) already given with an addition of optimized work efficient approach.
```
*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   0   0   3   3   3   2   0   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 0.0005ms    (std::chrono Measured)
    [   3   3   3   2 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 0.0002ms    (std::chrono Measured)
    [   3   3   3 ]
    passed
==== cpu compact with scan ====
   elapsed time: 0.1199ms    (std::chrono Measured)
    [   3   3   3   2 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 0.39424ms    (CUDA Measured)
    [   3   3   3   2 ]
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 0.270848ms    (CUDA Measured)
    [   3   3   3 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 0.33216ms    (CUDA Measured)
    [   3   3   3   2 ]
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 0.282624ms    (CUDA Measured)
    [   3   3   3 ]
    passed
```

## Extra Credits
### Optimized Work Efficient Approach
I analyzed the performance of Work Efficient GPU approach. The occupancy of the threads keeps on dropping by a factor of two with each iteration of up and down sweep. I implemented the optimized variant of Work Efficient Approach to make sure that all threads are occupied in each iteration. It makes sure that It only lanuches the required number of threads in each iteration rather than launchng the fixed number of threads in each iteartion. It improved the performance significantly in scan as well as stream compaction, it can be seen in the plots above. 
