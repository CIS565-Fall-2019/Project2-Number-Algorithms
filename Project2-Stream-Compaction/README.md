CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Jiangping Xu
  * [LinkedIn](https://www.linkedin.com/in/jiangping-xu-365b19134/)
* Tested on: Windows 10, i7-4700MQ @ 2.40GHz 8GB, GT 755M 6100MB (personal laptop)
_________________________________________________________________________
[Introduction](#Stream-Compaction) - [Performance Analysis](#performance-analysis) - [Questions](#questions) - [Output](#output)
_________________________________________________________________________
## Introduction
The goal of stream compaction is to remove the redundent elements in a given array. In this project, I implemented several stream compaction algorithms either in CPU or GPU. The main steps are basically the same for these implementations. First, map the input array to a boolean array where 1(true) stands for useful elements and 0(false) stands for the elements need to be removed. Then a prefix-sum scan is performed to find the indices of the elements we keep in the output array. Finally scatter the elements based on the boolean array and the index array to get the final result.

All features are as follows:
* A CPU version of scan algorithm (a simple for loop), with a CPU version of stream compation that based on scan.
* A CPU version of stream compaction without using the scan function
* Naive GPU Scan Algorithm (see [Example 39-1](https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html))
* Work-Efficient GPU Scan (see [Section 39.2.2](https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html))
* A GPU scatter funtion and a mapping to boolean array function that are used by both of the GPU scan algorithm.
* A GPU scan using Thrust library.

## Performance Analysis
__Find The Optimal Block Size__
<p align="center">
    <img src = img/ScanTimeCostWithIncreasingBlockSize.png>
</p>

First the performances of naive and work efficient method with different block sizes are tested. Two different input array sizes are considered: a power of two length (256) and a non-power of two length (253). From the graph above we notice that the performance of naive gpu scan doesn't change much along with the change of block size, while the work efficient scan seems to achieve the best performance with a block size of 512. Block sizes for naive and work efficent scan will both set to 512 in the following tests.

__Comparison__
<p align="center">
    <img src = img/ScanTimeCostWithIncreasingInputArrayLength.png>
</p>
We can see that the methods with better performance are CPU and thrust scan. See the next section for a more detailed discussion.

## Questions
### Can you find the performance bottlenecks? Is it memory I/O? Computation? Is it different for each implementation?

`CPU`: the computation is the bottle neck while memory access is fast.

`GPU Naive`: theoretically it should be faster than the CPU scan. Although the total number of addition ( O(nlog(n)) ) is much more than a simple CPU loop (n - 1), multiple threads compute at the same time. In the best situation, only O(log(n)) time is needed. But the memory access speed may become the bottle neck for GPU scan. Reading and writing global memory is really time consuming. Besides, the naive gpu implementation above needs further optimization. Warp divergence and bank conflict also slow down the performance.

`Work Efficient`: this algorithm requires O(n) additions to scan an array. When the array length goes up, theoratically it beats the naive GPU scan. But it also suffers from the problems mentioned above.

`Thrust`: warp partition is less occured in this implememntation. I guess it may also use the share memory to speed up memory access. 

## Output
blockSize = 512, ArraySize = 256 / 253
```
****************
** SCAN TESTS **
****************
    [   9  35  14  13  10  25  14  25  43   0   7  36  26 ...  38   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 0.0008ms    (std::chrono Measured)
    [   0   9  44  58  71  81 106 120 145 188 188 195 231 ... 6032 6070 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 0.0008ms    (std::chrono Measured)
    [   0   9  44  58  71  81 106 120 145 188 188 195 231 ... 5983 5998 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 0.05296ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 0.0496ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 0.092896ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 0.090368ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 0.093824ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 0.092896ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   1   1   2   1   2   3   0   3   1   2   3   2   2 ...   2   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 0.0013ms    (std::chrono Measured)
    [   1   1   2   1   2   3   3   1   2   3   2   2   2 ...   2   2 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 0.0011ms    (std::chrono Measured)
    [   1   1   2   1   2   3   3   1   2   3   2   2   2 ...   1   2 ]
    passed
==== cpu compact with scan ====
   elapsed time: 0.0049ms    (std::chrono Measured)
    [   1   1   2   1   2   3   3   1   2   3   2   2   2 ...   2   2 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 0.222208ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 0.157696ms    (CUDA Measured)
    passed
```

