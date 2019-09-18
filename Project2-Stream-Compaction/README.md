CUDA Stream Compaction
======================

In the subproject part 2A, we will be comparing different version of GPU scan versions (namely, Naive, Work-Efficent and Thrust) with the serialized CPU version. The metrics for comparison would be the time elapsed for the above algorithms for diferent array sizes.

## 1. CPU Scan

This implementation is the simple serialized scan version of CPU in which we are writing the simple for loop to go all the elements and calculating the exclusive sum of the elements.

Later on, we are using the above scan function to perform the stream compaction where we are removing all the elements with 0 in the array. We also implemented the stream compaction without the scan in implementations.

## 2. GPU Versions

For GPU versions, we implemented three different versions which are mentioned above. The brief implementation details for each of them is below:

### 2 a) Naive Implementation 

Here, we are doing naive implementation where we are finding the exclusive scan result by iteration over the `ceil(log(n))` levels and adding on the elements with stride equal to power of 2 for these levels. Here, we get the final result same as the exclusive scan but we make some paralleilization which reduces the time complexity, but we are performing more number of additions as compared to the naive CPU scan.

### 2 b) Work-efficient Implementation and Stream Compaction Using Work Efficient Scan

In this implementation, we are further trying to optimize by reducing the number of additions to `O(n)`. For performing this, we are assuming the array is the balanced binary tree representation and performing the upsweep and downsweep algorithms for calculating the exclusive scan. Note all the calculations taking place here is in place as we don't have to worry about the possible race conditions in this scenario.

In the upsweep part, we are adding th elements by taking the stride of power of 2 when we are going up the levels, which are `ceil(log(n))`, where `log` is in the base of 2. 

In the downsweep part, we are putting the last element as zero and then updating the array assuming it as the balanced binary tree again. So, we are creating the left node and right node childs in the stride of power of 2 coming from high to low. So, we are putting the value in left child as the current value and updating the right child node as the sum of left node and right node child. We continue doing these for `ceil(log(n))` levels. The resulting array would be the exclusive scan on the input array.

In stream compaction part, we are using the above work-efficient scan implementation. Before using the fucntion, we are doing boolean mapping to ensure that all the non-zero values are 1 so that they can de easily differentialted with 0. After performing the scan along with the boolean data to create the unique indices for the compaction output, we are performing scatter in which we are taking the indices value whose boolean mapped array value is 1 and putting that value which is in original array into the new output array with the unique indices value from the scanned output.  

### 2 c) Thrust implementtion

For comparing our alogirhtms with the exclusive scna implementation thrust library uses, we are using the exclusive scan implementation of the thrust. We created a wrapper function in which we are initializing the device pointers to the input data and using these pinters to create the exlcusinve scan and copying the output to the pointer located in the host memory.

## 3. Performance Analysis

In the performance analysis, we are first checking the best blockSize value for Naive and Work-Efficient algorithms. The plot for the analysis of the time elapsed vs Block size for both these implementations are given below for the data size at `2^20` (Note that the analysis is for power of two number of elements):

<p align="center"><img src="https://github.com/somanshu25/Project2-Number-Algorithms/blob/master/Project2-Stream-Compaction/img/Block_Size_vary.png" width="800"/></p>


For the above, we see that the optimized implementations are when the Block size is 512 for both the algorithms. For checking how all of the versions of GPU compare with CPU version when varying the input size, I have broken the ployts into 2 plots as shown below. The first plot on the top is from input size `2^10` to `2^18` while the second plot on the bottom varies from `2^19` to `2^25` for better visualization of the plots and see how the algorithms are working for larger range of the input array (Note that the analysis is for power of two number of elements).

<p align="center"><img src="https://github.com/somanshu25/Project2-Number-Algorithms/blob/master/Project2-Stream-Compaction/img/Plot_Scan_1.png" width="800"/></p>

<p align="center"><img src="https://github.com/somanshu25/Project2-Number-Algorithms/blob/master/Project2-Stream-Compaction/img/Plot_Scan_2.png" width="800"/></p>

## 4. Questions and Inferences

#### Q1. Write a brief explanation of the phenomena you see here?

From the above analysis, we observe that the performance of all the versions vary with the array input size. Initially, with the less input size array, we see CPU scan implementation being faster than the rest and for some of very small input sizes, Thrust implementation also takes more time. As the array size incrases, we see the curver getting changed and word-efficeint GPU version becoming better as compared to CPU, which we observe in the second plot where the curves start diverging with GPU-naive taking the most time following Serial-CPU, GPU-Work Efficient and Thrust Implementation is taking the minimum time as its the most optimized one. The reason for these graphs is as follows
  
  * As input size increases, GPU-Naive is doing more additions (which are doubling), hence, the performance is taking the hit.
  * As input size increases, GPU-Work Efficient time less complexity and less additions take the precedence over Serial-CPU and Naive-       GPU and starts performing better.

#### Can you find the performance bottlenecks? Is it memory I/O? Computation? Is it different for each implementation?

Here, we can observe that the bottleneck for Work-efficient GPU stream compaction is the number of cuda Malloc we need to do when we are calling the Work-efficient scan. That is one of the unnecessary memory copy in and out which is the reason for initial bottleneck in the algoirthm with respect to the CPU non-scan version one. For the case of naive, the performance is more due to the umber of threads being called and the increase in the number of additions which is of order `O(nlog(n))`. It was also observed that when input array is greater than 2^17, then the GPU-work efficient compaction becomes better than the CPU with scan compaction.  

#### Paste the output of the test program into a triple-backtick block in your README.

The output of the time elapsed for input size `2^20` data and blockSize of 512 for both the implementations of GPU-Naive and GPU-Work Efficient is below (Note that the analysis is for power of two number of elements):
```
****************
** SCAN TESTS **
****************
    [   3  30  38  42  45  21  37  46  38  43   4  40  36 ...   5   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 4.7234ms    (std::chrono Measured)
    [   0   3  33  71 113 158 179 216 262 300 343 347 387 ... 25656958 25656963 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 1.7361ms    (std::chrono Measured)
    [   0   3  33  71 113 158 179 216 262 300 343 347 387 ... 25656875 25656913 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 5.15686ms    (CUDA Measured)
    [   0   3  33  71 113 158 179 216 262 300 343 347 387 ... 25656958 25656963 ]
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 5.21149ms    (CUDA Measured)
    [   0   3  33  71 113 158 179 216 262 300 343 347 387 ...   0   0 ]
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 4.29978ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 3.95059ms    (CUDA Measured)
    [   0   3  33  71 113 158 179 216 262 300 343 347 387 ... 25656875 25656913 ]
    passed
==== thrust scan, power-of-two ====
   elapsed time: 0.32256ms    (CUDA Measured)
    [   0   3  33  71 113 158 179 216 262 300 343 347 387 ... 25656958 25656963 ]
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 0.346112ms    (CUDA Measured)
    [   0   3  33  71 113 158 179 216 262 300 343 347 387 ... 25656875 25656913 ]
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   1   2   2   2   3   1   1   0   0   3   2   0   0 ...   3   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 2.9368ms    (std::chrono Measured)
    [   1   2   2   2   3   1   1   3   2   3   2   2   1 ...   3   3 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 2.6901ms    (std::chrono Measured)
    [   1   2   2   2   3   1   1   3   2   3   2   2   1 ...   1   2 ]
    passed
==== cpu compact with scan ====

   elapsed time: 12.4664ms    (std::chrono Measured)
    [   1   2   2   2   3   1   1   3   2   3   2   2   1 ...   3   3 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 11.4031ms    (CUDA Measured)
    [   1   2   2   2   3   1   1   3   2   3   2   2   1 ...   3   3 ]
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 11.9585ms    (CUDA Measured)
    [   1   2   2   2   3   1   1   3   2   3   2   2   1 ...   1   2 ]
    passed
```
