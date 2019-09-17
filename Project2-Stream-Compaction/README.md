CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Saket Karve
  * [LinkedIn](https://www.linkedin.com/in/saket-karve-43930511b/), [twitter](), etc.
* Tested on:  Windows 10 Education, Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz 16GB, NVIDIA Quadro P1000 @ 4GB (Moore 100B Lab)

### Description

The scan algorithm is implemented using different approaches starting from a CPU version, then paralellizing it using a Naive parallel implementation and then optimizing it on the GPU. A performance analysis is included here showing a comparison between the different approaches. The thrust version of scan is also included in the comparisons.

The scan algorithm is further used to implement stream compaction. The CPU version and the optimize GPU version is compared for performance.

### Scan Algorithm - Performance Analysis

Scan algorithm is used for finding the prefix-sum of an array of numbers. Each element of the output is the sum of all the previous elements with (or without) the elemnt at that position in the inclusive (or exclusive) scan. For stream compaction, exclusive scan is needed. Hence all comparisons are made for algorithms computing the exclusive scan of the input array.

Performance is measured in terms of the time (in milliseconds) required to execute the algorithm.

There are two main parameters used for comparison namely,
- Block Size
- Size of the input array

The performance is compared for the following approaches of the scan algorithm with respect to the above parameters.
- CPU implementation
- Naive Parallel implementation
- Work-efficient parallel implementation
- Optimized Work-efficient parallel implementation \[EXTRA CREDIT\]
- Thrust's implementation

#### Performance against Block Size

The block size matters only for the Naive parallel implementation and the Work-efficient implementations. 

The following graph shows how the performance varies with increasing block size on the three parallel implementations.

![](img/scan_blocksize_20.PNG)

It can be seen that the performance more or less remains constant with block size. For all the implementations, block size of 128 is optimal.

I also tested the performance for the work-efficient implementation (optimized and unoptimized) for array sizes set to 2^10 and 2^20. The plots were as follows,

![](img/scan_blocksize_both.PNG)

A significant drop can be seen in the runtime when the array size is reduced. But, no specific trend is observer between array size and block size. This is probably because we are not using the shared memory.

Hence, for testing against array size, block size is set to 128 wherever applicable.

#### Performance against Array Size

Performance of the CPU implementation with all the GPU implementations can be found in the following figure.

![](img/array_size_cpu_all.PNG)

As expected, the performance decreases with increasing array size. The naive GPU and work efficient GPU impelemntation can be found to perform worse than the CPU implementation for smaller array sizes. This is primarily because of the extra overhead in managing the threads on the GPU which overpowers the effect of parallelization.

The following graph shows a comparison between the optimized work efficient implementation with the regular implementation.

![](img/scan_array_size_optimized_unoptimized.PNG	)

We can see that the CPU implementation is better than the GPU implementation for smaller array sizes. However, as the size of the array increases, CPU implementation takes more time and the optimized work efficient implementation performs the best. This can be seen from the following graph.

![](img/scan_array_size_bar.PNG)

**Optimized Work-efficient implementation

The optimized version of the work-efficient scan implementation is done by taking into account the number of idle (or extra) threads launched during each subsequent implementation of the algorithm. The way the algorithm is implemented, the number of active threads is halved every iteration. So to optimize the regular implementation, I launched only as many threads as needed for that iteration. The indices were then updated as per the stride which makes the overhead in managing the threads much simpler. This accounts for the performance improvement.

The work efficient implementation is applicable only when the size of the array is a power of two. So if the given array has a size which is not a power of two, then the input array is padded with zeros to make the size a power of two. The performance does not matter with respect to this parameter since eventually it is always expanded to a power of two and will give a perfomance equivalent to that with the next power of two.

### Stream Compaction Algorithm - Performance Analysis

Stream compaction is used to eliminate elements from an array based on some condition. The exclusive scan of the input array is used to do stream compaction. 

Performance is measured in terms of the time (in milliseconds) required to execute the algorithm.

There are two main parameters used for comparison namely,
- Block Size
- Size of the input array

The performance is compared for the following approaches of the scan algorithm with respect to the above parameters.
- CPU implementation (with and without scan)
- Work-efficient parallel implementation
- Optimized Work-efficient parallel implementation \[EXTRA CREDIT\]

#### Performance against Block Size

The block size matters only for the Naive parallel implementation and the Work-efficient implementations. 

The following graph shows how the performance varies with increasing block size on the two parallel implementations.

![](img/sc_block_size.PNG)

It can be seen that the performance more or less remains constant with block size. For all the implementations, block size of 128 is optimal.

Hence, for testing against array size, block size is set to 128 wherever applicable.

#### Performance against Array Size

There are two CPU implementations compared in this section. One uses the scan to do stream compaction and the other simply loops over the input array and stores the required elements in a separate buffer.

A comparison between the two CPU implementations can be found in the following figure.

![](img/sc_array_size_1.PNG)

Performance of the CPU implementation with all the GPU implementations can be found in the following figure.

![](img/sc_array_size_2.PNG)

As expected, the performance decreases with increasing array size. The naive GPU and work efficient GPU impelemntation can be found to perform worse than the CPU implementation for smaller array sizes. This is primarily because of the extra overhead in managing the threads on the GPU which overpowers the effect of parallelization.

The following graph shows a comparison between the optimized work efficient implementation with the regular implementation.

![](img/sc_array_size_3.PNG)

There is a significant improvement in performance with the optimized version of the work efficient scan for stream compaction. The regular implementation is most of the times worse than the CPU implementation.
