CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Jiangping Xu
  * [LinkedIn](https://www.linkedin.com/in/jiangping-xu-365b19134/)
* Tested on: Windows 10, i7-4700MQ @ 2.40GHz 8GB, GT 755M 6100MB (personal laptop)
_________________________________________________________________________
[Introduction](#Stream-Compaction) - [Performance Analysis](#performance-analysis) - [Questions](#questions)
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


