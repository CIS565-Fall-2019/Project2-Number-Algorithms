## Project 2 Part 1 - CUDA Character Recognition
**University of Pennsylvania
CIS 565: GPU Programming and Architecture**

* Author: Chhavi Sharma ([LinkedIn](https://www.linkedin.com/in/chhavi275/))
* Tested on: Windows 10, Intel Core(R) Core(TM) i7-6700 CPU @ 3.40GHz 16GB, 
             NVIDIA Quadro P1000 4GB (MOORE100B-06)

![Some IMage]()

### Index

- [Introduction]( )
- [Algorithm]()

### Introduciton

In this project, we implement GPU based Stream Compaction in CUDA. To aid stream compaction, we also implement various versions of the *Scan* (*Prefix Sum*) algorithm, such as CPu, GPU Naive and GPU Work Efficient versions.
algorithm

Stream compaction, also known as stream filtering or selection, usually produces a smaller output array which contains only the wanted elements from the input array based on some criteria for further processing, while preserving order. For our implementation, We will attemp to remove '0's from an array of ints.

### Algorithms
We implement stream compaction in two ways:
 - Naive CPU based stream compaction: 
   - Loop over the input data array
   - Copy non-zero elements to output array
   - count copies to track size
 
   ```
   compactWithoutScan(N, Odata, Idata)
     if n > 0
       int size=0;
       for i in Idata
           if (idata[i] != 0) 
              Odata[counter] = Idata[i]
              size+=1
       return size
   ```


### Implementations
