## Project 2 Part 1 - CUDA Character Recognition
**University of Pennsylvania
CIS 565: GPU Programming and Architecture**

* Author: Chhavi Sharma ([LinkedIn](https://www.linkedin.com/in/chhavi275/))
* Tested on: Windows 10, Intel Core(R) Core(TM) i7-6700 CPU @ 3.40GHz 16GB, 
             NVIDIA Quadro P1000 4GB (MOORE100B-06)

### Index

- [Introduction]( )
- [Algorithm]()

### Introduciton

In this project, we implement GPU based Stream Compaction in CUDA. To aid stream compaction, we also implement various versions of the *Scan* (*Prefix Sum*) algorithm, such as CPU, GPU Naive and GPU Work Efficient versions.
algorithm

Stream compaction, also known as stream filtering or selection, usually produces a smaller output array which contains only the wanted elements from the input array based on some criteria for further processing, while preserving order. For our implementation, We will attemp to remove '0's from an array of ints.

### Algorithms

####  1: CPU Scan & Stream Compaction
 
 We implement stream compaction in two ways:
 
 - CPU based stream compaction without using scan: 
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
 - CPU based stream compaction with CPU based scan: 
   - Compute *Indicator Array* of the input data size that is 1 for non zero elements, an 0 otherwise.
   - Compute *Scan* over indicator Array to get another array. This gives us write indices for the valid elements in the output array. It also gives us the total valid elelemts.
   - *Scatter* data, read from the input array where Indiacator Array is 1, write to the outut array at index given by the scan array.

    ```
     compactWithScan(n, odata, idata) 
        Compute indicator array
        Compute scan
        Scatter
    ```
####  2: Naive GPU Scan Algorithm

####  3: Work-Efficient GPU Scan & Stream Compaction

##### 3.1: Work-Efficient Scan

##### 3.2: Work-Efficient Stream Compaction

####  4: Using Thrust's Implementation


### Questions and Performance Analysis

* **BlockSize Optimization for each Implementation**
  We compare the rumtime of GPU Naive scan and and the work efficient naive scan with the number of threads per block to pick  the most optimal configuration for furhter tests.
  
 *Block Size v/s Runtime*
![](img/BlockSize_vs_Runtime.png)

* **Compare all of these GPU Scan implementations (Naive, Work-Efficient, and
  Thrust) to the serial CPU version of Scan. Plot a graph of the comparison
  (with array size on the independent axis).**
  
   Description.
  
   SCAN with increasing data size
  ![](img/Scan1.png)    
  ![](img/Scan2.png)    
  
   SCAN with increasing data size and nonPowersOf2
  ![](img/Scan1NP.png)    
  ![](img/Scan2NP.png)   
  
  * To guess at what might be happening inside the Thrust implementation (e.g.
    allocation, memory copy), take a look at the Nsight timeline for its
    execution. Your analysis here doesn't have to be detailed, since you aren't
    even looking at the code for the implementation.

* **Write a brief explanation of the phenomena you see here.**

* **Can you find the performance bottlenecks? Is it memory I/O? Computation? Is
    it different for each implementation?**

*  **Paste the output of the test program into a triple-backtick block in your
   README.**
  
    ```
    ****************
    ** SCAN TESTS **
    ****************
        [  31  43   6   3  37   9   8  33  20  31   5   1  48 ...  31   0 ]
    ==== cpu scan, power-of-two ====
       elapsed time: 4.8772ms    (std::chrono Measured)
        [   0  31  74  80  83 120 129 137 170 190 221 226 227 ... 25658550 25658581 ]
    ==== cpu scan, non-power-of-two ====
       elapsed time: 1.6622ms    (std::chrono Measured)
        [   0  31  74  80  83 120 129 137 170 190 221 226 227 ... 25658439 25658476 ]
        passed
    ==== GPU naive scan, power-of-two ====
       elapsed time: 5.3512ms    (CUDA Measured)
        passed
    ==== GPU naive scan, non-power-of-two ====
       elapsed time: 5.20192ms    (CUDA Measured)
        passed
    ==== work-efficient scan, power-of-two ====
       elapsed time: 3.72512ms    (CUDA Measured)
        passed
    ==== work-efficient scan, non-power-of-two ====
       elapsed time: 3.75536ms    (CUDA Measured)
        passed
    ==== thrust scan, power-of-two ====
    Created Thrust pointers
       elapsed time: 0.438272ms    (CUDA Measured)
        passed
    ==== thrust scan, non-power-of-two ====
    Created Thrust pointers
       elapsed time: 0.38912ms    (CUDA Measured)
        passed

    *****************************
    ** STREAM COMPACTION TESTS **
    *****************************
        [   3   1   0   1   1   1   2   3   2   1   1   3   2 ...   1   0 ]
    ==== cpu compact without scan, power-of-two ====
       elapsed time: 2.8238ms    (std::chrono Measured)
        [   3   1   1   1   1   2   3   2   1   1   3   2   3 ...   1   1 ]
        passed
    ==== cpu compact without scan, non-power-of-two ====
       elapsed time: 2.8316ms    (std::chrono Measured)
        [   3   1   1   1   1   2   3   2   1   1   3   2   3 ...   1   3 ]
        passed
    ==== cpu compact with scan ====
       elapsed time: 12.6421ms    (std::chrono Measured)
        [   3   1   1   1   1   2   3   2   1   1   3   2   3 ...   1   1 ]
        passed
    ==== work-efficient compact, power-of-two ====
       elapsed time: 13.7866ms    (CUDA Measured)
        passed
    ==== work-efficient compact, non-power-of-two ====
       elapsed time: 14.1353ms    (CUDA Measured)
        passed
    ```
