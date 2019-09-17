CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Grace Gilbert
  * gracelgilbert.com
* Tested on: Windows 10, i9-9900K @ 3.60GHz 64GB, GeForce RTX 2080 40860MB


## Overview
In this project, I implemented the stream compaction algorithm on the GPU in CUDA.  Stream compaction is an algorithm that, given an array of values marked to remove or keep, removes the values and returns a new, shorter array with the values removed. Below is a diagram representing the stream compaction algorithm:

# INCLUDE SC DIAGRAM FROM LECTURE

This algorithm has many practical applications, including path tracing, as it lets us mark certain elements as unwanted and remove them.  While there is a simple way to perform this algorithm using loops on the CPU, it can also be parallelized to be more efficiently performed on the GPU.  

An important step in the stream compaction algorithm is the scan algorithm.  This algorithm goes through an array and accumulates additively all of the elements in the array.  An exclusive scan excludes the current index in the accumulated sum, whereas inclusive scan includes the current index.  Steam compaction uses an exclusive scan.  Below is a diagram representing the scan algorithm:

# INCLUDE SCAN DIAGRAM FROM LECTURE

I implemented 4 versions of the above algorithms, scan and stream compaction on the CPU, a naive version of scan on the GPU, a more efficient version of both on the GPU, and then using the thrust implementation of scan.

## CPU
#### Scan
The scan algorithm on the cpu is a simple loop over the data.  For an exclusive scan, we set the first value of the output to 0, as no sum has been accumulated.  Then from index 1 through arrayLength - 1, we set the output at that index to the sum of the output at the previous index and the input array at the previous index:

```
outputData[0] = 0;
for (int k = 1; k < n; ++k) {
    outputData[k] = outoutData[k - 1] + inputData[k - 1];
}
```

#### Stream Compaction without Scan
The basic stream compaction algorithm tracks a counter of how many elements to include that we have seen. If we see an element to include, it sets the output at the index of the counter to the element value, then increments the counter.
```
int counter = 0;
for (int k = 0; k < n; ++k) {
    int currentValue = inputData[k];
    if (currentValue != 0) {
        outputData[counter] = currentValue;
        counter++;
    }
}
```

#### Stream Compaction with Scan and Scatter
In this version, I start by creating a temporary array that contains a 1 wherever the input array had a nonzero value, and 0 where the input array had a zero value:
```
int *tempArray = new int[n];
for (int k = 0; k < n; ++k) {
    tempArray[k] = (int) inputData[k] != 0;
}
```
I then call scan on that 0/1 array using the CPU implementation from above. I then iterate through all the indices in the input array, and if the value should be included, I get the scan result value at that index and put the input value at that scan result index in the output array.
```
int *scanResult = new int[n];
scan(n, scanResult, tempArray);
for (int k = 0; k < n; ++k) {
    if (tempArray[k]) {
        int index = scanResult[k];
        odata[index] = idata[k];
    }
}
```


## GPU
### Naive
### Efficient
### Thrust
For this implementation, I simply cast the input and output array buffers to thrust device pointers and then run thrust's exclusive scane on the buffers.  
```
thrust::exclusive_scan(dev_thrust_inputArray, dev_thrust_inputArray + bufferLength, dev_thrust_outputArray);
```

## Performance Analysis

## Questions

