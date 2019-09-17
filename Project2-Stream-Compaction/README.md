CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Grace Gilbert
  * gracelgilbert.com
* Tested on: Windows 10, i9-9900K @ 3.60GHz 64GB, GeForce RTX 2080 40860MB


## Overview
In this project, I implemented the stream compaction algorithm on the GPU in CUDA.  Stream compaction is an algorithm that, given an array of values marked to remove or keep, removes the values and returns a new, shorter array with the values removed. Below is a diagram representing the stream compaction algorithm:

# INCLUDE SC DIAGRAM FROM LECTURE

This algorithm has many practical applications, including path tracing, as it lets us mark certain elements as unwanted and remove them.  While there is a simple way to perform this algorithm using loops on the CPU, it can also be parallelized to be more efficiently performed on the GPU.  I implemented 4 versions of this algorithm, one on the CPU, a naive version on the GPU, a more efficient version on the GPU, and then using the thrust implementation.

An important step in the stream compaction algorithm is the scan algorithm.  This algorithm goes through an array and accumulates additively all of the elements in the array.  An exclusive scan excludes the current index in the accumulated sum, whereas inclusive scan includes the current index.  Steam compaction uses an exclusive scan.  Below is a diagram representing the scan algorithm:

# INCLUDE SCAN DIAGRAM FROM LECTURE

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

## GPU
### Naive
### Efficient
### Thrust

## Performance Analysis

## Questions

