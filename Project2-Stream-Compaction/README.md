CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Grace Gilbert
  * gracelgilbert.com
* Tested on: Windows 10, i9-9900K @ 3.60GHz 64GB, GeForce RTX 2080 40860MB


## Overview
In this project, I implemented the stream compaction algorithm on the GPU in CUDA.  Stream compaction is an algorithm that, given an array of values marked to remove or keep, removes the values and returns a new, shorter array with the values removed.  This algorithm has many practical applications, including path tracing, as it lets us mark certain elements as unwanted and remove them.  While there is a simple way to perform this algorithm using loops on the CPU, it can also be parallelized to be more efficiently performed on the GPU.  I implemented 4 versions of this algorithm, one on the CPU, a naive version on the GPU, a more efficient version on the GPU, and then using the thrust implementation. 

## CPU

## GPU
### Naive
### Efficient
### Thrust

## Performance Analysis

## Questions

