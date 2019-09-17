CUDA Stream Compaction
======================

# CUDA Stream Compaction

In the subproject part 2A, we will be comparing different version of GPU scan versions (namely, Naive, Work-Efficent and Thrust) with the serialized CPU version. The metrics for comparison would be the time elapsed for the above algorithms for diferent array sizes.

## 1. CPU Scan

This implementation is the simple serialized scan version of CPU in which we are writing the simple for loop to go all the elements and calculating the exclusive sum of the elements.

Later on, we are using the above scan function to perform the stream compaction where we are removing all the elements with 0 in the array. We also implemented the stream compaction without the scan in implementations.

## 2. GPU Versions

For GPU versions, we implemented three different versions which are mentioned above. The implementation details for each of them is below:

### 2 a) Naive Implementation 

Here, we are doing naive implementation where we are finding the exclusive scan result by iteration over the ceil(log(n)) levels and adding on the elements with stride equal to power of 2 for these levels. Here, we get the final result same as the exclusive scan but we make some paralleilization which reduces the time complexity, but we are performaing more number of additions as compared to the naive CPU scan.

### 2 b) Work-efficient Implementation

In this implementation, we are further trying to optimize by reducing the number of additions to O(n). For performing this, we are assuming the array is the balanced binary tree representation and performing the upsweep and downsweep algorithms for calculating the exclusive scan. In the upsweep part, we are adding th elements by taking the stride of power of 2 when we are going up the levels, which are ceil(log(n)), where log is in the base of 2. 

