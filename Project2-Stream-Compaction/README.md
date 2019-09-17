CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Eric Micallef
  * https://www.linkedin.com/in/eric-micallef-99291714b/
  
* Tested on: Windows 10, i7-6700 @ 3.4GHz 16GB, Nvidia Quadro P1000 (Moore 100B Lab)

From the data gathered we can see that the ideal block size is 128 for both Naive and Work Efficient Scans and compact algorithms.
a block size of 32 drastically increased the run time of the algorithms where as more blocks made the algorithms slower. My guess is that because so many threads are not active 128 and 64 seem to be a happy place for hiding some memory latency that occurs.



![](img/differingblocksraw.PNG)

![](img/differingblocks.PNG)

When comparing the performances of the naive, work efficient and cpu implementations we can see that CPU scans are very quick. once we get to about 32k the compact algorithm on the CPU slows down quite a bit but performing scans on the CPU is significanlty faster than on a GPU. This could be because for the CPU scan we are getting nice sequential read access where as in the work efficient algorithm we are getting random access memory patterns which causes some bottlenecks. In the graph with the larger arrays we see a similar pattern forming where the cpu compact is terribly slow in comparison to the GPU version. during these higher array sizes the GPU implementation starts to become slightly better than the CPU.

In all implementations thrust performed poorly. 

Although the naive implementation has more work to do we see that the times are not significantly higher than that of the work efficient algorithm. This could be because in the work efficient algorithm we have manyy warps  with jusst one thread active and the memory patttererns are random causing for less latency hiding. The naive version is a bit more sequential so although more work is being done we may be able to hide a bit more latency.

![](img/graph_raw.PNG)

![](img/larger_graph.PNG)

![](img/smaller_graph.PNG)

### Results

![](img/moderate_test_result.PNG)

![](img/large_test_result.PNG)
