CUDA Number Algorithms
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Yan Dong
  - [LinkedIn](https://www.linkedin.com/in/yan-dong-572b1113b/)
  - [personal website](coffeier.com)
  - [github](https://github.com/coffeiersama)
* Tested on: Windows 10, i7-8750 @ 2.22GHz  (12CPUs)  16GB, GTX 1060 14202MB (OMEN 15-dc0xxx)



### Homework2

Link to the readmes of the other two subprojects.

[stream compaction](https://github.com/coffeiersama/Project2-Number-Algorithms/tree/master/Project2-Stream-Compaction)

[character recognization](https://github.com/coffeiersama/Project2-Number-Algorithms/tree/master/Project2-Character-Recognition)



### Some useful knowledge review

##### shared_memory

on-board memory(global memory, high latency)

on-chip memory(shared memory, low latency)

shared memory is faster than global memory about 20-30 times

one block has a part of shared memory.

###### Definition:

extern "___shared___" int tile[];//do not know the shared memory size

[Reference](https://www.cnblogs.com/1024incn/p/4605502.html)



##### resolve the external symbol problem:

the program can find the h file, but can not find the dll file.

so use link!