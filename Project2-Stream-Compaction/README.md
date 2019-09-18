CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Alexis Ward
  * [LinkedIn](https://www.linkedin.com/in/alexis-ward47/), [personal website](https://www.alexis-ward.tech/)
* Tested on: Windows 10, i7-8750H CPU @ 2.20GHz 16GB, GTX 1050 Ti (Same borrowed computer as last time, for the second half of the assignment.)

![](img/scan_tests_output.png)

# README

In this project, I implemented GPU stream compaction in CUDA, from scratch. This is good practice in advance of the path tracer assignment.

I began by coding a CPU implementation of scan and compact, and then coded different GPU editions (starting with a "Naive" approach, then "Work-Efficient", then by just using Thrust's implementation).

This readme will compare the runtime results of all versions.


## Performance Analysis

### Scan Implementations

![](img/scan.png)

I ran my program on power-of-two array sizes (from 2^8 to 2^20), and recorded and plotted the results.

### Stream Compaction Implementations

![](img/scan.png)

I ran my program on power-of-two array sizes (from 2^8 to 2^18), and recorded and plotted the results.