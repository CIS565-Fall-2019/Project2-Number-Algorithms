# Project 2a: CUDA Stream Compaction
**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 2 - Stream Compaction**

Caroline Lachanski: [LinkedIn](https://www.linkedin.com/in/caroline-lachanski/), [personal website](http://carolinelachanski.com/)

Tested on: Windows 10, i5-6500 @ 3.20GHz 16GB, GTX 1660 (personal computer)

## Project Description

This goal of this project was to gain to familiarity with writing parallel algorithms, specifically Scan (prefix sum) and Stream Compaction (used to remove values of zero from an array), from scratch in CUDA. The implementations in this project were mainly based on [these slides](https://onedrive.live.com/view.aspx?resid=A6B78147D66DD722!93669&ithint=file%2cpptx&authkey=!AOZdbr6KT8OQ9fs). This project contains several implementations: a CPU version, a naive GPU version, a work-efficient GPU version, and one using the Thrust library functions.

### CPU

The CPU implementations were done simply to allow me to become more familiar with the algorithms. These are straightforward and simple to implement, and include an implementations of Scan, Scatter, Stream Compaction without using Scan, and Stream Compaction using Scan and Scatter.

INSERT PICTURES HERE

### Naive

The first GPU implementation of Scan is a naive one, and results in an inclusive Scan that must be modified (using a shift right kernel) to an exclusive Scan. 

INSERT PICTURE

### Work-Efficient

Next we have work-efficient GPU versions of Scan, Scatter, and Stream Compaction. The work-efficient version of Scan involves two phases, an upsweep and a downsweep:

INSERT PICTURE

Because of REASON, the input array must be buffered with zeros until its size is a power of 2.

We can then use Scan in a GPU implementation of Stream Compaction, which also uses Scatter and MapToBoolean kernels.

INSERT PICTURE

### Thrust

Lastly, we perform Stream Compaction with the Thrust library's thrust::exclusive_scan.

## Analysis

