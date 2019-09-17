CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* (TODO) YOUR NAME HERE
  * (TODO) [LinkedIn](), [personal website](), [twitter](), etc.
* Tested on: (TODO) Windows 22, i7-2222 @ 2.22GHz 22GB, GTX 222 222MB (Moore 2222 Lab)

### Design Decisions

I decided to bundle the `upsweep` threads in such a fashion that adjacent threads would be doing work as we went up the levels of the tree, allowing later threads to quit early. An area of future development would be to only spawn the necessary number of threads, but I didn't quite get there.

For the `downsweep` function, I made no such optimizations, due to time constraints and a desire to get the right answer over a more complex answer.

### Performance Anaylsis

All tests performed with a `BLOCKSIZE` of 256

### Limitations

Testing size currently limited to data up to size `2^16`. The scan's upsweep and downsweep start to break above that. I'm not entirely sure why, but I suspect it relates to either GPU maximums or some silly thing I did where I put an int somewhere that a long should be. I am, unfortunately, out of time to determine why this is the case.

