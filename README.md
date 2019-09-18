CUDA Number Algorithms
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

Davis Polito 
*  [https://github.com/davispolito/Project0-Getting-Started/blob/master]()
* Tested on: Windows 10, i7-8750H @ 2.20GHz 16GB, GTX 1060       
#Optimizing Block Size Per Algorithm

First we must optimize block size per algorithm
![block optimization graph](/Project2-Stream-Compaction/img/blocksizeopt.PNG)
After creating this graph I chose blocksize to be 128 and 512 for naive and work efficient scan

#Scan comparison
![size vs. time graph](/Project2-Stream-Compaction/img/sizevstime.PNG)


##Questions
#Explanations for each result

######cpu This method has O(n) runtime and is only affected by size of the aray
######naivei this method has O(nlogn) runtime
######work-efficient This has a possible runtime of O(n) but do to memory access and non ideal uses of threads and warps (i.e. warp branching) We see a slower runtime than cpu
#####thrust 
![Console Output From Steam compaction](/Project2-Stream-Compaction/img/consoleOutput.PNG)



