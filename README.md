CUDA Number Algorithms
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

Davis Polito 
*  [https://github.com/davispolito/Project0-Getting-Started/blob/master]()
* Tested on: Windows 10, i7-8750H @ 2.20GHz 16GB, GTX 1060       
#Optimizing Block Size Per Algorithm

First we must optimize block size per algorithm
![block optimization graph](/Project2-Character-Recognition/img/blocksizeopt.png)
After creating this graph I chose blocksize to be 128 and 512 for naive and work efficient scan

#Scan comparison
![size vs. time graph](/Project2-Character-Recognition/img/sizevstime.png)


##Questions
#Explanations for each result

######cpu
######naive
######work-efficient
#####thrust

