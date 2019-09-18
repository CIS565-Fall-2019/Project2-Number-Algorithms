CUDA Character Recognition
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Klayton Wittler
	* [LinkedIn](https://www.linkedin.com/in/klayton-wittler/)
* Tested on: Windows 10 Pro, i7-7700K @ 4.20GHz 16.0GB, GTX 1070 8.192GB (my PC)

## Sections

* [Introduction](#introduction)
* [Impelmentation](#implementation)
* [Additions](#additions)

# Introduction
This project is an attempt at developing a multi-layer perceptron from scratch that utilizing some of the parallelization offered by the GPU.

# Implementation
My implemenation contains a flag ```XOR``` which switches from the XOR test to the image dataset. I currently have the ability to load the images, store them into an arary and give it to training or testing functions. 

The current issue with forward propagation for prediction is matrices are not multiplying correctly. This can be seen in the image below for the XOR example where the first value is correct but the second is not.


The implementation of matrix multiplcation is correct and can be verified with the ```testMatrixMultiply()``` function, an example is below.

Once forward propagation is complete the rest of the structure to train a network is in place. I can permute the data over epochs as to never see the data in the same order, which is helpful in performing gradient descent on the loss. The loss and back propagation structure are in place but have not been tested since the forward proagation results are not correct.

# Additions
## Matrix multiplication
Each layer of the network is handle with a matrix multiplication through cublas then feed through the sigmoid activation in parallel.

