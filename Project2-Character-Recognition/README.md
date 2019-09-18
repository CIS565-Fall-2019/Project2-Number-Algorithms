CUDA Character Recognition
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* John Marcao
  * [LinkedIn](https://www.linkedin.com/in/jmarcao/)
  * [Personal Website](https://jmarcao.github.io)
* Tested on: Windows 10, i5-4690K @ 3.50GHz, 8GB DDR3, RTX 2080 TI 3071MB (Personal)

# Goals

The goal of this project is to explore one of the many applications of parallel programming: Machine Learning. We start with implementing a Perceptron, a basic ML construct that uses several layers, each with varying weights and biases, to implement a character recognition machine. The Perceptron takes in data in its input layer and then utilizes parallel algotihms to perform matrix mutliplication and operations to transform the input data into a guess in the output data. With enough "training", the Perceptron can detects what character is written in an image.

The perceptron I designed has three layers:
* Input Layer - Accepts formatted image data
* Hidden Layer - Intermediate layer that reduces the number of datapoints in the image by 80%
* Output Layer - Final layer that produces an output based on the weights learned by the Perceptron.

Unfortunetly I was not able to get my Perceptron up and running in the alotted time. Since I cannot discuss performance characterization, I will go over some issues in my design and some lessons learned.

## What is Working

* Forward-Propohation... sort of. My machine is able to take an input and feed it through the perceptron to form an output decision. There were some changes I had to make in my system that deviated from the traditional models I studied. My node values were particularly high, so much so that the Softmax equation applied to the last layer would fail due to overflowing float values (e^1023043 is too much?). I remidied this by adding a step where each value in the output layer is scaled down such that the Softmax equation still works.
* Matrix Manipulation - Using the cublas library, I was able to set up several functions and calls to perform a variety of transformations on the low-level matrix values through my more high-level classes.

## What is not Working

* Learning/Backpropagation - Right now, the system can go through one learning epoch and then it can apply the deltas based on the error to the weights of the system. However, during the second epoch, the system diverges and my float values overflow. I am not sure why this is the case. Thoughts include inverted operations, invertedt matrix indicies, etc. 

# Challenges

The most challenging part of this project was getting the complexity under control. The perceptron has a lot of moving parts and a lot of equations, and getting them confused and mixed up is easy. There is also additional complexity with the introduction of the cublas library with CUDA. The library is incredibly popwerful, providing several functions for Vector and Matrix operations. Part of the challenge of this project was understanding the library as well as its API and properly using it. I found that there are a lot of math concepts that, although vital, were lost on me.