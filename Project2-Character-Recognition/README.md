CUDA Character Recognition
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Tabatha Hickman
  * LinkedIn:https://www.linkedin.com/in/tabatha-hickman-335987140/
* Tested on: Windows 10 Pro, i7-5600U CPU @ 2.60GHz 16GB, GeForce 840M (personal computer)

## Neural Network Implementation

This project's purpose was to create a neural network which does its computations on the GPU. I created a multi-layer perceptron with one hidden layer, so in total there are 3 layers (input, hidden, output). We evaluate the network by feeding information forward to the next layer. To process each new layer, I performed a summation for each output node on all the input nodes multiplied by the corresponding weight between those two nodes, then ran that sum through an activation function. In this case our function was ```f(x) = 1/(1+e^-x)```. 

We want to find the best set of weights so that the outputs of the network are as accurate as possible. We do this by entering a training phase. First we start with random values for the weights. Then, provided with inputs and corresponding target outputs, we run the inputs through the network and compare the outputs with their targets and find the error associated. Then through backward propagation, we can go through each weight and update it based on the results so that next time the output is more accurate.

Once the network has been trained adequately, we can run new inputs on it and see if we get some good results.

Using provided weights for a working XOR neural network, I was able to verify my code correctly feeds forward and builds the network. I've also been able to produce my own fairly accurate weights for XOR: (This had a target error of 0.01)

```
Ran 13101 iterations of training
    (0, 0) expected: 0.000000, result 0.071486
    (0, 1) expected: 1.000000, result 0.930205
    (1, 0) expected: 1.000000, result 0.923021
    (1, 1) expected: 0.000000, result 0.063928
```

Unfortunately, I was having a lot of trouble extending this to character recognition. Training does not seem to be working - the error is huge and doesn't improve at all with further iterations. I attempted to debug this and started getting "CUDA grid launch failed" errors. Upon looking this up I found out this has to do with the TDR of my Debugger, but I can't find the place to change that setting. 
 

