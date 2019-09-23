CUDA Character Recognition
======================
**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

Dhruv Karthik: [LinkedIn](https://www.linkedin.com/in/dhruv_karthik/)

Tested on: Windows 10 Home, Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz, 16GM, GTX 2070 - Compute Capability 7.5
____________________________________________________________________________________
![Developer](https://img.shields.io/badge/Developer-Dhruv-0f97ff.svg?style=flat) ![CUDA 10.1](https://img.shields.io/badge/CUDA-10.1-yellow.svg) ![Built](https://img.shields.io/appveyor/ci/gruntjs/grunt.svg) ![Issues](https://img.shields.io/badge/issues-none-green.svg)
____________________________________________________________________________________
## Outcome
### XOR Convergence
![](img/chareg.PNG)


## Additional Implementation Features
### Variable MLP Builder & Batched Updates
Define any MLP very easily as follows:
```C++
    //Network Structure
int numSamples = 1;
int inputDim = 2;
int numLayers = 1;
int hiddenDim[1] = {5};
int outputDim = 2;
```
Notice ```numSamples```. This allows you to set the batchSize of the Neural Network to perform Batched Gradient Descent, as opposed to stochastic gradient descent which is the base implementation. 

## Tragic Historical Significance of the XOR Problem
Neural Networks are not new.In 1958, [Frank Rosenblatt](https://en.wikipedia.org/wiki/Frank_Rosenblatt) proposed a hypothetical model of a brain's nervous system and coined it the *perceptron*. Essentially, this model fit a line to a dataset. However, as seen below, you can't fit a line to an XOR function. 

![](img/goodperceptron.png)

The perceptron got a ton of hype in the 60's, but two authors published a [book](https://mitpress.mit.edu/books/perceptrons) on emphasizing why perceptron's are terrible, because they can't fit the XOR function. This single handedly resulted in the first of three AI Winters. 
