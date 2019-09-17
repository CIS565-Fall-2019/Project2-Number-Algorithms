CUDA Character Recognition
======================

### Overview

The overview of the subproject is to create and train a Neural Network for Character Recognition. In more detail, given the image of the alphabet, the Neural Network can identify what character it is and perform the classification. For first testing our Neural network, we will create the network and train on the 2X2 XOR data and then extend it to the character recognition.

### Neural Network

Neural Network in the Multi-Layer Perceptron where the classifier is designing by combining many perceptrons at the same time. As peach perceptron is a linear classfier, it can't classify some of the complex datasets and model the high complexity, The Neural network have the non-linearity aspect as it is made up of several such single multi perceptrons. 

![](img/Weighting.PNG)

As shown in the figure above, we will be creating a neural network which has a single hidden layer. But we will vary the number of hidden units in the hidden layer. Note that the above figure is for the single datapoint. We see the for each hidden unit is calculated as the weighted sum of each of features present in the data. Later on, to bring non-linearity, activation function such as Signmoid, ReLu are used. After hidden units are computed, we can calculate the value of the units as the same as that of the hidden layer and then add on the classifier which increases the confidence of the classification by increasing and decreasing the probabilites to the extremes. 

### My Neural Network Implementation

In my implemetation of Neural network, I'm training all the data points at the same time, hence, we have the input data as the array of 'DATA_POINTS * NUMBER_OF_FEATURES'
