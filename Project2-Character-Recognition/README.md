CUDA Character Recognition
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**
* Jiangping Xu
  * [LinkedIn](https://www.linkedin.com/in/jiangping-xu-365b19134/)
* Tested on: Windows 10, i7-4700MQ @ 2.40GHz 8GB, GT 755M 6100MB (personal laptop)
_________________________________________________________________________
[Introduction](#Stream-Compaction) - [Performance](#performance) - [Result](#result)
_________________________________________________________________________
## Introduction
In this project I implement a GPU accelerated Neural Network framework with the following features: 

* the number of hidden layers and the number of neural for each layer can be set freely
* allow batch training
* use GPU accelerated matrix multiply (using tiles and shared memory) during forward and back propagation
* Xavier Uniform Initialization is applied here for weights initialization

A tricky thing is that when doing matrix multiply in GPU, launching one thread for each element in output matrix is not enough. For example, when we do M1 * M2 = P, where M1 is a 3 by 4 matrix and M2 is a 4 by 3 matrix, we need 12 threads to load the entire matrix rather than 9.

## Performance
<p align="center">
    <img src = img/Timeused1.png>
</p>
Computation time goes up when increasing the number of layers. (For each hidden layer, there are 30 neural.)

<p align="center">
    <img src = img/Timeused2.png>
</p>
When changing the number of neural, the change in time cost is not obvious. Maybe the GPU version matrix multiply is not sensetive to the matrix size.

<p align="center">
    <img src = img/Timeused3.png>
</p>

## Result
learning rate = 0.1, batch size = 52

a 5-layer network with a structure of 10201 - 30 - 10 - 30 -52
```
Epoch: 0 Cost: 6.494277
Epoch: 1 Cost: 6.283911
Epoch: 2 Cost: 6.080731
Epoch: 3 Cost: 5.884087
Epoch: 4 Cost: 5.693419
Epoch: 5 Cost: 5.508251
Epoch: 6 Cost: 5.328183
Epoch: 7 Cost: 5.152885
Epoch: 8 Cost: 4.982097
Epoch: 9 Cost: 4.815609
Epoch: 10 Cost: 4.653265
Epoch: 11 Cost: 4.494954
Epoch: 12 Cost: 4.340601
Epoch: 13 Cost: 4.190164
Epoch: 14 Cost: 4.043626
Epoch: 15 Cost: 3.900990
Epoch: 16 Cost: 3.762273
Epoch: 17 Cost: 3.627504
Epoch: 18 Cost: 3.496716
Epoch: 19 Cost: 3.369947
Epoch: 20 Cost: 3.247230
Epoch: 21 Cost: 3.128597
Epoch: 22 Cost: 3.014070
Epoch: 23 Cost: 2.903666
Epoch: 24 Cost: 2.797389
Epoch: 25 Cost: 2.695230
Epoch: 26 Cost: 2.597171
Epoch: 27 Cost: 2.503177
Epoch: 28 Cost: 2.413202
Epoch: 29 Cost: 2.327186
Epoch: 30 Cost: 2.245058
Epoch: 31 Cost: 2.166734
Epoch: 32 Cost: 2.092118
Epoch: 33 Cost: 2.021107
Epoch: 34 Cost: 1.953589
Epoch: 35 Cost: 1.889446
Epoch: 36 Cost: 1.828552
Epoch: 37 Cost: 1.770779
Epoch: 38 Cost: 1.715999
Epoch: 39 Cost: 1.664077
Epoch: 40 Cost: 1.614883
Epoch: 41 Cost: 1.568285
Epoch: 42 Cost: 1.524154
Epoch: 43 Cost: 1.482364
Epoch: 44 Cost: 1.442791
Epoch: 45 Cost: 1.405317
Epoch: 46 Cost: 1.369825
Epoch: 47 Cost: 1.336205
Epoch: 48 Cost: 1.304351
Epoch: 49 Cost: 1.274163
Epoch: 50 Cost: 1.245543
Epoch: 51 Cost: 1.218400
Epoch: 52 Cost: 1.192649
Epoch: 53 Cost: 1.168206
Epoch: 54 Cost: 1.144996
Epoch: 55 Cost: 1.122945
Epoch: 56 Cost: 1.101985
Epoch: 57 Cost: 1.082053
Epoch: 58 Cost: 1.063086
Epoch: 59 Cost: 1.045029
Epoch: 60 Cost: 1.027830
Epoch: 61 Cost: 1.011437
Epoch: 62 Cost: 0.995804
Epoch: 63 Cost: 0.980887
Epoch: 64 Cost: 0.966647
Epoch: 65 Cost: 0.953043
Epoch: 66 Cost: 0.940041
Epoch: 67 Cost: 0.927607
Epoch: 68 Cost: 0.915708
Epoch: 69 Cost: 0.904317
Epoch: 70 Cost: 0.893405
Epoch: 71 Cost: 0.882945
Epoch: 72 Cost: 0.872915
Epoch: 73 Cost: 0.863291
Epoch: 74 Cost: 0.854052
Epoch: 75 Cost: 0.845177
Epoch: 76 Cost: 0.836649
Epoch: 77 Cost: 0.828448
Epoch: 78 Cost: 0.820560
Epoch: 79 Cost: 0.812967
Epoch: 80 Cost: 0.805656
Epoch: 81 Cost: 0.798613
Epoch: 82 Cost: 0.791824
Epoch: 83 Cost: 0.785277
Epoch: 84 Cost: 0.778962
Epoch: 85 Cost: 0.772866
Epoch: 86 Cost: 0.766981
Epoch: 87 Cost: 0.761295
Epoch: 88 Cost: 0.755800
Epoch: 89 Cost: 0.750488
Epoch: 90 Cost: 0.745350
Epoch: 91 Cost: 0.740378
Epoch: 92 Cost: 0.735566
Epoch: 93 Cost: 0.730906
Epoch: 94 Cost: 0.726391
Epoch: 95 Cost: 0.722017
Epoch: 96 Cost: 0.717776
Epoch: 97 Cost: 0.713663
Epoch: 98 Cost: 0.709673
Epoch: 99 Cost: 0.705801
```