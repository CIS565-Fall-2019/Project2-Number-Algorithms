CUDA Character Recognition
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Taylo Nelms
  * [LinkedIn](https://www.linkedin.com/in/taylor-k-7b2110191/), [twitter](https://twitter.com/nelms_taylor), etc.
* Tested on: Windows 10, Intel i3 Coffee Lake 4-core 3.6GHz processor, 16GB RAM, NVidia GeForce GTX1650 4GB

### CMake Notes

Notably, I needed to add the following line to `CMakeLists.txt`:
`link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)`
Additionally, under the `target_link_libraries` function, I added links to the `cublas` and `curand` libraries

### ML Design Notes
As a relative novice to machine learning, I have elected to primarily put in only two fully-connected hidden layers.

However, I HAVE elected to process the images on the front end through some common image convolutions, followed by a run through a max pooling layer. These results become the first "feature layer." This has the advantage of reducing the number of "pixels" from 10201 to 6534, as I'm running the images through convolution with 3x3 kernels, followed by a 3x3 max pooling operation. Both the convolution and max-pooling are hand-implemented.

There is the slight hiccup that I am not sure how to property back-propogate through a convolutional layer and update convolution kernel weights. That said, I stand by the decision, as the resultant data sets have more activated information than the original data.

