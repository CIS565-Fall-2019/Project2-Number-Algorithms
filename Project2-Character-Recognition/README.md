CUDA Character Recognition
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Taylo Nelms
  * [LinkedIn](https://www.linkedin.com/in/taylor-k-7b2110191/), [twitter](https://twitter.com/nelms_taylor), etc.
* Tested on: Windows 10, Intel i3 Coffee Lake 4-core 3.6GHz processor, 16GB RAM, NVidia GeForce GTX1650 4GB

### CMake Notes

Notably, I needed to add the following line to `CMakeLists.txt`:
`ink_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)`
Additionally, under the `target_link_libraries` function, I added links to the `cublas` and `curand` libraries

