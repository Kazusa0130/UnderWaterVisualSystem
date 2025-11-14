## Prerequisites
The program has been tested on the NVIDIA Jetson Orin NX (8GB RAM) developer kit running Ubuntu 22.04.
- OpenCV 4.12.0
- OpenCV-Contrib 4.12.0
- CUDA 12.6.68
- yaml-cpp 0.8.0


## Build dependencies
1. Install OpenCV && OpenCV-Contrib

```sh
git clone https://github.com/opencv/opencv.git -b 4.12.0
git clone https://github.com/opencv/opencv_contrib.git -b 4.12.0
cd opencv 
mkdir build && cd build
cmake -D WITH_CUDA=ON \
      -D WITH_CUDNN=ON \
      -D CUDA_ARCH_BIN="8.7" \
      -D CUDA_ARCH_PTX="" \
      -D OPENCV_GENERATE_PKGCONFIG=ON \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
      -D WITH_GSTREAMER=ON \
      -D WITH_LIBV4L=ON \
      -D BUILD_opencv_python3=ON \
      -D BUILD_TESTS=OFF \
      -D BUILD_PERF_TESTS=OFF \
      -D BUILD_EXAMPLES=OFF \
      -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      ..
make -j$(nproc)
sudo make install
```

- Future implementations will attempt to utilize the FetchContent module for integrating third-party libraries.

## Test Environment

![测试环境](.\README.assets\image.png)
