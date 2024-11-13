# RTSpatial: A Library for Fast Spatial Indexing

## 1. Build

### 1.1 Dependencies
(1) CMake 3.27+

(2) [NVIDIA OptiX 8.0](https://developer.nvidia.com/designworks/optix/downloads/legacy) is required to build RTSpatial

(3) [NVIDIA CUDA 12+](https://developer.nvidia.com/cuda-11-6-0-download-archive) is also required

(4) NVIDIA Driver 535+

(5) [gflags](https://github.com/gflags/gflags) is only required if you want to build the example

(6) [googletest](https://github.com/google/googletest) is only required if you want to build unit tests


Instructions for install OptiX: 
```shell
export AE_DEPS=~/ae_deps

mkdir -p $AE_DEPS
./NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64.sh --prefix=$AE_DEPS --exclude-subdir --skip-license
```

### 1.2 Building Instructions

```shell
mkdir build
cd build
# Search OptiX from $AE_DEPS and install RTSpatial to $AE_DEPS
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$AE_DEPS -DCMAKE_INSTALL_PREFIX=$AE_DEPS ..
make install # Nothing to build, only copying the headers and shaders to $AE_DEPS
```
### 1.3 OS Requirement
RTSpatial has only been tested on Linux.
