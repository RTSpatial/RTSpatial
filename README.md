# LibRTS: A Spatial Indexing Library by Ray Tracing
The Ray-Tracing (RT) core has become a widely integrated feature in modern GPUs to accelerate ray-tracing rendering. 
Recent research has shown that RT cores can also be repurposed to accelerate non-rendering workloads. 
Since the RT core essentially serves as a hardware accelerator for Bounding Volume Hierarchy (BVH) tree traversal, 
it holds the potential to significantly improve the performance of spatial workloads. 
However, the specialized RT programming model poses challenges for using RT cores in these scenarios. 
Inspired by the core functionality of RT cores, we designed and implemented LibRTS, 
a spatial index library that leverages RT cores to accelerate spatial queries. 
LibRTS supports both point and range queries and remains mutable to accommodate changing data. 
Instead of relying on a case-by-case approach, 
LibRTS provides a general, high-performance spatial indexing framework for spatial data processing. 
By formulating spatial queries as RT-suitable problems and overcoming load-balancing challenges, 
LibRTS delivers superior query performance through RT cores 
without requiring developers to master complex programming on this specialized hardware. 
Compared to CPU and GPU spatial libraries, 
LibRTS achieves speedups of up to 85.1x for point queries, 94.0x for range-contains queries, and 
11.0x for range-intersects queries. In a real-world application, point-in-polygon testing, 
LibRTS also surpasses the state-of-the-art RT method by up to 3.8x.

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

## 2. TODO List

- [ ] Add support for 3D geometries
- [ ] Implement line-box intersection queries
- [ ] Test and validate double-precision computations
- [ ] Optimize query performance under high update loads  

