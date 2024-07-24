#ifndef RTSPATIAL_UTILS_SHARED_VALUE_H
#define RTSPATIAL_UTILS_SHARED_VALUE_H

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "rtspatial/utils/exception.h"
#include "rtspatial/utils/util.h"

namespace rtspatial {
template <typename T>
class SharedValue {
 public:
  SharedValue() {
    d_buffer_.resize(1);
    h_buffer_.resize(1);
  }

  void set(const T& t) { d_buffer_[0] = t; }

  void set(cudaStream_t stream, const T& t) {
    h_buffer_[0] = t;
    CUDA_CHECK(cudaMemcpyAsync(thrust::raw_pointer_cast(d_buffer_.data()),
                               thrust::raw_pointer_cast(h_buffer_.data()),
                               sizeof(T), cudaMemcpyHostToDevice, stream));
  }

  typename thrust::device_vector<T>::reference get() { return d_buffer_[0]; }

  typename thrust::device_vector<T>::const_reference get() const {
    return d_buffer_[0];
  }

  T get(cudaStream_t stream) const {
    CUDA_CHECK(cudaMemcpyAsync(
        const_cast<T*>(thrust::raw_pointer_cast(h_buffer_.data())),
        thrust::raw_pointer_cast(d_buffer_.data()), sizeof(T),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return h_buffer_[0];
  }

  T* data() { return thrust::raw_pointer_cast(d_buffer_.data()); }

  const T* data() const { return thrust::raw_pointer_cast(d_buffer_.data()); }

  void Assign(const SharedValue<T>& rhs) {
    CUDA_CHECK(cudaMemcpy(thrust::raw_pointer_cast(d_buffer_.data()),
                          thrust::raw_pointer_cast(rhs.d_buffer_.data()),
                          sizeof(T), cudaMemcpyDefault));
  }

  void Swap(SharedValue<T>& rhs) { d_buffer_.swap(rhs.d_buffer_); }

 private:
  thrust::device_vector<T> d_buffer_;
  pinned_vector<T> h_buffer_;
};
}  // namespace rtspatial

#endif  // RTSPATIAL_UTILS_SHARED_VALUE_H
