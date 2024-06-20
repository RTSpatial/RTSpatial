#ifndef RTSPATIAL_UTILS_ARRAY_VIEW_H
#define RTSPATIAL_UTILS_ARRAY_VIEW_H
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/swap.h>

#include "rtspatial/utils/util.h"

namespace rtspatial {
template <typename T>
class ArrayView {
 public:
  ArrayView() = default;

  template <typename ALLOC_T>
  ArrayView(const thrust::device_vector<T, ALLOC_T>& vec)
      : data_(const_cast<T*>(thrust::raw_pointer_cast(vec.data()))),
        size_(vec.size()) {}

  ArrayView(const thrust::device_vector<T>& vec)
      : data_(const_cast<T*>(thrust::raw_pointer_cast(vec.data()))),
        size_(vec.size()) {}

  ArrayView(const pinned_vector<T>& vec)
      : data_(const_cast<T*>(thrust::raw_pointer_cast(vec.data()))),
        size_(vec.size()) {}

  DEV_HOST ArrayView(T* data, size_t size) : data_(data), size_(size) {}

  DEV_HOST_INLINE T* data() { return data_; }

  DEV_HOST_INLINE const T* data() const { return data_; }

  DEV_HOST_INLINE size_t size() const { return size_; }

  DEV_HOST_INLINE bool empty() const { return size_ == 0; }

  DEV_INLINE T& operator[](size_t i) {
    assert(i < size_);
    return data_[i];
  }

  DEV_INLINE const T& operator[](size_t i) const {
#ifndef NDEBUG
    if (i >= size_) {
      printf("thread: %u i: %llu size: %llu\n", TID_1D, i, size_);
    }
#endif
    assert(i < size_);
    return data_[i];
  }

  DEV_HOST_INLINE void Swap(ArrayView<T>& rhs) {
    thrust::swap(data_, rhs.data_);
    thrust::swap(size_, rhs.size_);
  }

  DEV_HOST_INLINE T* begin() { return data_; }

  DEV_HOST_INLINE T* end() { return data_ + size_; }

  DEV_HOST_INLINE const T* begin() const { return data_; }

  DEV_HOST_INLINE const T* end() const { return data_ + size_; }

 private:
  T* data_{};
  size_t size_{};
};
}  // namespace rtspatial

#endif  // RTSPATIAL_UTILS_ARRAY_VIEW_H
