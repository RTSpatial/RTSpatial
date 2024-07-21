
#ifndef RTSPATIAL_DETAILS_REUSABLE_BUFFER_H
#define RTSPATIAL_DETAILS_REUSABLE_BUFFER_H
#include <thrust/device_vector.h>

namespace rtspatial {

class ReusableBuffer {
 public:
  ReusableBuffer() = default;

  explicit ReusableBuffer(size_t capacity) : tail_(0) { buf_.resize(capacity); }

  void Init(size_t capacity) { buf_.resize(capacity); }

  void Clear() { tail_ = 0; }

  size_t GetCapacity() const { return buf_.size(); }

  size_t GetOccupiedSize() const { return tail_; }

  char* GetData() { return thrust::raw_pointer_cast(buf_.data()); }

  char* GetDataTail() { return thrust::raw_pointer_cast(buf_.data()) + tail_; }

  size_t GetTail() const { return tail_; }

  char* Acquire(size_t size) {
    assert(tail_ + size <= buf_.size());
    char* prev = thrust::raw_pointer_cast(buf_.data()) + tail_;
    tail_ += size;
    return prev;
  }

  void Release(size_t size) {
    assert(tail_ >= size);
    tail_ -= size;
  }

  void SetTail(size_t size) {
    assert(size < buf_.size());
    tail_ = size;
  }

 private:
  thrust::device_vector<char> buf_;
  size_t tail_;
};

}  // namespace rtspatial
#endif  // RTSPATIAL_DETAILS_REUSABLE_BUFFER_H
