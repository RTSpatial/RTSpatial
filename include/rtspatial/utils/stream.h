#ifndef RTSPATIAL_UTILS_STREAM_H
#define RTSPATIAL_UTILS_STREAM_H
#include <cuda_runtime.h>

#include "rtspatial/utils/exception.h"
#include "rtspatial/utils/util.h"

namespace rtspatial {
enum class StreamPriority { kDefault, kHigh, kLow };

class Stream {
 public:
  explicit Stream(StreamPriority priority = StreamPriority::kDefault)
      : priority_(priority) {
    if (priority_ == StreamPriority::kDefault) {
      CUDA_CHECK(
          cudaStreamCreateWithFlags(&cuda_stream_, cudaStreamNonBlocking));
    } else {
      int leastPriority, greatestPriority;
      CUDA_CHECK(
          cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
      CUDA_CHECK(cudaStreamCreateWithPriority(
          &cuda_stream_, cudaStreamNonBlocking,
          priority_ == StreamPriority::kHigh ? greatestPriority
                                             : leastPriority));
    }
  }

  Stream(const Stream& other) = delete;

  Stream(Stream&& other) noexcept
      : priority_(other.priority_), cuda_stream_(other.cuda_stream_) {
    other.cuda_stream_ = nullptr;
  }

  Stream& operator=(const Stream& other) = delete;

  Stream& operator=(Stream&& other) noexcept {
    if (this != &other) {
      this->cuda_stream_ = other.cuda_stream_;
      other.cuda_stream_ = nullptr;
    }
    return *this;
  }

  ~Stream() {
    if (cuda_stream_ != nullptr) {
      CUDA_CHECK(cudaStreamDestroy(cuda_stream_));
    }
  }

  void Sync() const { CUDA_CHECK(cudaStreamSynchronize(cuda_stream_)); }

  cudaStream_t cuda_stream() const { return cuda_stream_; }

 private:
  StreamPriority priority_;
  cudaStream_t cuda_stream_{};
};
}  // namespace rtspatial

#endif  // RTSPATIAL_UTILS_STREAM_H
