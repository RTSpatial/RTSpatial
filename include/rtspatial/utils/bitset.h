#ifndef RTSPATIAL_UTILS_BITSET_H
#define RTSPATIAL_UTILS_BITSET_H
#include <cooperative_groups.h>
#include <thrust/device_vector.h>

#include "rtspatial/utils/array_view.h"
#include "rtspatial/utils/shared_value.h"
#include "rtspatial/utils/stream.h"
#include "rtspatial/utils/util.h"

namespace rtspatial {
namespace dev {
template <typename SIZE_T>
class Bitset;

template <>
class Bitset<uint32_t> {
 public:
  Bitset() = default;

  __host__ __device__ Bitset(ArrayView<uint64_t> data, uint32_t size,
                             uint32_t* positive_count)
      : data_(data), size_(size), positive_count_(positive_count) {}

  __device__ __forceinline__ bool set_bit(uint32_t pos) {
    assert(pos < size_);
    auto bit = (uint64_t) 1l << bit_offset(pos);
    if (data_[word_offset(pos)] & bit) {
      return false;
    }
    atomicAdd(positive_count_, 1);
    data_[word_offset(pos)] |= bit;
    return true;
  }

  __device__ __forceinline__ bool set_bit_atomic(uint32_t pos) {
    assert(pos < size_);
    uint64_t old_val, new_val;
    do {
      old_val = data_[word_offset(pos)];
      if (old_val & ((uint64_t) 1l << bit_offset(pos))) {
        return false;
      }
      new_val = old_val | ((uint64_t) 1l << bit_offset(pos));
    } while (
        old_val !=
        atomicCAS(
            reinterpret_cast<unsigned long long int*>(  // NOLINT(runtime/int)
                data_.data() + word_offset(pos)),
            old_val, new_val));
    if ((old_val & (1l << bit_offset(pos))) == 0) {
      auto g = cooperative_groups::coalesced_threads();

      if (g.thread_rank() == 0) {
        atomicAdd(positive_count_, g.size());
      }
      return true;
    }
    return false;
  }

  __device__ __forceinline__ void clear() {
    auto tid = TID_1D;
    auto nthreads = TOTAL_THREADS_1D;

    for (uint64_t i = 0 + tid; i < data_.size(); i += nthreads) {
      data_[i] = 0;
    }

    if (tid == 0) {
      *positive_count_ = 0;
    }
  }

  __device__ __forceinline__ bool get_bit(uint32_t pos) const {
    assert(pos < size_);
    return (data_[word_offset(pos)] >> bit_offset(pos)) & 1l;
  }

  __device__ __forceinline__ uint32_t get_size() const { return data_.size(); }

  __device__ __forceinline__ uint32_t get_positive_count() const {
    return *positive_count_;
  }

 private:
  __device__ __forceinline__ uint64_t word_offset(uint32_t n) const {
    return n / kBitsPerWord;
  }

  __device__ __forceinline__ uint64_t bit_offset(uint32_t n) const {
    return n & (kBitsPerWord - 1);
  }
  static const uint32_t kBitsPerWord = 64;

  ArrayView<uint64_t> data_;
  uint32_t size_{};
  uint32_t* positive_count_{};
};

template <>
class Bitset<uint64_t> {
 public:
  static_assert(sizeof(uint64_t) ==
                sizeof(unsigned long long int));  // NOLINT(runtime/int)

  Bitset() = default;

  __host__ __device__ Bitset(ArrayView<uint64_t> data, uint64_t size,
                             uint64_t* positive_count)
      : data_(data), size_(size), positive_count_(positive_count) {}

  __device__ __forceinline__ bool set_bit(uint64_t pos) {
    assert(pos < size_);
    auto bit = (uint64_t) 1l << bit_offset(pos);
    if (data_[word_offset(pos)] & bit) {
      return false;
    }
    atomicAdd((unsigned long long int*) positive_count_,  // NOLINT(runtime/int)
              1);
    data_[word_offset(pos)] |= bit;
    return true;
  }

  __device__ __forceinline__ bool set_bit_atomic(uint64_t pos) {
    assert(pos < size_);
    uint64_t old_val, new_val;
    do {
      old_val = data_[word_offset(pos)];
      if (old_val & ((uint64_t) 1l << bit_offset(pos))) {
        return false;
      }
      new_val = old_val | ((uint64_t) 1l << bit_offset(pos));
    } while (
        old_val !=
        atomicCAS(
            reinterpret_cast<unsigned long long int*>(  // NOLINT(runtime/int)
                data_.data() + word_offset(pos)),
            old_val, new_val));
    if ((old_val & (1l << bit_offset(pos))) == 0) {
      auto g = cooperative_groups::coalesced_threads();

      if (g.thread_rank() == 0) {
        atomicAdd(
            (unsigned long long int*) positive_count_,  // NOLINT(runtime/int)
            g.size());
      }
      return true;
    }
    return false;
  }

  __device__ __forceinline__ void clear() {
    auto tid = TID_1D;
    auto nthreads = TOTAL_THREADS_1D;

    for (uint64_t i = 0 + tid; i < data_.size(); i += nthreads) {
      data_[i] = 0;
    }

    if (tid == 0) {
      *positive_count_ = 0;
    }
  }

  __device__ __forceinline__ bool get_bit(uint64_t pos) const {
    assert(pos < size_);
    return (data_[word_offset(pos)] >> bit_offset(pos)) & 1l;
  }

  __device__ __forceinline__ uint64_t get_size() const { return data_.size(); }

  __device__ __forceinline__ uint64_t get_positive_count() const {
    return *positive_count_;
  }

 private:
  __device__ __forceinline__ uint64_t word_offset(uint64_t n) const {
    return n / kBitsPerWord;
  }

  __device__ __forceinline__ uint64_t bit_offset(uint64_t n) const {
    return n & (kBitsPerWord - 1);
  }
  static const uint32_t kBitsPerWord = 64;

  ArrayView<uint64_t> data_;
  uint64_t size_{};
  uint64_t* positive_count_{};
};

}  // namespace dev

namespace bitset_kernels {
template <typename SIZE_T>
__global__ void SetBit(dev::Bitset<SIZE_T> bitset, size_t pos) {
  if (threadIdx.x == 0 && blockIdx.x == 0)
    bitset.set_bit(pos);
}
}  // namespace bitset_kernels

template <typename SIZE_T>
class Bitset {
 public:
  Bitset() = default;

  explicit Bitset(SIZE_T size) : size_(size) {
    data_.resize(getNumWords(size));
    positive_count_.set(0);
  }

  void Init(SIZE_T size) {
    size_ = size;
    data_.resize(getNumWords(size), 0);
    positive_count_.set(0);
  }

  dev::Bitset<SIZE_T> DeviceObject() {
    return dev::Bitset<SIZE_T>(ArrayView<uint64_t>(data_), size_,
                               positive_count_.data());
  }

  void Clear() {
    positive_count_.set(0);
    thrust::fill(data_.begin(), data_.end(), 0);
  }

  void Clear(cudaStream_t stream) {
    positive_count_.set(0, stream);
    CUDA_CHECK(cudaMemsetAsync(thrust::raw_pointer_cast(data_.data()), 0,
                               sizeof(uint64_t) * data_.size(), stream));
  }

  void SetBit(SIZE_T pos) {
    CHECK_LT(pos, size_);
    bitset_kernels::SetBit<SIZE_T><<<1, 1>>>(this->DeviceObject(), pos);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  void SetBit(SIZE_T pos, cudaStream_t stream) {
    bitset_kernels::SetBit<SIZE_T>
        <<<1, 1, 0, stream>>>(this->DeviceObject(), pos);
  }

  void Swap(Bitset<SIZE_T>& other) {
    data_.swap(other.data_);
    std::swap(size_, other.size_);
    positive_count_.Swap(other.positive_count_);
  }

  SIZE_T GetSize() const { return size_; }

  SIZE_T GetPositiveCount() const { return positive_count_.get(); }

  SIZE_T GetPositiveCount(cudaStream_t stream) const {
    return positive_count_.get(stream);
  }

  std::vector<SIZE_T> DumpPositives(cudaStream_t stream) {
    auto n_pos = GetPositiveCount(stream);
    dump_tmp_.resize(n_pos);

    auto* p_last_dump_tmp = last_dump_tmp_.data();
    auto* p_dump_tmp = thrust::raw_pointer_cast(dump_tmp_.data());
    auto obj = DeviceObject();

    last_dump_tmp_.set(stream, 0);
    thrust::for_each(thrust::cuda::par.on(stream),
                     thrust::make_counting_iterator<SIZE_T>(0),
                     thrust::make_counting_iterator<SIZE_T>(size_),
                     [=] __device__(SIZE_T i) mutable {
                       if (obj.get_bit(i)) {
                         auto last_pos = atomicAdd(p_last_dump_tmp, 1);
                         p_dump_tmp[last_pos] = i;
                       }
                     });
    std::vector<SIZE_T> dump_tmp;
    dump_tmp.assign(dump_tmp_.begin(), dump_tmp_.end());
    return dump_tmp;
  }

 private:
  static const uint64_t kBitsPerWord = 64;

  static SIZE_T getNumWords(SIZE_T size) {
    return (size + kBitsPerWord - 1) / kBitsPerWord;
  }

  thrust::device_vector<uint64_t> data_;
  pinned_vector<SIZE_T> dump_tmp_;
  SharedValue<SIZE_T> last_dump_tmp_;
  SIZE_T size_{};
  SharedValue<SIZE_T> positive_count_;
};
}  // namespace rtspatial
#endif  // RTSPATIAL_UTILS_BITSET_H
