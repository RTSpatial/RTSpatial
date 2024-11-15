#ifndef RTSPATIAL_TESTS_TEST_POINT_QUERIES_H
#define RTSPATIAL_TESTS_TEST_POINT_QUERIES_H
#include <gtest/gtest.h>

#include "rtspatial/utils/stream.h"
#include "test_commons.h"
namespace rtspatial {

TEST(PointQueries, fp32_contains_point_large) {
  SpatialIndex<float, 2> index;
  size_t n1 = 100000, n2 = 1000;

  thrust::device_vector<envelope_f2d_t> envelopes =
      GenerateUniformBoxes<float>(n1, 0.5, 0.5);
  thrust::device_vector<point_f2d_t> queries = GenerateUniformPoints<float>(n2);

  Stream stream;
  Config config;

  config.ptx_root = ptx_root;
  index.Init(config);
  index.Insert(ArrayView<envelope_f2d_t>(envelopes), stream.cuda_stream());
  counter.set(stream.cuda_stream(), 0);
  index.Query(Predicate::kContains, ArrayView<point_f2d_t>(queries),
              counter.data(), stream.cuda_stream());
  auto n_res = counter.get(stream.cuda_stream());
  ASSERT_EQ(n_res, 4224111);
}

TEST(PointQueries, fp32_contains_point_large_batch) {
  SpatialIndex<float, 2> index;
  size_t n1 = 100000, n2 = 1000;

  thrust::device_vector<envelope_f2d_t> envelopes =
      GenerateUniformBoxes<float>(n1, 0.5, 0.5);
  thrust::device_vector<point_f2d_t> queries = GenerateUniformPoints<float>(n2);

  Stream stream;
  Config config;

  config.ptx_root = ptx_root;
  index.Init(config);
  int n_batches = 10;
  int avg_size = (envelopes.size() + n_batches - 1) / n_batches;

  for (int i_batch = 0; i_batch < n_batches; i_batch++) {
    size_t begin = i_batch * avg_size;
    size_t end = std::min(begin + avg_size, envelopes.size());
    size_t size = end - begin;

    index.Insert(ArrayView<envelope_f2d_t>(
                     thrust::raw_pointer_cast(envelopes.data()) + begin, size),
                 stream.cuda_stream());
  }
  counter.set(stream.cuda_stream(), 0);
  index.Query(Predicate::kContains, ArrayView<point_f2d_t>(queries),
              counter.data(), stream.cuda_stream());
  auto n_res = counter.get(stream.cuda_stream());
  ASSERT_EQ(n_res, 4224111);
}

TEST(PointQueries, fp32_contains_point) {
  spatial_index_f2d_t index;
  pinned_vector<envelope_f2d_t> envelopes;
  pinned_vector<point_f2d_t> points;

  envelopes.push_back(envelope_f2d_t(point_f2d_t(0, 0), point_f2d_t(1, 1)));

  envelopes.push_back(
      envelope_f2d_t(point_f2d_t(0.2, 0.2), point_f2d_t(0.8, 0.8)));

  points.push_back(point_f2d_t(0.1, 0.1));  // 0
  points.push_back(point_f2d_t(0.5, 0.5));  // 0, 1
  points.push_back(point_f2d_t(0.5, 0.6));  // 0, 1

  points.push_back(point_f2d_t(1.0, 1.1));  //
  points.push_back(point_f2d_t(2.0, 2.0));  //

  Stream stream;
  Config config;

  config.ptx_root = ptx_root;
  index.Init(config);
  index.Insert(ArrayView<envelope_f2d_t>(envelopes), stream.cuda_stream());
  counter.set(stream.cuda_stream(), 0);
  index.Query(Predicate::kContains, ArrayView<point_f2d_t>(points),
              counter.data(), stream.cuda_stream());
  auto n_res = counter.get(stream.cuda_stream());
  ASSERT_EQ(n_res, 5);
}

TEST(PointQueries, fp64_contains_point) {
  spatial_index_d2d_t index;
  pinned_vector<envelope_d2d_t> envelopes;
  pinned_vector<point_d2d_t> points;

  envelopes.push_back(envelope_d2d_t(point_d2d_t(0, 0), point_d2d_t(1, 1)));

  envelopes.push_back(
      envelope_d2d_t(point_d2d_t(0.2, 0.2), point_d2d_t(0.8, 0.8)));

  points.push_back(point_d2d_t(0.1, 0.1));  // 1
  points.push_back(point_d2d_t(0.5, 0.5));  // 2
  points.push_back(point_d2d_t(0.5, 0.6));  // 2

  points.push_back(point_d2d_t(1.0, 1.1));  // 0
  points.push_back(point_d2d_t(2.0, 2.0));  // 0

  Stream stream;
  Config config;

  config.ptx_root = ptx_root;
  index.Init(config);
  index.Insert(ArrayView<envelope_d2d_t>(envelopes), stream.cuda_stream());
  counter.set(stream.cuda_stream(), 0);
  index.Query(Predicate::kContains, ArrayView<point_d2d_t>(points),
              counter.data(), stream.cuda_stream());
  auto n_res = counter.get(stream.cuda_stream());
  ASSERT_EQ(n_res, 5);
}
}  // namespace rtspatial
#endif  // RTSPATIAL_TESTS_TEST_POINT_QUERIES_H
