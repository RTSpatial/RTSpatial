#ifndef RTSPATIAL_TESTS_TEST_POINT_QUERIES_H
#define RTSPATIAL_TESTS_TEST_POINT_QUERIES_H
#include <gtest/gtest.h>

#include "rtspatial/utils/stream.h"
#include "test_commons.h"
namespace rtspatial {

TEST(PointQueries, fp32_contains_point_triangle_large) {
  SpatialIndex<float, 2, true> index;
  Queue<thrust::pair<size_t, size_t>> result;
  size_t n1 = 100000, n2 = 1000;

  thrust::device_vector<envelope_f2d_t> envelopes =
      GenerateUniformBoxes<float>(n1, 0.5, 0.5);
  thrust::device_vector<point_f2d_t> queries = GenerateUniformPoints<float>(n2);

  result.Init(n1 * n2 * 0.1);
  Stream stream;

  index.Init(exec_root);
  index.Load(ArrayView<envelope_f2d_t>(envelopes), stream.cuda_stream());
  index.ContainsWhatQuery(ArrayView<point_f2d_t>(queries), result,
                          stream.cuda_stream());
  uint32_t n_res = result.size(stream.cuda_stream());
  ASSERT_EQ(n_res, 4224110);
}

TEST(PointQueries, fp32_contains_point_large) {
  SpatialIndex<float, 2, false> index;
  Queue<thrust::pair<size_t, size_t>> result;
  size_t n1 = 100000, n2 = 1000;

  thrust::device_vector<envelope_f2d_t> envelopes =
      GenerateUniformBoxes<float>(n1, 0.5, 0.5);
  thrust::device_vector<point_f2d_t> queries = GenerateUniformPoints<float>(n2);

  result.Init(n1 * n2 * 0.1);
  Stream stream;

  index.Init(exec_root);
  index.Load(ArrayView<envelope_f2d_t>(envelopes), stream.cuda_stream());
  index.ContainsWhatQuery(ArrayView<point_f2d_t>(queries), result,
                          stream.cuda_stream());
  uint32_t n_res = result.size(stream.cuda_stream());
  ASSERT_EQ(n_res, 4224111);
}

TEST(PointQueries, fp32_contains_point_triangle) {
  SpatialIndex<float, 2, true> index;
  pinned_vector<envelope_f2d_t> envelopes;
  pinned_vector<point_f2d_t> points;
  Queue<thrust::pair<size_t, size_t>> result;

  envelopes.push_back(envelope_f2d_t(point_f2d_t(0, 0), point_f2d_t(1, 1)));

  envelopes.push_back(
      envelope_f2d_t(point_f2d_t(0.2, 0.2), point_f2d_t(0.8, 0.8)));

  points.push_back(point_f2d_t(0.1, 0.1));  // 0
  points.push_back(point_f2d_t(0.5, 0.5));  // 0, 1
  points.push_back(point_f2d_t(0.5, 0.6));  // 0, 1

  points.push_back(point_f2d_t(1.0, 1.1));  //
  points.push_back(point_f2d_t(2.0, 2.0));  //

  result.Init(1000);
  Stream stream;

  index.Init(exec_root);
  index.Load(ArrayView<envelope_f2d_t>(envelopes), stream.cuda_stream());
  index.ContainsWhatQuery(ArrayView<point_f2d_t>(points), result,
                          stream.cuda_stream());
  uint32_t n_res = result.size(stream.cuda_stream());
  ASSERT_EQ(n_res, 5);
}

TEST(PointQueries, fp32_contains_point) {
  spatial_index_f2d_t index;
  pinned_vector<envelope_f2d_t> envelopes;
  pinned_vector<point_f2d_t> points;
  Queue<thrust::pair<size_t, size_t>> result;

  envelopes.push_back(envelope_f2d_t(point_f2d_t(0, 0), point_f2d_t(1, 1)));

  envelopes.push_back(
      envelope_f2d_t(point_f2d_t(0.2, 0.2), point_f2d_t(0.8, 0.8)));

  points.push_back(point_f2d_t(0.1, 0.1));  // 0
  points.push_back(point_f2d_t(0.5, 0.5));  // 0, 1
  points.push_back(point_f2d_t(0.5, 0.6));  // 0, 1

  points.push_back(point_f2d_t(1.0, 1.1));  //
  points.push_back(point_f2d_t(2.0, 2.0));  //

  result.Init(1000);
  Stream stream;

  index.Init(exec_root);
  index.Load(ArrayView<envelope_f2d_t>(envelopes), stream.cuda_stream());
  index.ContainsWhatQuery(ArrayView<point_f2d_t>(points), result,
                          stream.cuda_stream());
  uint32_t n_res = result.size(stream.cuda_stream());
  ASSERT_EQ(n_res, 5);
}

TEST(PointQueries, fp64_contains_point) {
  spatial_index_d2d_t index;
  pinned_vector<envelope_d2d_t> envelopes;
  pinned_vector<point_d2d_t> points;
  Queue<thrust::pair<size_t, size_t>> result;

  envelopes.push_back(envelope_d2d_t(point_d2d_t(0, 0), point_d2d_t(1, 1)));

  envelopes.push_back(
      envelope_d2d_t(point_d2d_t(0.2, 0.2), point_d2d_t(0.8, 0.8)));

  points.push_back(point_d2d_t(0.1, 0.1));  // 1
  points.push_back(point_d2d_t(0.5, 0.5));  // 2
  points.push_back(point_d2d_t(0.5, 0.6));  // 2

  points.push_back(point_d2d_t(1.0, 1.1));  // 0
  points.push_back(point_d2d_t(2.0, 2.0));  // 0

  result.Init(1000);
  Stream stream;

  index.Init(exec_root);
  index.Load(ArrayView<envelope_d2d_t>(envelopes), stream.cuda_stream());
  index.ContainsWhatQuery(ArrayView<point_d2d_t>(points), result,
                          stream.cuda_stream());
  uint32_t n_res = result.size(stream.cuda_stream());
  ASSERT_EQ(n_res, 5);
}

}  // namespace rtspatial
#endif  // RTSPATIAL_TESTS_TEST_POINT_QUERIES_H
