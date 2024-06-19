
#ifndef RTSPATIAL_TESTS_TEST_ENVELOPE_QUERIES_H
#define RTSPATIAL_TESTS_TEST_ENVELOPE_QUERIES_H

#include <gtest/gtest.h>

#include "rtspatial/utils/stream.h"
#include "test_commons.h"
namespace rtspatial {

TEST(EnvelopeQueries, fp32_intersects_envelope) {
  spatial_index_f2d_t index;
  pinned_vector<envelope_f2d_t> envelopes;
  pinned_vector<envelope_f2d_t> query_envelopes;
  Queue<thrust::pair<size_t, size_t>> result;

  envelopes.push_back(envelope_f2d_t(point_f2d_t(0, 0), point_f2d_t(1, 1)));

  envelopes.push_back(
      envelope_f2d_t(point_f2d_t(0.5, -0.5), point_f2d_t(1.5, 0.5)));

  query_envelopes.push_back(
      envelope_f2d_t(point_f2d_t(0.75, 0.25), point_f2d_t(1.25, 0.75)));

  result.Init(1000);
  Stream stream;

  index.Init(exec_root);
  index.Load(ArrayView<envelope_f2d_t>(envelopes), stream.cuda_stream());
  index.IntersectsWhatQuery(ArrayView<envelope_f2d_t>(query_envelopes), result,
                            stream.cuda_stream());
  uint32_t n_res = result.size(stream.cuda_stream());
  std::cout << "Intersects " << n_res << std::endl;
  ASSERT_EQ(n_res, 2);
}

TEST(EnvelopeQueries, fp32_intersects_envelope_real) {
  size_t n1 = 100000, n2 = 1000;
  thrust::device_vector<envelope_f2d_t> envelopes =
      GenerateUniformBoxes<float>(n1, 0.5, 0.5);

  thrust::device_vector<envelope_f2d_t> queries =
      GenerateUniformBoxes<float>(n2, 0.1, 0.1);

  spatial_index_f2d_t index;
  Queue<thrust::pair<size_t, size_t>> result;
  Stream stream;

  result.Init(n1 * n2 * 0.1);
  index.Init(exec_root);
  index.Load(ArrayView<envelope_f2d_t>(envelopes), stream.cuda_stream());
  index.IntersectsWhatQuery(ArrayView<envelope_f2d_t>(queries), result,
                            stream.cuda_stream());
  uint32_t n_res = result.size(stream.cuda_stream());
  ASSERT_EQ(n_res, 6549771);
}

}  // namespace rtspatial

#endif  // RTSPATIAL_TESTS_TEST_ENVELOPE_QUERIES_H
