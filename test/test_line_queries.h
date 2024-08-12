#ifndef RTSPATIAL_TESTS_TEST_POINT_QUERIES_H
#define RTSPATIAL_TESTS_TEST_POINT_QUERIES_H
#include <gtest/gtest.h>

#include "rtspatial/utils/stream.h"
#include "test_commons.h"
namespace rtspatial {

TEST(LineQueries, fp32_intersects) {
  SpatialIndex<float, 2> index;
  size_t n1 = 100000, n2 = 1000;

  thrust::device_vector<envelope_f2d_t> envelopes =
      GenerateUniformBoxes<float>(n1, 0.5, 0.5);
  thrust::device_vector<Line<point_f2d_t>> queries = GenerateUniformLines<float>(n2);

  Stream stream;
  Config config;

  config.ptx_root = ptx_root;
  index.Init(config);
  index.Insert(ArrayView<envelope_f2d_t>(envelopes), stream.cuda_stream());
  counter.set(stream.cuda_stream(), 0);
  index.Query(Predicate::kIntersects, ArrayView<point_f2d_t>(queries),
              counter.data(), stream.cuda_stream());
  auto n_res = counter.get(stream.cuda_stream());
//  ASSERT_EQ(n_res, 4224111);
}

}  // namespace rtspatial
#endif  // RTSPATIAL_TESTS_TEST_POINT_QUERIES_H
