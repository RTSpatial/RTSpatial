
#ifndef RTSPATIAL_TESTS_TEST_ENVELOPE_QUERIES_H
#define RTSPATIAL_TESTS_TEST_ENVELOPE_QUERIES_H
#include <gtest/gtest.h>

#include <unordered_set>

#include "rtspatial/utils/stream.h"
#include "test_commons.h"
namespace rtspatial {
TEST(EnvelopeQueries, fp32_intersects_envelope) {
  spatial_index_f2d_t index;
  pinned_vector<envelope_f2d_t> envelopes;
  pinned_vector<envelope_f2d_t> query_envelopes;

  envelopes.push_back(envelope_f2d_t(point_f2d_t(0, 0), point_f2d_t(1, 1)));

  envelopes.push_back(
      envelope_f2d_t(point_f2d_t(0.5, -0.5), point_f2d_t(1.5, 0.5)));

  query_envelopes.push_back(
      envelope_f2d_t(point_f2d_t(0.75, 0.25), point_f2d_t(1.25, 0.75)));

  Stream stream;
  Config config;
  config.ptx_root = ptx_root;

  index.Init(config);
  index.Insert(ArrayView<envelope_f2d_t>(envelopes), stream.cuda_stream());
  counter.set(stream.cuda_stream(), 0);
  index.Query(Predicate::kIntersects,
              ArrayView<envelope_f2d_t>(query_envelopes), counter.data(),
              stream.cuda_stream());
  auto n_res = counter.get(stream.cuda_stream());
  ASSERT_EQ(n_res, 2);
}

TEST(EnvelopeQueries, fp32_intersects_envelope_big) {
  size_t n1 = 100000, n2 = 1000;
  thrust::device_vector<envelope_f2d_t> envelopes =
      GenerateUniformBoxes<float>(n1, 0.5, 0.5);

  thrust::device_vector<envelope_f2d_t> queries =
      GenerateUniformBoxes<float>(n2, 0.1, 0.1);

  spatial_index_f2d_t index;
  Stream stream;
  Config config;

  config.ptx_root = ptx_root;
  index.Init(config);
  index.Insert(ArrayView<envelope_f2d_t>(envelopes), stream.cuda_stream());
  counter.set(stream.cuda_stream(), 0);
  index.Query(Predicate::kIntersects, ArrayView<envelope_f2d_t>(queries),
              counter.data(), stream.cuda_stream());
  auto n_res = counter.get(stream.cuda_stream());
  ASSERT_EQ(n_res, 6549771);
}

TEST(EnvelopeQueries, fp32_intersects_envelope_batch) {
  size_t n1 = 100000, n2 = 1000;
  thrust::device_vector<envelope_f2d_t> envelopes =
      GenerateUniformBoxes<float>(n1, 0.5, 0.5);

  thrust::device_vector<envelope_f2d_t> queries =
      GenerateUniformBoxes<float>(n2, 0.1, 0.1);

  spatial_index_f2d_t index;
  Stream stream;
  Config config;

  config.ptx_root = ptx_root;
  index.Init(config);
  int n_batches = 10;
  int avg_size = (envelopes.size() + n_batches - 1) / n_batches;
  counter.set(stream.cuda_stream(), 0);

  for (int i_batch = 0; i_batch < n_batches; i_batch++) {
    size_t begin = i_batch * avg_size;
    size_t end = std::min(begin + avg_size, envelopes.size());
    size_t size = end - begin;

    index.Insert(ArrayView<envelope_f2d_t>(
                     thrust::raw_pointer_cast(envelopes.data()) + begin, size),
                 stream.cuda_stream());
  }
  index.Query(Predicate::kIntersects, ArrayView<envelope_f2d_t>(queries),
              counter.data(), stream.cuda_stream());
  auto n_res = counter.get(stream.cuda_stream());
  ASSERT_EQ(n_res, 6549771);
}

TEST(EnvelopeQueries, fp32_intersects_envelope_batch_update) {
  size_t n1 = 100000, n2 = 1000;
  float update_ratio = 0.05;
  thrust::device_vector<envelope_f2d_t> envelopes =
      GenerateUniformBoxes<float>(n1, 0.5, 0.5);

  thrust::device_vector<envelope_f2d_t> queries =
      GenerateUniformBoxes<float>(n2, 0.1, 0.1);

  spatial_index_f2d_t index;
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
  int n_updates = n1 * update_ratio;
  std::vector<thrust::pair<size_t, envelope_f2d_t>> updates;
  auto new_boxes = GenerateUniformBoxes<float>(n_updates, 0.5, 0.5);
  std::unordered_set<size_t> ids;
  std::uniform_int_distribution<size_t> id_dist(0, n1 - 1);
  std::mt19937 mt(n_updates);

  while (ids.size() < n_updates) {
    size_t id = id_dist(mt);
    if (ids.find(id) == ids.end()) {
      updates.push_back(thrust::make_pair(id, new_boxes[ids.size()]));
      ids.insert(id);
    }
  }

  size_t answer;
  {
    spatial_index_f2d_t tmp_index;
    pinned_vector<envelope_f2d_t> tmp_envelopes = envelopes;

    for (auto& update : updates) {
      tmp_envelopes[update.first] = update.second;
    }
    tmp_index.Init(config);
    tmp_index.Insert(ArrayView<envelope_f2d_t>(tmp_envelopes),
                     stream.cuda_stream());
    counter.set(stream.cuda_stream(), 0);
    tmp_index.Query(Predicate::kIntersects, ArrayView<envelope_f2d_t>(queries),
                    counter.data(), stream.cuda_stream());
    answer = counter.get(stream.cuda_stream());
  }
  thrust::device_vector<thrust::pair<size_t, envelope_f2d_t>> d_updates =
      updates;
  std::cout << "Before update\n";
  index.Update(ArrayView<thrust::pair<size_t, envelope_f2d_t>>(d_updates),
               stream.cuda_stream());
  stream.Sync();
  std::cout << "After update\n";
  counter.set(stream.cuda_stream(), 0);
  index.Query(Predicate::kIntersects, ArrayView<envelope_f2d_t>(queries),
              counter.data(), stream.cuda_stream());
  auto n_res = counter.get(stream.cuda_stream());
  ASSERT_EQ(n_res, answer);
}

TEST(EnvelopeQueries, fp32_test_delete) {
  spatial_index_f2d_t index;
  pinned_vector<envelope_f2d_t> envelopes;
  pinned_vector<envelope_f2d_t> query_envelopes;

  envelopes.push_back(envelope_f2d_t(point_f2d_t(0, 0), point_f2d_t(1, 1)));

  envelopes.push_back(
      envelope_f2d_t(point_f2d_t(0.5, -0.5), point_f2d_t(1.5, 0.5)));

  query_envelopes.push_back(
      envelope_f2d_t(point_f2d_t(0.75, 0.25), point_f2d_t(1.25, 0.75)));

  Stream stream;
  Config config;
  config.ptx_root = ptx_root;

  index.Init(config);
  index.Insert(ArrayView<envelope_f2d_t>(envelopes), stream.cuda_stream());
  counter.set(stream.cuda_stream(), 0);
  index.Query(Predicate::kIntersects,
              ArrayView<envelope_f2d_t>(query_envelopes), counter.data(),
              stream.cuda_stream());
  auto n_res = counter.get(stream.cuda_stream());
  ASSERT_EQ(n_res, 2);

  thrust::device_vector<size_t> deleted_ids;

  deleted_ids.push_back(1);

  index.Delete(ArrayView<size_t>(deleted_ids), stream.cuda_stream());
  counter.set(stream.cuda_stream(), 0);
  stream.Sync();
  index.Query(Predicate::kIntersects,
              ArrayView<envelope_f2d_t>(query_envelopes), counter.data(),
              stream.cuda_stream());
  n_res = counter.get(stream.cuda_stream());
  ASSERT_EQ(n_res, 1);
}

TEST(EnvelopeQueries, fp32_test_delete_compact) {
  spatial_index_f2d_t index;
  pinned_vector<envelope_f2d_t> envelopes;
  pinned_vector<envelope_f2d_t> query_envelopes;

  envelopes.push_back(envelope_f2d_t(point_f2d_t(0, 0), point_f2d_t(1, 1)));

  envelopes.push_back(
      envelope_f2d_t(point_f2d_t(0.5, -0.5), point_f2d_t(1.5, 0.5)));

  query_envelopes.push_back(
      envelope_f2d_t(point_f2d_t(0.75, 0.25), point_f2d_t(1.25, 0.75)));

  Stream stream;
  Config config;
  config.ptx_root = ptx_root;
  config.compact = true;

  index.Init(config);
  index.Insert(ArrayView<envelope_f2d_t>(envelopes), stream.cuda_stream());
  counter.set(stream.cuda_stream(), 0);
  index.Query(Predicate::kIntersects,
              ArrayView<envelope_f2d_t>(query_envelopes), counter.data(),
              stream.cuda_stream());
  auto n_res = counter.get(stream.cuda_stream());
  ASSERT_EQ(n_res, 2);

  thrust::device_vector<size_t> deleted_ids;

  deleted_ids.push_back(1);

  index.Delete(ArrayView<size_t>(deleted_ids), stream.cuda_stream());
  counter.set(stream.cuda_stream(), 0);
  stream.Sync();
  index.Query(Predicate::kIntersects,
              ArrayView<envelope_f2d_t>(query_envelopes), counter.data(),
              stream.cuda_stream());
  n_res = counter.get(stream.cuda_stream());
  ASSERT_EQ(n_res, 1);
}
}  // namespace rtspatial

#endif  // RTSPATIAL_TESTS_TEST_ENVELOPE_QUERIES_H
