#ifndef RTSPATIAL_TESTS_TEST_COMMONS_H
#define RTSPATIAL_TESTS_TEST_COMMONS_H
#include <random>
#include <vector>

#include "rtspatial/spatial_index.cuh"

std::string exec_root;

namespace rtspatial {
using spatial_index_f2d_t = SpatialIndex<float, 2>;
using spatial_index_d2d_t = SpatialIndex<double, 2>;

using point_f2d_t = spatial_index_f2d_t::point_t;
using envelope_f2d_t = spatial_index_f2d_t::envelope_t;

using point_d2d_t = spatial_index_d2d_t::point_t;
using envelope_d2d_t = spatial_index_d2d_t::envelope_t;

template <typename COORD_T>
std::vector<Envelope<Point<COORD_T, 2>>> GenerateUniformBoxes(
    size_t n, COORD_T max_width, COORD_T max_height) {
  std::mt19937 mt(n);
  std::uniform_real_distribution<COORD_T> min_x(0.0, 1.0);
  std::uniform_real_distribution<COORD_T> min_y(0.0, 1.0);

  std::uniform_real_distribution<COORD_T> w_dist(0.0, max_width);
  std::uniform_real_distribution<COORD_T> h_dist(0.0, max_height);

  std::vector<Envelope<Point<COORD_T, 2>>> envelopes;

  envelopes.resize(n);

  for (size_t i = 0; i < n; i++) {
    auto x = min_x(mt);
    auto y = min_y(mt);
    auto w = w_dist(mt);
    auto h = h_dist(mt);

    Point<COORD_T, 2> p1(x, y);
    Point<COORD_T, 2> p2(std::min(x + w, (COORD_T) 1.0),
                         std::min(y + h, (COORD_T) 1.0));

    envelopes[i] = Envelope<Point<COORD_T, 2>>(p1, p2);
  }
  return envelopes;
}

template <typename COORD_T>
std::vector<Point<COORD_T, 2>> GenerateUniformPoints(size_t n) {
  std::mt19937 mt(n);
  std::uniform_real_distribution<COORD_T> gen_x(0.0, 1.0);
  std::uniform_real_distribution<COORD_T> gen_y(0.0, 1.0);
  std::vector<Point<COORD_T, 2>> points;

  points.resize(n);

  for (size_t i = 0; i < n; i++) {
    auto x = gen_x(mt);
    auto y = gen_y(mt);

    points[i] = Point<COORD_T, 2>(x, y);
  }
  return points;
}
}  // namespace rtspatial
#endif  // RTSPATIAL_TEST_COMMONS_H
