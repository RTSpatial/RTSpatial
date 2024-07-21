#ifndef RTSPATIAL_BENCHMARK_H
#define RTSPATIAL_BENCHMARK_H
#include <thrust/sort.h>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/linestring.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/geometries/register/box.hpp>
#include <boost/geometry/geometries/register/point.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <boost/iterator/function_output_iterator.hpp>
#include <fstream>
#include <mutex>
#include <thread>

#include "flags.h"
#include "rtspatial/geom/envelope.cuh"
#include "rtspatial/geom/point.cuh"
#include "rtspatial/spatial_index.cuh"
#include "rtspatial/utils/stopwatch.h"
#include "rtspatial/utils/stream.h"

struct rtree_point_t {
  uint32_t id;
  float x, y;
  rtree_point_t() : id(std::numeric_limits<uint32_t>::max()), x(0), y(0) {}
  rtree_point_t(float in_x, float in_y) : x(in_x), y(in_y) {}
};

struct rtree_box_t {
  size_t id;
  rtree_point_t min, max;
};

struct rtree_point_equal_to {
  inline bool operator()(rtree_point_t const& l, rtree_point_t const& r) const {
    return l.id == r.id;
  }
};

BOOST_GEOMETRY_REGISTER_POINT_2D(rtree_point_t, double,
                                 boost::geometry::cs::cartesian, x, y);
BOOST_GEOMETRY_REGISTER_BOX(rtree_box_t, rtree_point_t, min, max);

namespace rtspatial {
template <typename COORD_T>
std::vector<Envelope<Point<COORD_T, 2>>> LoadBoxes(
    const std::string& path, int limit = std::numeric_limits<int>::max()) {
  std::ifstream ifs(path);
  std::string line;
  using point_type = boost::geometry::model::d2::point_xy<COORD_T>;
  using polygon_type = boost::geometry::model::polygon<point_type>;
  using region_t = Envelope<Point<COORD_T, 2>>;
  std::vector<region_t> regions;

  while (std::getline(ifs, line)) {
    if (!line.empty()) {
      std::vector<point_type> points;

      if (line.rfind("MULTIPOLYGON", 0) == 0) {
        boost::geometry::model::multi_polygon<polygon_type> c;
        boost::geometry::read_wkt(line, c);

        for (auto& poly : c) {
          for (auto& p : poly.outer()) {
            points.push_back(p);
          }
        }
      } else if (line.rfind("POLYGON", 0) == 0) {
        boost::geometry::model::polygon<point_type> c;
        boost::geometry::read_wkt(line, c);

        for (auto& p : c.outer()) {
          points.push_back(p);
        }
      } else {
        std::cerr << "Bad Geometry " << line << "\n";
        abort();
      }

      COORD_T lows[2] = {std::numeric_limits<COORD_T>::max(),
                         std::numeric_limits<COORD_T>::max()};
      COORD_T highs[2] = {std::numeric_limits<COORD_T>::lowest(),
                          std::numeric_limits<COORD_T>::lowest()};

      for (auto& p : points) {
        lows[0] = std::min(lows[0], p.x());
        highs[0] = std::max(highs[0], p.x());
        lows[1] = std::min(lows[1], p.y());
        highs[1] = std::max(highs[1], p.y());
      }

      region_t region(Point<COORD_T, 2>(lows[0], lows[1]),
                      Point<COORD_T, 2>(highs[0], highs[1]));
      regions.push_back(region);
      if (regions.size() >= limit) {
        break;
      }
    }
  }
  ifs.close();
  return regions;
}

template <typename COORD_T>
std::vector<Point<COORD_T, 2>> LoadPoints(
    const std::string& path, int limit = std::numeric_limits<int>::max()) {
  std::ifstream ifs(path);
  std::string line;
  using point_type = boost::geometry::model::d2::point_xy<COORD_T>;
  using polygon_type = boost::geometry::model::polygon<point_type>;
  std::vector<Point<COORD_T, 2>> points;

  while (std::getline(ifs, line)) {
    if (!line.empty()) {
      if (line.rfind("MULTIPOLYGON", 0) == 0) {
        boost::geometry::model::multi_polygon<polygon_type> c;
        boost::geometry::read_wkt(line, c);

        for (auto& poly : c) {
          for (auto& p : poly.outer()) {
            points.push_back(Point<COORD_T, 2>(p.x(), p.y()));
          }
        }
      } else if (line.rfind("POLYGON", 0) == 0) {
        boost::geometry::model::polygon<point_type> c;
        boost::geometry::read_wkt(line, c);

        for (auto& p : c.outer()) {
          points.push_back(Point<COORD_T, 2>(p.x(), p.y()));
        }
      } else if (line.rfind("POINT", 0) == 0) {
        point_type c;
        boost::geometry::read_wkt(line, c);

        points.push_back(Point<COORD_T, 2>(c.x(), c.y()));
      } else {
        std::cerr << "Bad Geometry " << line << "\n";
        abort();
      }
      if (points.size() >= limit) {
        break;
      }
    }
  }
  ifs.close();
  return points;
}

void DumpIntersects(std::vector<thrust::pair<size_t, size_t>>& intersects,
                    const std::string& path) {
  std::sort(intersects.begin(), intersects.end(),
            [](const thrust::pair<size_t, size_t>& lhs,
               const thrust::pair<size_t, size_t>& rhs) {
              if (lhs.first != rhs.first) {
                return lhs.first < rhs.first;
              }
              return lhs.second < rhs.second;
            });
  std::ofstream ofs(path);
  for (auto& x : intersects) {
    ofs << x.first << " " << x.second << "\n";
  }
  ofs.close();
}

template <typename COORD_T>
std::vector<thrust::pair<size_t, size_t>> RunBruteForceIntersectsEnvelopeQuery(
    const std::vector<Envelope<Point<COORD_T, 2>>>& envelopes,
    const std::vector<Envelope<Point<COORD_T, 2>>>& queries) {
  std::vector<thrust::pair<size_t, size_t>> xsects;
  size_t n_intersects = 0;
  for (size_t i = 0; i < envelopes.size(); i++) {
    for (size_t j = 0; j < queries.size(); j++) {
      if (envelopes[i].Intersects(queries[j])) {
        assert(queries[j].Intersects(envelopes[i]));
        xsects.emplace_back(thrust::make_pair(i, j));
        n_intersects++;
      }
    }
  }

  return xsects;
}

template <typename COORD_T>
std::vector<thrust::pair<size_t, size_t>> RunRTreeIntersectsEnvelopeQuery(
    const std::vector<Envelope<Point<COORD_T, 2>>>& envelopes,
    const std::vector<Envelope<Point<COORD_T, 2>>>& queries) {
  std::vector<rtree_box_t> vec_boxes;
  std::vector<thrust::pair<size_t, size_t>> xsects;

  vec_boxes.resize(envelopes.size());

  for (size_t i = 0; i < envelopes.size(); i++) {
    vec_boxes[i].id = i;
    vec_boxes[i].min.x = envelopes[i].get_min().get_x();
    vec_boxes[i].min.y = envelopes[i].get_min().get_y();
    vec_boxes[i].max.x = envelopes[i].get_max().get_x();
    vec_boxes[i].max.y = envelopes[i].get_max().get_y();
  }
  Stopwatch sw;

  sw.start();
  boost::geometry::index::rtree<rtree_box_t, boost::geometry::index::rstar<16>,
                                boost::geometry::index::indexable<rtree_box_t>>
      rtree(vec_boxes);
  sw.stop();
  double t_load = sw.ms();

  sw.start();

  for (size_t i = 0; i < queries.size(); i++) {
    rtree_box_t q;

    q.min.x = queries[i].get_min().get_x();
    q.min.y = queries[i].get_min().get_y();
    q.max.x = queries[i].get_max().get_x();
    q.max.y = queries[i].get_max().get_y();
    rtree.query(boost::geometry::index::intersects(q),
                boost::make_function_output_iterator([&](const rtree_box_t& b) {
                  xsects.emplace_back(thrust::make_pair(b.id, i));
                }));
  }
  sw.stop();

  double t_query = sw.ms();
  std::cout << "Rtree, load " << t_load << " ms, query " << t_query << " ms"
            << std::endl;
  return xsects;
}

template <typename COORD_T>
std::vector<thrust::pair<size_t, size_t>> RunRTreeContainsPointQuery(
    const std::vector<Envelope<Point<COORD_T, 2>>>& envelopes,
    const std::vector<Point<COORD_T, 2>>& queries) {
  std::vector<rtree_box_t> vec_boxes;
  std::vector<thrust::pair<size_t, size_t>> xsects;

  vec_boxes.resize(envelopes.size());

  for (size_t i = 0; i < envelopes.size(); i++) {
    vec_boxes[i].id = i;
    vec_boxes[i].min.x = envelopes[i].get_min().get_x();
    vec_boxes[i].min.y = envelopes[i].get_min().get_y();
    vec_boxes[i].max.x = envelopes[i].get_max().get_x();
    vec_boxes[i].max.y = envelopes[i].get_max().get_y();
  }
  Stopwatch sw;

  sw.start();
  boost::geometry::index::rtree<rtree_box_t, boost::geometry::index::rstar<16>,
                                boost::geometry::index::indexable<rtree_box_t>>
      rtree(vec_boxes);
  sw.stop();
  double t_load = sw.ms();

  sw.start();

  for (size_t i = 0; i < queries.size(); i++) {
    rtree_point_t q;

    q.id = i;
    q.x = queries[i].get_x();
    q.y = queries[i].get_y();

    rtree.query(boost::geometry::index::contains(q),
                boost::make_function_output_iterator([&](const rtree_box_t& b) {
                  xsects.emplace_back(thrust::make_pair(b.id, i));
                }));
  }
  sw.stop();

  double t_query = sw.ms();
  std::cout << "Rtree, load " << t_load << " ms, query " << t_query << " ms"
            << std::endl;
  return xsects;
}

template <typename COORD_T>
std::vector<thrust::pair<size_t, size_t>>
RunRTreeIntersectsEnvelopeQueryParallel(
    const std::vector<Envelope<Point<COORD_T, 2>>>& envelopes,
    const std::vector<Envelope<Point<COORD_T, 2>>>& queries) {
  std::vector<rtree_box_t> vec_boxes;
  std::vector<thrust::pair<size_t, size_t>> all_xsects;

  vec_boxes.resize(envelopes.size());

  for (size_t i = 0; i < envelopes.size(); i++) {
    vec_boxes[i].id = i;
    vec_boxes[i].min.x = envelopes[i].get_min().get_x();
    vec_boxes[i].min.y = envelopes[i].get_min().get_y();
    vec_boxes[i].max.x = envelopes[i].get_max().get_x();
    vec_boxes[i].max.y = envelopes[i].get_max().get_y();
  }
  all_xsects.reserve(vec_boxes.size() * queries.size() * 0.0001);

  Stopwatch sw;

  sw.start();
  boost::geometry::index::rtree<rtree_box_t, boost::geometry::index::rstar<16>,
                                boost::geometry::index::indexable<rtree_box_t>>
      rtree(vec_boxes);
  sw.stop();
  double t_load = sw.ms();

  std::vector<std::thread> ths;
  std::mutex mu;

  int n_ths = std::thread::hardware_concurrency();
  size_t avg_queries = (queries.size() + n_ths - 1) / n_ths;

  sw.start();

  for (size_t tid = 0; tid < n_ths; tid++) {
    ths.emplace_back(std::thread(
        [&queries, &rtree, &all_xsects, &mu, avg_queries](int tid) {
          auto begin = std::min(tid * avg_queries, queries.size());
          auto end = std::min(begin + avg_queries, queries.size());
          std::vector<thrust::pair<size_t, size_t>> xsects;

          for (auto i = begin; i < end; i++) {
            rtree_box_t q;

            q.min.x = queries[i].get_min().get_x();
            q.min.y = queries[i].get_min().get_y();
            q.max.x = queries[i].get_max().get_x();
            q.max.y = queries[i].get_max().get_y();
            rtree.query(
                boost::geometry::index::intersects(q),
                boost::make_function_output_iterator([&](const rtree_box_t& b) {
                  xsects.emplace_back(thrust::make_pair(b.id, i));
                }));
          }
          {
            std::unique_lock<std::mutex> lock(mu);
            all_xsects.insert(all_xsects.end(), xsects.begin(), xsects.end());
          }
        },
        tid));
  }

  for (auto& th : ths) {
    th.join();
  }
  sw.stop();

  double t_query = sw.ms();
  std::cout << "Parallel Rtree, load " << t_load << " ms, query " << t_query
            << " ms" << std::endl;
  return all_xsects;
}

template <typename COORD_T>
pinned_vector<thrust::pair<size_t, size_t>> RunRTSpatialIntersectsEnvelopeQuery(
    const std::string& root_exec,
    const std::vector<Envelope<Point<COORD_T, 2>>>& envelopes,
    const std::vector<Envelope<Point<COORD_T, 2>>>& queries) {
  SpatialIndex<COORD_T, 2, true> index;
  Queue<thrust::pair<uint32_t, uint32_t>> results;
  thrust::device_vector<Envelope<Point<COORD_T, 2>>> d_envelopes(envelopes);
  thrust::device_vector<Envelope<Point<COORD_T, 2>>> d_queries(queries);
  pinned_vector<thrust::pair<size_t, size_t>> xsects;
  Stream stream;
  Stopwatch sw;

  Config config;
  config.ptx_root = root_exec + "/ptx";
  index.Init(config);
  results.Init(
      std::max(1000u, (uint32_t) (envelopes.size() * queries.size() * 0.01)));

  sw.start();
  index.Insert(d_envelopes, stream.cuda_stream());
  stream.Sync();
  sw.stop();

  double t_load = sw.ms();

  sw.start();
  index.IntersectsWhatQuery(ArrayView<Envelope<Point<COORD_T, 2>>>(d_queries),
                            results, stream.cuda_stream());
  size_t n_results = results.size(stream.cuda_stream());
  sw.stop();
  double t_query = sw.ms();

  std::cout << "RT, load " << t_load << " ms, query " << t_query << " ms"
            << std::endl;

  results.CopyTo(xsects, stream.cuda_stream());
  stream.Sync();

  //  std::sort(xsects.begin(), xsects.end(),
  //            [](const thrust::pair<size_t, size_t>& lhs,
  //               const thrust::pair<size_t, size_t>& rhs) {
  //              if (lhs.first != rhs.first) {
  //                return lhs.first < rhs.first;
  //              }
  //              return lhs.second < rhs.second;
  //            });

  //  auto uniq_end = std::unique(xsects.begin(), xsects.end());
  //  size_t unique_size = std::distance(xsects.begin(), uniq_end);
  //  xsects.resize(unique_size);

  return xsects;
}

void RunIntersectsEnvelopeQuery(const std::string& exec_root) {
  using point_f2d_t = Point<float, 2>;

  std::vector<Envelope<point_f2d_t>> envelopes = LoadBoxes<float>(FLAGS_box);
  std::vector<Envelope<point_f2d_t>> queries =
      LoadBoxes<float>(FLAGS_box_query);

  std::cout << "envelopes " << envelopes.size() << std::endl;
  std::cout << "queries " << queries.size() << std::endl;

  //  auto brute_force_intersects =
  //      RunBruteForceIntersectsEnvelopeQuery(envelopes, queries);
  //
  //  std::cout << "Brute force intersects " << brute_force_intersects.size()
  //            << std::endl;
  //
  //  DumpIntersects(brute_force_intersects, "/tmp/brute_force_xsects");

  {
    auto rtree_intersects = RunRTreeIntersectsEnvelopeQuery(envelopes, queries);
    std::cout << "Rtree intersects " << rtree_intersects.size() << std::endl;
    std::cout << "Select rate "
              << (float) rtree_intersects.size() /
                     (envelopes.size() * queries.size())
              << std::endl;
  }
  {
    auto parallel_rtree_intersects =
        RunRTreeIntersectsEnvelopeQueryParallel(envelopes, queries);

    std::cout << "Parllel Rtree intersects " << parallel_rtree_intersects.size()
              << std::endl;
  }
  //  DumpIntersects(rtree_intersects, "/tmp/rtree_xsects");
  {
    //    envelopes.clear();
    //    queries.clear();

    //    Envelope<point_f2d_t> e(point_f2d_t(0, 0), point_f2d_t(1, 1));
    //    Envelope<point_f2d_t> q(point_f2d_t(0.5, 0.5), point_f2d_t(0.8, 0.8));
    //
    //    envelopes.push_back(e);
    //    queries.push_back(q);

    auto rtspatial_intersects =
        RunRTSpatialIntersectsEnvelopeQuery(exec_root, envelopes, queries);
    std::cout << "RTSpatial intersects " << rtspatial_intersects.size()
              << std::endl;
  }

  //  DumpIntersects(rtspatial_intersects, "/tmp/rtspatial_xsects");

  //  for (size_t i = 0; i < rtspatial_intersects.size(); i++) {
  //    auto& x = rtspatial_intersects[i];
  //    const auto& envelope = envelopes[x.first];
  //    const auto& query = queries[x.second];
  //    assert(envelope.Intersects(query));
  //    assert(query.Intersects(envelope));
  //  }
}

template <typename COORD_T>
void BoxContainsPointQueries(
    const std::string& exec_root,
    const std::vector<Envelope<Point<COORD_T, 2>>>& boxes,
    const std::vector<Point<COORD_T, 2>>& point_queries) {
  SpatialIndex<COORD_T, 2, true> index;
  Queue<thrust::pair<uint32_t, uint32_t>> results;
  thrust::device_vector<Envelope<Point<COORD_T, 2>>> d_boxes(boxes);
  thrust::device_vector<Point<COORD_T, 2>> d_point_queries(point_queries);
  Stream stream;
  Stopwatch sw;
  Config config;

  config.ptx_root = exec_root + "/ptx";
  index.Init(config);
  results.Init(std::max(
      1000u, (uint32_t) (d_boxes.size() * d_point_queries.size() * 0.01)));

  sw.start();
  index.Insert(d_boxes, stream.cuda_stream());
  stream.Sync();
  sw.stop();
  double t_load = sw.ms();

  sw.start();
  index.ContainsWhatQuery(ArrayView<Point<COORD_T, 2>>(d_point_queries),
                          results, stream.cuda_stream());
  size_t n_results = results.size(stream.cuda_stream());
  sw.stop();
  double t_query = sw.ms();

  std::cout << "RT, load " << t_load << " ms, query " << t_query
            << " ms, results: " << n_results << std::endl;
}

}  // namespace rtspatial

#endif  // RTSPATIAL_BENCHMARK_H
