#include <boost/geometry.hpp>

#include "flags.h"
#include "rtspatial/spatial_index.cuh"
#include "rtspatial/utils/stream.h"

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
}  // namespace rtspatial

int main(int argc, char* argv[]) {
  //  FLAGS_stderrthreshold = 0;

  gflags::SetUsageMessage("Usage: -poly1");
  if (argc == 1) {
    gflags::ShowUsageWithFlags(argv[0]);
    exit(1);
  }
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  gflags::ShutDownCommandLineFlags();

  std::string exec_path = argv[0];
  auto exec_root = exec_path.substr(0, exec_path.find_last_of('/'));
  std::string box_path = FLAGS_box;
  std::string box_query_path = FLAGS_box_query;
  std::string point_query_path = FLAGS_point_query;
  std::string predicate = FLAGS_predicate;
  using coord_t = float;
  using namespace rtspatial;

  int limit_box =
      FLAGS_limit_box > 0 ? FLAGS_limit_box : std::numeric_limits<int>::max();
  int limit_query = FLAGS_limit_query > 0 ? FLAGS_limit_query
                                          : std::numeric_limits<int>::max();

  std::vector<Envelope<Point<coord_t, 2>>> boxes =
      LoadBoxes<coord_t>(box_path, limit_box);
  thrust::device_vector<Envelope<Point<coord_t, 2>>> d_boxes(boxes);
  std::cout << "Loaded boxes " << boxes.size() << std::endl;

  SpatialIndex<coord_t, 2, false> index;
  Config config;
  Queue<thrust::pair<uint32_t, uint32_t>> results;
  Stream stream;
  Stopwatch sw;

  config.ptx_root = PTX_ROOT;

  index.Init(config);

  sw.start();
  index.Insert(d_boxes, stream.cuda_stream());
  stream.Sync();
  sw.stop();
  double t_load = sw.ms(), t_query;
  size_t n_results;
  thrust::device_vector<dev::Queue<thrust::pair<uint32_t, uint32_t>>> d_results;

  if (!box_query_path.empty()) {
    thrust::device_vector<Envelope<Point<coord_t, 2>>> d_queries =
        LoadBoxes<coord_t>(box_query_path, limit_query);
    std::cout << "Loaded box queries " << d_queries.size() << std::endl;

    results.Init(std::max(1000u, (uint32_t) (boxes.size() * d_queries.size() *
                                             FLAGS_load_factor)));
    d_results.push_back(results.DeviceObject());
    sw.start();
    if (predicate == "contains") {
      index.ContainsWhatQuery(ArrayView<Envelope<Point<coord_t, 2>>>(d_queries),
                              thrust::raw_pointer_cast(d_results.data()),
                              stream.cuda_stream());
    } else if (predicate == "intersects") {
      index.IntersectsWhatQuery(
          ArrayView<Envelope<Point<coord_t, 2>>>(d_queries),
          thrust::raw_pointer_cast(d_results.data()), stream.cuda_stream(),
          FLAGS_parallelism);
    } else {
      std::cout << "Unsupported predicate\n";
      abort();
    }

    n_results = results.size(stream.cuda_stream());
    sw.stop();
    t_query = sw.ms();
  } else if (!point_query_path.empty()) {
    auto queries = LoadPoints<coord_t>(point_query_path, limit_query);
    thrust::device_vector<Point<coord_t, 2>> d_queries = queries;
    std::cout << "Loaded point queries " << d_queries.size() << std::endl;

    results.Init(std::max(1000u, (uint32_t) (boxes.size() * d_queries.size() *
                                             FLAGS_load_factor)));
    d_results.push_back(results.DeviceObject());

    sw.start();
    index.ContainsWhatQuery(ArrayView<Point<coord_t, 2>>(d_queries),
                            thrust::raw_pointer_cast(d_results.data()),
                            stream.cuda_stream());
    n_results = results.size(stream.cuda_stream());
    sw.stop();
    t_query = sw.ms();
  }
  std::cout << "RT, load " << t_load << " ms, query " << t_query
            << " ms, results: " << n_results << std::endl;
  return 0;
}