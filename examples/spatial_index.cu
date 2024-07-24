#include <boost/geometry.hpp>

#include "flags.h"
#include "rtspatial/spatial_index.cuh"
#include "rtspatial/utils/stream.h"
#include "wkt_loader.h"

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
  std::string serialize = FLAGS_serialize;
  std::string point_query_path = FLAGS_point_query;
  std::string predicate = FLAGS_predicate;
  using namespace rtspatial;

  int limit_box =
      FLAGS_limit_box > 0 ? FLAGS_limit_box : std::numeric_limits<int>::max();
  int limit_query = FLAGS_limit_query > 0 ? FLAGS_limit_query
                                          : std::numeric_limits<int>::max();

  auto boxes = LoadBoxes(box_path, serialize, limit_box);
  thrust::device_vector<Envelope<Point<coord_t, 2>>> d_boxes;
  std::cout << "Loaded boxes " << boxes.size() << std::endl;

  CopyBoxes(boxes, d_boxes);

  SpatialIndex<coord_t, 2, false> index;
  Config config;
  Stream stream;
  Stopwatch sw;

  config.ptx_root = PTX_ROOT;
  config.prefer_fast_build_query = false;

  index.Init(config);

  sw.start();
  index.Insert(d_boxes, stream.cuda_stream());
  stream.Sync();
  sw.stop();
  double t_load = sw.ms(), t_query;
  size_t n_results;
  SharedValue<unsigned long long int> counter;

  counter.set(stream.cuda_stream(), 0);

  if (!box_query_path.empty()) {
    auto queries = LoadBoxes(box_query_path, limit_query);
    thrust::device_vector<Envelope<Point<coord_t, 2>>> d_queries;
    std::cout << "Loaded box queries " << queries.size() << std::endl;

    CopyBoxes(queries, d_queries);

    ArrayView<Envelope<Point<coord_t, 2>>> v_queries(d_queries);

    sw.start();
    if (predicate == "contains") {
      index.ContainsWhatQuery(v_queries, counter.data(), stream.cuda_stream());
    } else if (predicate == "intersects") {
      int best_parallelism =
          index.CalculateBestParallelism(v_queries, stream.cuda_stream());

      index.IntersectsWhatQuery(v_queries, counter.data(), stream.cuda_stream(),
                                best_parallelism);
      n_results = counter.get(stream.cuda_stream());
    } else {
      std::cout << "Unsupported predicate\n";
      abort();
    }

    n_results = counter.get(stream.cuda_stream());
    sw.stop();
    t_query = sw.ms();
  } else if (!point_query_path.empty()) {
    auto queries = LoadPoints(point_query_path, limit_query);
    thrust::device_vector<Point<coord_t, 2>> d_queries;
    std::cout << "Loaded point queries " << queries.size() << std::endl;

    CopyPoints(queries, d_queries);

    sw.start();
    index.ContainsWhatQuery(ArrayView<Point<coord_t, 2>>(d_queries),
                            counter.data(), stream.cuda_stream());
    n_results = counter.get(stream.cuda_stream());
    sw.stop();
    t_query = sw.ms();
  }
  std::cout << "RT, load " << t_load << " ms, query " << t_query
            << " ms, results: " << n_results << std::endl;
  return 0;
}