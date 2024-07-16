#include "benchmark.h"
#include "flags.h"
#include "rtspatial/spatial_index.cuh"
#include "rtspatial/utils/stream.h"

namespace rtspatial {}

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

  SpatialIndex<coord_t, 2, true> index;
  Queue<thrust::pair<size_t, size_t>> results;
  Stream stream;
  Stopwatch sw;

  index.Init(exec_root);

  sw.start();
  index.Insert(d_boxes, stream.cuda_stream());
  stream.Sync();
  sw.stop();
  double t_load = sw.ms(), t_query;
  size_t n_results;

  if (!box_query_path.empty()) {
    thrust::device_vector<Envelope<Point<coord_t, 2>>> d_queries =
        LoadBoxes<coord_t>(box_query_path, limit_query);
    std::cout << "Loaded box queries " << d_queries.size() << std::endl;

    results.Init(std::max(1000u, (uint32_t) (d_boxes.size() * d_queries.size() *
                                             FLAGS_load_factor)));

    sw.start();
    if (predicate == "contains") {
      index.ContainsWhatQuery(ArrayView<Envelope<Point<coord_t, 2>>>(d_queries),
                              results, stream.cuda_stream());
    } else if (predicate == "intersects") {
      index.IntersectsWhatQuery(
          ArrayView<Envelope<Point<coord_t, 2>>>(d_queries), results,
          stream.cuda_stream());
    } else if (predicate == "intersects1") {
      index.IntersectsWhatQueryLB(
          ArrayView<Envelope<Point<coord_t, 2>>>(d_queries), results,
          FLAGS_parallelism, stream.cuda_stream());
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

    results.Init(std::max(1000u, (uint32_t) (d_boxes.size() * d_queries.size() *
                                             FLAGS_load_factor)));

    sw.start();
    index.ContainsWhatQuery(ArrayView<Point<coord_t, 2>>(d_queries), results,
                            stream.cuda_stream());
    n_results = results.size(stream.cuda_stream());
    sw.stop();
    t_query = sw.ms();

    auto rtree_results = RunRTreeContainsPointQuery<coord_t>(boxes, queries);
    std::cout << "rtree_results " << rtree_results.size() << "\n";
  }
  std::cout << "RT, load " << t_load << " ms, query " << t_query
            << " ms, results: " << n_results << std::endl;
}