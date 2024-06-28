#include "rtspatial/spatial_index.cuh"
#include "rtspatial/utils/stream.h"
#include "benchmark.h"
#include "flags.h"

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

  rtspatial::RunIntersectsEnvelopeQuery(exec_root);
}