#include <gtest/gtest.h>

#include "test_commons.h"
#include "test_envelope_queries.h"
#include "test_point_queries.h"

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  std::string exec_path = argv[0];
  ptx_root = PTX_ROOT;

  return RUN_ALL_TESTS();
}
