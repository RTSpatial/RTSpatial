#include <gtest/gtest.h>

#include "test_commons.h"
#include "test_envelope_queries.h"
#include "test_point_queries.h"

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  std::string exec_path = argv[0];
  exec_root = exec_path.substr(0, exec_path.find_last_of('/'));

  return RUN_ALL_TESTS();
}
