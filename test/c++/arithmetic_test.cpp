
#include <triqs/test_tools/gfs.hpp>

#include "arithmetic.hpp"

using namespace app4triqs;

TEST(arithmetic, power_of_two) {
  EXPECT_EQ(power_of_two(2.), 4.);
}

MAKE_MAIN;
