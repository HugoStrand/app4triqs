
#include "app4triqs/arithmetic.hpp"

TEST(arithmetic, power_of_two) {
  EXPECT_EQ(power_of_two(2.), 4.);
}

MAKE_MAIN;
