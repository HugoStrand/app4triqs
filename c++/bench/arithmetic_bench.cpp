
#include <benchmark/benchmark.h>

#include "arithmetic.hpp"

using namespace app4triqs;

static void BM_power_of_two(benchmark::State& state) {
  double x = 2.0;
  for (auto _ : state) {
    double x = power_of_two(x);
  }
}
BENCHMARK(BM_power_of_two);

BENCHMARK_MAIN();
