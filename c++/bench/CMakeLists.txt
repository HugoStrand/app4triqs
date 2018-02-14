set(BENCHMARK_SOURCES
    arithmetic_bench.cpp
    )

add_executable(benchmark_suite ${BENCHMARK_SOURCES})
target_link_libraries(benchmark_suite PUBLIC arithmetic_c triqs benchmark)

# Pubic options : any option affecting the headers should go here
target_compile_options(benchmark_suite PUBLIC)

#add_test(NAME arithmetic_bench COMMAND ${CMAKE_CURRENT_BINARY_DIR}/benchmark_suite)

add_custom_target(bench
   COMMAND ${CMAKE_CURRENT_BINARY_DIR}/benchmark_suite
   DEPENDS benchmark_suite)
