
# Copy h5 files to binary dir
file(GLOB all_h5_files *.h5)
file(COPY ${all_h5_files} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# add_test with possibly various -DXX -DYYY
macro(add_test_defs name)
 set(t "${name}${ARGV1}")
 add_executable(${t} ${name}.cpp)
 target_link_libraries(${t} PRIVATE arithmetic_c gtest)
 foreach(opt ${ARGV2})
  target_compile_definitions(${t} PRIVATE ${opt}) 
 endforeach()
 add_test(NAME ${t} COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${t})
endmacro()

add_test_defs(arithmetic_test)
