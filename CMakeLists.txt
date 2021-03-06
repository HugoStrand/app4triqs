# Version number 
set (APP4TRIQS_VERSION "0.1")

# start configuration
cmake_minimum_required(VERSION 2.8)
project(app4triqs CXX)

# Default to Release build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Type of build" FORCE)
endif()
message( STATUS "-------- BUILD-TYPE: ${CMAKE_BUILD_TYPE} -------------")

# Use shared libraries
set(BUILD_SHARED_LIBS ON)

# Load TRIQS and Cpp2Py
find_package(TRIQS 1.5 EXACT REQUIRED)
find_package(Cpp2Py REQUIRED)

# We will need the FindNNFT of triqs, so we adjust CMake Module Path to find it
list(APPEND CMAKE_MODULE_PATH ${TRIQS_ROOT}/share/cmake)

# Default Install directory to TRIQS_ROOT if not given. Checks an absolute name is given.
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT OR (NOT IS_ABSOLUTE ${CMAKE_INSTALL_PREFIX}))
  message(STATUS " No install prefix given (or invalid). Defaulting to TRIQS_ROOT")
  set(CMAKE_INSTALL_PREFIX ${TRIQS_ROOT} CACHE PATH "default install path" FORCE)
endif()
message(STATUS "-------- CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX} -------------")

# Macro defined in TRIQS which picks the hash of repo.
triqs_get_git_hash_of_source_dir(APP4TRIQS_GIT_HASH)
message(STATUS "Git hash: ${APP4TRIQS_GIT_HASH}")

option(Tests "Enable Tests" ON)
option(Documentation "Build documentation" OFF)

# All PRIVATE common options.
# The std for all targets
add_compile_options(-std=c++17)

# Common to all targets (c++, python, tests).
include_directories(c++/app4triqs)

# To be fixed!!
triqs_set_rpath_for_all_targets()

add_subdirectory(c++/app4triqs)

# Tests
if (${Tests})
 enable_testing()
 add_subdirectory(test/c++)
 add_subdirectory(test/python)
endif()

if (${TRIQS_WITH_PYTHON_SUPPORT})

 # Python interface
 add_subdirectory(python/triqs_app4triqs)

 # Build the documentation
 if (${Documentation})
   add_subdirectory(doc)
 endif()

endif()



