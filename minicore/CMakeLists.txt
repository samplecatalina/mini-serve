# The C++ core: KV block bookkeeping behind a pybind11 module, plus its gtest suite.
#
# Two targets, deliberately independent:
#   _minicore   the Python extension, built in place as miniserve/cache/_minicore*.so.
#               Builds offline, so a cluster job can rebuild it inside the same
#               container without network access.
#   minicore_tests  the gtest suite. Fetches googletest, so it is built only on
#               request (`make minicore-test`), never as a side effect of the module.
#
# The compiler baseline is not set here: it comes from CXXFLAGS in the image
# (-march=x86-64-v2), because -march=native would emit instructions the cluster
# nodes do not have. A build outside that image gets whatever CXXFLAGS says.

cmake_minimum_required(VERSION 3.22)
project(minicore LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(minicore_core STATIC
  src/block_allocator.cpp
  src/block_table.cpp
)
target_include_directories(minicore_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(minicore_core PRIVATE -Wall -Wextra)

# ------------------------------------------------------------------ Python module
option(MINICORE_BUILD_MODULE "Build the pybind11 extension module" ON)
if(MINICORE_BUILD_MODULE)
  find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module)
  # pybind11 ships its CMake config inside the wheel.
  execute_process(
    COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
  )
  find_package(pybind11 CONFIG REQUIRED)

  pybind11_add_module(_minicore src/bindings.cpp)
  target_link_libraries(_minicore PRIVATE minicore_core)
  target_compile_options(_minicore PRIVATE -Wall -Wextra)
  # Built in place, inside the package that imports it. The repository is used
  # from PYTHONPATH and never installed, so there is nowhere else for it to go.
  # It must not land in minicore/ itself: that directory is on the path as an
  # implicit namespace package, which would shadow the module name.
  set_target_properties(_minicore PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../miniserve/cache
  )
endif()

# ------------------------------------------------------------------ gtest
option(MINICORE_BUILD_TESTS "Build the gtest suite (downloads googletest)" OFF)
if(MINICORE_BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  enable_testing()
  add_executable(minicore_tests tests/test_block_allocator.cpp tests/test_block_table.cpp)
  target_link_libraries(minicore_tests PRIVATE minicore_core GTest::gtest_main)
  include(GoogleTest)
  gtest_discover_tests(minicore_tests)
endif()
