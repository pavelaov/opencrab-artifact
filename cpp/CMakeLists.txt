# OpenCrab v2 C++17 mirror: build with CMake and the system compiler.
#
# The C++ tree has two build drivers that compile the same sources with the
# same flags (-std=c++17 -O2 -ffp-contract=off):
#   * this file, the driver on Ubuntu 22.04 (GCC 11.4, CMake 3.22);
#   * build.py, the driver on Windows, where there is neither a system compiler
#     nor CMake: it runs the clang of the Python ziglang package.
# Both build one executable per apps/*.cpp and tests/*.cpp on top of the three
# layers (here three static libraries, in build.py the object files).
#
# Run from the root of the tree:
#   cmake -S cpp -B cpp/build                      Release is the default build type
#   cmake --build cpp/build --parallel
#   ctest --test-dir cpp/build --output-on-failure      every test executable, judged by its exit status
#   cmake --build cpp/build --target tests         the same tests through one target that prints ONE
#                                                  line "[tests] SUMMARY: ..." like "python cpp/build.py tests"
#
# Where the executables go. Always cpp/build/bin (OPENCRAB2_BIN_DIR), whatever
# build directory -B names: python/tests, scripts/ and cpp/tests/*.py look for
# the binaries there.
# -DOPENCRAB2_BIN_DIR=<dir> puts them elsewhere (pass --bin <dir> to the Python
# helpers then). Every test runs with the root of the tree as its working
# directory: results/calibration and cpp/tests/vectors are found from there.
#
# Flags. Release is -O2, the flag of build.py (CMake's own Release default would
# be -O3 -DNDEBUG), so both drivers time the same configuration. With GCC and
# Clang every source is compiled with -ffp-contract=off: a multiplication and
# an addition stay two IEEE 754 operations on every compiler and CPU, so the
# cost-model predictions written into tests/vectors/cpp_costmodel_vectors.json
# are the same doubles under both drivers (python/tests compares them exactly).
#
# Source manifest (same rule as build.py). Library sources and targets are
# found by globbing, so a new file is built without editing this file.
# OPENCRAB2_EXPECTED_SOURCES names files the tree is expected to carry; a
# missing one is reported as "SKIP <path>: source file missing" at configure
# time and the configuration goes on. A target whose own source or one of its
# required library sources is missing is skipped instead of failing the build.
#
# Code identity. Every app and test is compiled with
# OPENCRAB2_CODE_COMMIT=cpp-tree-<sha256>, the content hash of include/**/*.hpp,
# src/**/*.cpp and apps/*.cpp that build.py computes too (one line
# "<sha256 of the file>  <relative path>" per file in sorted path order, then
# sha256 of those lines). It is computed at configure time: reconfigure after
# editing sources before producing evidence records.
#
# Hardware AES. src/lowlevel/aes128.cpp carries an AES-NI path next to the portable
# one. Its functions are compiled with a per-function target attribute (GCC, Clang)
# or plain intrinsics (MSVC), so no -maes flag is set here, the same as in build.py;
# the path is selected at run time by CPUID and OPENCRAB2_AES=portable in the
# environment forces the portable path (run the tests once that way as well):
#   OPENCRAB2_AES=portable ctest --test-dir cpp/build --output-on-failure
# -DOPENCRAB2_AES_NO_HARDWARE (CMAKE_CXX_FLAGS) removes the hardware path.
cmake_minimum_required(VERSION 3.16)
project(opencrab2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set(CMAKE_CXX_FLAGS_RELEASE "-O2")
  add_compile_options(-ffp-contract=off)
endif()

option(OPENCRAB2_WARNINGS "Compile with -Wall -Wextra" OFF)
set(OPENCRAB2_BIN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/build/bin" CACHE PATH
    "Directory of the executables (the Python tests and scripts look in cpp/build/bin)")

set(OPENCRAB2_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ---------------------------------------------------------------------------
# manifest: expected sources and per-target requirements
# ---------------------------------------------------------------------------
set(OPENCRAB2_EXPECTED_SOURCES
    src/highlevel/bridge.cpp
    tests/test_bridge.cpp
    tests/test_ledger.cpp)
# Library sources a target cannot link without (beyond its own main file).
set(OPENCRAB2_REQUIRES_test_bridge src/highlevel/bridge.cpp)

set(OPENCRAB2_SKIPPED_TESTS "")
foreach(rel ${OPENCRAB2_EXPECTED_SOURCES})
  if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${rel})
    message(STATUS "SKIP ${rel}: source file missing")
    get_filename_component(missing_name ${rel} NAME_WE)
    if(missing_name MATCHES "^test_")
      list(APPEND OPENCRAB2_SKIPPED_TESTS ${missing_name})
    endif()
  endif()
endforeach()

# ---------------------------------------------------------------------------
# code identity (the algorithm of build.py tree_hash)
# ---------------------------------------------------------------------------
file(GLOB_RECURSE OPENCRAB2_HASH_HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS
     ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)
file(GLOB_RECURSE OPENCRAB2_HASH_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS
     ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB OPENCRAB2_HASH_APPS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS
     ${CMAKE_CURRENT_SOURCE_DIR}/apps/*.cpp)
set(OPENCRAB2_HASH_FILES ${OPENCRAB2_HASH_HEADERS} ${OPENCRAB2_HASH_SOURCES} ${OPENCRAB2_HASH_APPS})
list(SORT OPENCRAB2_HASH_FILES)
set(OPENCRAB2_HASH_LINES "")
foreach(rel ${OPENCRAB2_HASH_FILES})
  file(SHA256 ${CMAKE_CURRENT_SOURCE_DIR}/${rel} file_digest)
  string(APPEND OPENCRAB2_HASH_LINES "${file_digest}  ${rel}\n")
endforeach()
string(SHA256 OPENCRAB2_TREE_DIGEST "${OPENCRAB2_HASH_LINES}")
set(OPENCRAB2_CODE_COMMIT "cpp-tree-${OPENCRAB2_TREE_DIGEST}")
message(STATUS "code_commit=${OPENCRAB2_CODE_COMMIT}")

# ---------------------------------------------------------------------------
# libraries
# ---------------------------------------------------------------------------
function(opencrab2_layer name)
  file(GLOB sources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/${name}/*.cpp)
  add_library(opencrab2_${name} STATIC ${sources})
  target_include_directories(opencrab2_${name} PUBLIC ${OPENCRAB2_INCLUDE})
  if(OPENCRAB2_WARNINGS AND NOT MSVC)
    target_compile_options(opencrab2_${name} PRIVATE -Wall -Wextra)
  endif()
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(opencrab2_${name} PRIVATE OPENCRAB2_DEBUG=1)
  endif()
endfunction()

opencrab2_layer(lowlevel)
opencrab2_layer(highlevel)
opencrab2_layer(platform)
target_link_libraries(opencrab2_highlevel PUBLIC opencrab2_lowlevel)
target_link_libraries(opencrab2_platform PUBLIC opencrab2_highlevel opencrab2_lowlevel)

# std::thread. Where the C library keeps the thread functions in a library of
# their own (glibc before 2.34) the executables get -pthread at compile and
# link time; on Ubuntu 22.04 (glibc 2.35) nothing has to be added.
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
set(OPENCRAB2_SYSTEM_LIBS Threads::Threads)
if(WIN32)
  list(APPEND OPENCRAB2_SYSTEM_LIBS ws2_32)
endif()

# ---------------------------------------------------------------------------
# executables: every one links all three layers (same rule as build.py)
# ---------------------------------------------------------------------------
# Sets ${out_var} to TRUE when the target was added, FALSE when it was skipped
# because a library source it requires is missing.
function(opencrab2_executable source out_var)
  get_filename_component(name ${source} NAME_WE)
  set(${out_var} FALSE PARENT_SCOPE)
  foreach(req ${OPENCRAB2_REQUIRES_${name}})
    if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${req})
      message(STATUS "SKIP target ${name}: requires ${req} (missing)")
      return()
    endif()
  endforeach()
  add_executable(${name} ${source})
  target_include_directories(${name} PRIVATE ${OPENCRAB2_INCLUDE})
  target_link_libraries(${name} PRIVATE opencrab2_platform opencrab2_highlevel opencrab2_lowlevel ${OPENCRAB2_SYSTEM_LIBS})
  # stamped into the main translation unit only, never into the libraries (platform/evidence.hpp)
  target_compile_definitions(${name} PRIVATE OPENCRAB2_CODE_COMMIT=${OPENCRAB2_CODE_COMMIT})
  if(OPENCRAB2_WARNINGS AND NOT MSVC)
    target_compile_options(${name} PRIVATE -Wall -Wextra)
  endif()
  set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${OPENCRAB2_BIN_DIR})
  set(${out_var} TRUE PARENT_SCOPE)
endfunction()

file(GLOB OPENCRAB2_APPS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/apps/*.cpp)
foreach(app ${OPENCRAB2_APPS})
  opencrab2_executable(${app} app_added)
endforeach()

# CTest judges a test by its exit status: every test executable exits non-zero
# on a failed check, and so does one that crashes after its "PASS <n> checks"
# line.
enable_testing()
set(OPENCRAB2_TEST_TARGETS "")
set(OPENCRAB2_TEST_EXES "")
file(GLOB OPENCRAB2_TESTS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp)
foreach(test_source ${OPENCRAB2_TESTS})
  get_filename_component(test_name ${test_source} NAME_WE)
  opencrab2_executable(${test_source} test_added)
  if(NOT test_added)
    list(APPEND OPENCRAB2_SKIPPED_TESTS ${test_name})
    continue()
  endif()
  list(APPEND OPENCRAB2_TEST_TARGETS ${test_name})
  if(OPENCRAB2_TEST_EXES STREQUAL "")
    set(OPENCRAB2_TEST_EXES "$<TARGET_FILE:${test_name}>")
  else()
    string(APPEND OPENCRAB2_TEST_EXES "|$<TARGET_FILE:${test_name}>")
  endif()
  if(test_name STREQUAL "test_vectors")
    add_test(NAME ${test_name} COMMAND ${test_name} ${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors
             WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..)
  else()
    add_test(NAME ${test_name} COMMAND ${test_name} WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..)
  endif()
  set_tests_properties(${test_name} PROPERTIES TIMEOUT 3600)
endforeach()

# ---------------------------------------------------------------------------
# the single "tests" target: run every test executable, print one summary line
# ---------------------------------------------------------------------------
# The runner is a CMake script written at configure time (no extra file in the
# source tree). A test passes when it exits 0 AND printed "PASS <n> checks";
# the runner tallies the checks like build.py run_tests.
file(WRITE ${CMAKE_BINARY_DIR}/opencrab2_run_tests.cmake [==[
# Generated by cpp/CMakeLists.txt. Arguments (-D): OPENCRAB2_TEST_EXES ("|" separated paths),
# OPENCRAB2_VECTORS_DIR, OPENCRAB2_WORK_DIR, OPENCRAB2_SKIPPED_TESTS ("|" separated names).
string(REPLACE "|" ";" exes "${OPENCRAB2_TEST_EXES}")
string(REPLACE "|" ";" skipped "${OPENCRAB2_SKIPPED_TESTS}")
set(passed 0)
set(failed 0)
set(total 0)
set(checks 0)
set(failed_names "")
foreach(exe ${exes})
  get_filename_component(name "${exe}" NAME_WE)
  set(test_args "")
  if(name STREQUAL "test_vectors")
    set(test_args "${OPENCRAB2_VECTORS_DIR}")
  endif()
  execute_process(COMMAND "${exe}" ${test_args}
                  WORKING_DIRECTORY "${OPENCRAB2_WORK_DIR}"
                  RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err TIMEOUT 3600)
  math(EXPR total "${total} + 1")
  string(REGEX MATCHALL "PASS [0-9]+ checks" pass_lines "${out}")
  list(LENGTH pass_lines n_pass_lines)
  if("${rc}" STREQUAL "0" AND n_pass_lines GREATER 0)
    list(GET pass_lines -1 last_line)
    string(REGEX REPLACE "^PASS ([0-9]+) checks$" "\\1" n_checks "${last_line}")
    math(EXPR checks "${checks} + ${n_checks}")
    math(EXPR passed "${passed} + 1")
    execute_process(COMMAND "${CMAKE_COMMAND}" -E echo "  OK   ${name}: ${last_line}")
  else()
    math(EXPR failed "${failed} + 1")
    list(APPEND failed_names "${name}")
    execute_process(COMMAND "${CMAKE_COMMAND}" -E echo "  FAIL ${name} (exit ${rc})")
    message(STATUS "${out}")
    message(STATUS "${err}")
  endif()
endforeach()
list(LENGTH skipped n_skipped)
set(detail "")
if(failed GREATER 0)
  string(REPLACE ";" "," failed_joined "${failed_names}")
  set(detail "${detail} failed=[${failed_joined}]")
endif()
if(n_skipped GREATER 0)
  string(REPLACE ";" "," skipped_joined "${skipped}")
  set(detail "${detail} skipped=[${skipped_joined}]")
endif()
execute_process(COMMAND "${CMAKE_COMMAND}" -E echo
                "[tests] SUMMARY: ${passed}/${total} executables passed, ${failed} failed, ${n_skipped} skipped, ${checks} checks${detail}")
if(failed GREATER 0 OR total EQUAL 0)
  message(FATAL_ERROR "tests failed")
endif()
]==])

string(REPLACE ";" "|" OPENCRAB2_SKIPPED_TESTS_JOINED "${OPENCRAB2_SKIPPED_TESTS}")
add_custom_target(tests
  COMMAND ${CMAKE_COMMAND}
          "-DOPENCRAB2_TEST_EXES=${OPENCRAB2_TEST_EXES}"
          "-DOPENCRAB2_VECTORS_DIR=${CMAKE_CURRENT_SOURCE_DIR}/tests/vectors"
          "-DOPENCRAB2_WORK_DIR=${CMAKE_CURRENT_SOURCE_DIR}/.."
          "-DOPENCRAB2_SKIPPED_TESTS=${OPENCRAB2_SKIPPED_TESTS_JOINED}"
          -P ${CMAKE_BINARY_DIR}/opencrab2_run_tests.cmake
  DEPENDS ${OPENCRAB2_TEST_TARGETS}
  USES_TERMINAL
  VERBATIM
  COMMENT "Running every OpenCrab v2 test executable")
