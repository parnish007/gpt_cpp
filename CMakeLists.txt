cmake_minimum_required(VERSION 3.18)
project(gpt_cpp VERSION 1.0.0 LANGUAGES CXX)

# ── C++ standard ────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Build type ───────────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ── Compiler flags ───────────────────────────────────────────────────────────
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CMAKE_CXX_FLAGS_Debug   "-g -fsanitize=address,undefined")

# ── Optional: OpenMP for multi-threaded matmul ───────────────────────────────
find_package(OpenMP)

# ── Include directories ──────────────────────────────────────────────────────
include_directories(include)

# ── Core library (shared between trainer and inference engine) ────────────────
file(GLOB_RECURSE CORE_SRC      src/core/*.cpp)
file(GLOB_RECURSE TRAINING_SRC  src/training/*.cpp)
file(GLOB_RECURSE INFERENCE_SRC src/inference/*.cpp)
file(GLOB_RECURSE UTILS_SRC     src/utils/*.cpp)

add_library(gpt_core STATIC
    ${CORE_SRC}
    ${UTILS_SRC}
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(gpt_core PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(gpt_core PUBLIC USE_OPENMP)
endif()

# ── Training executable ──────────────────────────────────────────────────────
add_executable(train_gpt
    ${TRAINING_SRC}
    tools/train_main.cpp
)
target_link_libraries(train_gpt PRIVATE gpt_core)

# ── Inference engine executable ──────────────────────────────────────────────
add_executable(inference_engine
    ${INFERENCE_SRC}
    tools/inference_main.cpp
)
target_link_libraries(inference_engine PRIVATE gpt_core)

# ── Benchmark executable ──────────────────────────────────────────────────────
add_executable(benchmark tools/benchmark.cpp)
target_link_libraries(benchmark PRIVATE gpt_core)

# ── Tests ────────────────────────────────────────────────────────────────────
enable_testing()
file(GLOB TEST_SRCS tests/*.cpp)
foreach(test_src ${TEST_SRCS})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PRIVATE gpt_core)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

# ── Install ──────────────────────────────────────────────────────────────────
install(TARGETS train_gpt inference_engine DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
