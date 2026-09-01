cmake_minimum_required(VERSION 3.23)

project(mpkmeans
        VERSION 0.1.0
        DESCRIPTION "Mixed precision K-Means with cluster exclusion bounds"
        LANGUAGES CXX CUDA)

if(NOT DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# A100 (Perlmutter). Override with -DCMAKE_CUDA_ARCHITECTURES=...
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

# Statistics collection (exclusion attribution counters and the FP64 oracle)
# costs kernel time and is only needed when measuring, not when timing.
option(MPK_STATS "collect exclusion statistics and enable the FP64 oracle" OFF)

add_library(mpkmeans STATIC
    src/kernels.cu
    src/mpkmeans.cu
    src/data.cu)
add_library(mpkmeans::mpkmeans ALIAS mpkmeans)

target_include_directories(mpkmeans
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(mpkmeans
    PUBLIC CUDA::cudart CUDA::cublas CUDA::cusparse)

if(MPK_STATS)
    target_compile_definitions(mpkmeans PUBLIC MPK_STATS)
endif()

# The exclusion tests rely on the exact FP32 semantics of the comparisons and
# on `-2*g + cnorm2` being a single fma. Contraction beyond that (fast math)
# would invalidate the bounds, so leave it off.
target_compile_options(mpkmeans PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

set_target_properties(mpkmeans PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION OFF)

add_executable(mpkmeans_bench bench/main.cu)
target_link_libraries(mpkmeans_bench PRIVATE mpkmeans::mpkmeans)
set_target_properties(mpkmeans_bench PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

# Perlmutter's Cray PE injects -lgdrapi -lmpich into every nvcc invocation.
target_link_options(mpkmeans_bench PRIVATE "LINKER:--as-needed")
