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

# The multiword (double-fp16) driver needs the mpemu split/GEMM routines from
# the sibling gemm-emulation project.  Found by default, so a normal in-tree
# build picks it up; the driver compiles to a stub that reports the missing
# dependency when it is not.
option(MPK_MULTIWORD "build the multiword (mpemu) driver" ON)
set(MPEMU_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../gemm-emulation"
    CACHE PATH "root of the gemm-emulation project")

# The cuVS FP32 baseline is an outside implementation used for scale.  Default
# OFF: it needs a RAPIDS install, which is not part of this tree, and the driver
# compiles to a stub reporting the missing dependency when it is absent.
#   cmake -DMPK_CUVS=ON -DCUVS_ROOT=/path/to/env ...
option(MPK_CUVS "build the cuVS (RAPIDS) FP32 k-means baseline" OFF)
set(CUVS_ROOT "" CACHE PATH
    "prefix holding include/cuvs, include/dlpack and lib/libcuvs_c.so")

add_library(mpkmeans STATIC
    src/kernels.cu
    src/baseline.cu
    src/multiword.cu
    src/cuvs_baseline.cu
    src/mpkmeans.cu
    src/data.cu)
add_library(mpkmeans::mpkmeans ALIAS mpkmeans)

if(MPK_MULTIWORD)
    find_library(MPEMU_LIBRARY mpemu HINTS "${MPEMU_ROOT}/build")
    find_path(MPEMU_INCLUDE_DIR mpemu/mpemu.h HINTS "${MPEMU_ROOT}/include")
    if(MPEMU_LIBRARY AND MPEMU_INCLUDE_DIR)
        message(STATUS "mpkmeans: multiword driver on (${MPEMU_LIBRARY})")
        target_compile_definitions(mpkmeans PUBLIC MPK_MULTIWORD)
        target_include_directories(mpkmeans PRIVATE "${MPEMU_INCLUDE_DIR}")
        target_link_libraries(mpkmeans PUBLIC "${MPEMU_LIBRARY}")
        # libmpemu.so lives outside any default search path
        set_target_properties(mpkmeans PROPERTIES
            INTERFACE_LINK_DIRECTORIES "${MPEMU_ROOT}/build")
    else()
        message(STATUS "mpkmeans: multiword driver OFF -- mpemu not found "
                       "under ${MPEMU_ROOT} (set -DMPEMU_ROOT=...)")
    endif()
endif()

if(MPK_CUVS)
    find_library(CUVS_C_LIBRARY cuvs_c HINTS "${CUVS_ROOT}/lib")
    find_library(CUVS_LIBRARY   cuvs   HINTS "${CUVS_ROOT}/lib")
    find_path(CUVS_INCLUDE_DIR  cuvs/cluster/kmeans.h HINTS "${CUVS_ROOT}/include")
    find_path(DLPACK_INCLUDE_DIR dlpack/dlpack.h HINTS "${CUVS_ROOT}/include")
    if(CUVS_C_LIBRARY AND CUVS_LIBRARY AND CUVS_INCLUDE_DIR AND DLPACK_INCLUDE_DIR)
        message(STATUS "mpkmeans: cuVS baseline on (${CUVS_C_LIBRARY})")
        target_compile_definitions(mpkmeans PUBLIC MPK_CUVS)
        target_include_directories(mpkmeans PRIVATE
            "${CUVS_INCLUDE_DIR}" "${DLPACK_INCLUDE_DIR}")
        target_link_libraries(mpkmeans PUBLIC "${CUVS_C_LIBRARY}" "${CUVS_LIBRARY}")
        # the RAPIDS libraries live outside any default search path
        set_target_properties(mpkmeans PROPERTIES
            INTERFACE_LINK_DIRECTORIES "${CUVS_ROOT}/lib")
    else()
        message(STATUS "mpkmeans: cuVS baseline OFF -- not found under "
                       "${CUVS_ROOT} (set -DCUVS_ROOT=...)")
    endif()
endif()

target_include_directories(mpkmeans
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(mpkmeans
    PUBLIC CUDA::cudart CUDA::cublas)

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

# libmpemu.so is not on any default search path, so bake its location into the
# binary rather than making every run depend on LD_LIBRARY_PATH.
set(MPK_BENCH_RPATH "")
if(MPK_MULTIWORD AND MPEMU_LIBRARY)
    list(APPEND MPK_BENCH_RPATH "${MPEMU_ROOT}/build")
endif()
if(MPK_CUVS AND CUVS_C_LIBRARY)
    list(APPEND MPK_BENCH_RPATH "${CUVS_ROOT}/lib")
endif()
if(MPK_BENCH_RPATH)
    set_target_properties(mpkmeans_bench PROPERTIES BUILD_RPATH "${MPK_BENCH_RPATH}")
endif()

# Perlmutter's Cray PE injects -lgdrapi -lmpich into every nvcc invocation.
target_link_options(mpkmeans_bench PRIVATE "LINKER:--as-needed")
