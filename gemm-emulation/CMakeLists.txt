cmake_minimum_required(VERSION 3.23)

project(mpemu
        VERSION 1.0.0
        DESCRIPTION "FP32 GEMM emulation on BF16 tensor cores (arXiv:1904.06376)"
        LANGUAGES CXX CUDA)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(MPEMU_BUILD_BENCH "Build the benchmark / accuracy driver" ON)

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

# ---------------------------------------------------------------- library ---
add_library(mpemu SHARED
    src/split.cu          # routine 1, bf16 (Henry/Tang/Heinecke)
    src/gemm.cu           # routine 2, bf16
    src/split_fp16.cu     # routine 1, multiword fp16 (Fasi et al.)
    src/gemm_fp16.cu      # routine 2, multiword fp16
    src/ozaki1.cu         # routine 1, Ozaki scheme I on INT8 (Ootomo et al.)
    src/ozaki1_gemm.cu    # routine 2, Ozaki scheme I
    src/context.cu       # one-shot workspace arena
    src/batched.cu       # batched products + single reduction
    src/util.cu)
add_library(mpemu::mpemu ALIAS mpemu)

target_include_directories(mpemu
    PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# cublas is part of the public interface: mpemuGemmBF16 takes a cublasHandle_t.
target_link_libraries(mpemu PUBLIC CUDA::cudart CUDA::cublas)

# NOTE: fast math must stay OFF. The split relies on `a - float(bf16(a))`
# being computed exactly; -ffast-math style contraction would break it.
# It is off by default in nvcc, so we only assert it here for clarity.
target_compile_options(mpemu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

# The Cray programming environment on Perlmutter injects `-lgdrapi -lmpich`
# into every nvcc invocation via NVCC_PREPEND_FLAGS. --as-needed drops those
# (and anything else unreferenced) so libmpemu.so depends only on cudart and
# cublas, which is what "standalone" should mean for a consumer.
target_link_options(mpemu PRIVATE "LINKER:--as-needed")

set_target_properties(mpemu PROPERTIES
    VERSION   ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION OFF
    C_VISIBILITY_PRESET   hidden
    CXX_VISIBILITY_PRESET hidden
    CUDA_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/include/mpemu/mpemu.h)

# ------------------------------------------------------------------ bench ---
if(MPEMU_BUILD_BENCH)
    add_executable(mpemu_bench bench/bench.cu)
    target_link_libraries(mpemu_bench PRIVATE mpemu::mpemu CUDA::cublas CUDA::cudart)
    set_target_properties(mpemu_bench PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
endif()

# ---------------------------------------------------------------- install ---
install(TARGETS mpemu
        EXPORT mpemuTargets
        LIBRARY       DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE       DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME       DESTINATION ${CMAKE_INSTALL_BINDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mpemu)

install(EXPORT mpemuTargets
        FILE mpemuTargets.cmake
        NAMESPACE mpemu::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mpemu)

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/mpemuConfig.cmake.in
"@PACKAGE_INIT@\ninclude(CMakeFindDependencyMacro)\nfind_dependency(CUDAToolkit)\ninclude(\"\${CMAKE_CURRENT_LIST_DIR}/mpemuTargets.cmake\")\n")

configure_package_config_file(
    ${CMAKE_CURRENT_BINARY_DIR}/mpemuConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/mpemuConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mpemu)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/mpemuConfigVersion.cmake
    COMPATIBILITY SameMajorVersion)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/mpemuConfig.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/mpemuConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mpemu)

# ------------------------------------------------------------------- test ---
if(MPEMU_BUILD_BENCH)
    enable_testing()
    add_executable(mpemu_test_split bench/test_split.cu)
    target_link_libraries(mpemu_test_split PRIVATE mpemu::mpemu CUDA::cudart)
    add_test(NAME split_exactness COMMAND mpemu_test_split)

    add_executable(mpemu_test_split_fp16 bench/test_split_fp16.cu)
    target_link_libraries(mpemu_test_split_fp16 PRIVATE mpemu::mpemu CUDA::cudart)
    add_test(NAME split_fp16 COMMAND mpemu_test_split_fp16)

    add_executable(mpemu_test_refine bench/test_refine.cu)
    target_link_libraries(mpemu_test_refine PRIVATE mpemu::mpemu CUDA::cublas CUDA::cudart)
    add_test(NAME refine_bitwise COMMAND mpemu_test_refine)

    add_executable(mpemu_test_ozaki1 bench/test_ozaki1.cu)
    target_link_libraries(mpemu_test_ozaki1 PRIVATE mpemu::mpemu CUDA::cublas CUDA::cudart)
    add_test(NAME ozaki1 COMMAND mpemu_test_ozaki1)

    add_executable(mpemu_bench_oz1 bench/bench_oz1.cu)
    target_link_libraries(mpemu_bench_oz1 PRIVATE mpemu::mpemu CUDA::cublas CUDA::cudart)

    # All three schemes, both distributions, one CSV for the plots.
    add_executable(mpemu_bench_all bench/bench_all.cu)
    target_link_libraries(mpemu_bench_all PRIVATE mpemu::mpemu CUDA::cublas CUDA::cudart)

    # Steady-state loop: A split once, only the small k x n operand re-split.
    add_executable(mpemu_bench_iter bench/bench_iter.cu)
    target_link_libraries(mpemu_bench_iter PRIVATE mpemu::mpemu CUDA::cublas CUDA::cudart)

    # Attributes forward error to accumulation vs decomposition.
    add_executable(mpemu_accum_probe bench/accum_probe.cu)
    target_link_libraries(mpemu_accum_probe PRIVATE mpemu::mpemu CUDA::cublas CUDA::cudart)

    # Multiword fp16 scheme vs both cublasSgemm and cublasDgemm.
    add_executable(mpemu_bench_mw bench/bench_mw.cu)
    target_link_libraries(mpemu_bench_mw PRIVATE mpemu::mpemu CUDA::cublas CUDA::cudart)
endif()
