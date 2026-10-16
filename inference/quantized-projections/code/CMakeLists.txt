cmake_minimum_required(VERSION 3.28)
project(crit_fp8 LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)
add_library(crit_fp8 STATIC fp8_cuda.cu)
target_compile_features(crit_fp8 PUBLIC cxx_std_20 cuda_std_20)
target_include_directories(crit_fp8 PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(crit_fp8 PUBLIC CUDA::cudart)
# std::span accessors in device code are constexpr; they do not allocate or throw.
target_compile_options(crit_fp8 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
