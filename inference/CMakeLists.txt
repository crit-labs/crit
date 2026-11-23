cmake_minimum_required(VERSION 3.28)
project(crit_inference VERSION 0.23.0 LANGUAGES CXX)
# The fixtures of Inference's native code (what plugins build at load) and of its host references. Inference uses Core only through its public SDK
# (crit::core); Core never builds or includes Inference code.
find_package(Threads REQUIRED)
include(CTest)
option(CRIT_INFERENCE_CUDA "Build Inference's CUDA reference kernels and their fixtures" OFF)
option(CRIT_INFERENCE_INSTALLED_CORE "Use an installed Core SDK (CMAKE_PREFIX_PATH) instead of this checkout's core/" OFF)
if(NOT TARGET crit::core)
  if(CRIT_INFERENCE_INSTALLED_CORE)
    find_package(critCore 0.23.0 CONFIG REQUIRED)
  else()
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/core" "${CMAKE_CURRENT_BINARY_DIR}/core")
  endif()
endif()

set(INFERENCE "${CMAKE_CURRENT_SOURCE_DIR}")
set(FIXTURES "${CMAKE_CURRENT_SOURCE_DIR}/../tests/inference")

function(crit_inference_warnings target)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4 /WX /permissive-)
  else()
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endif()
endfunction()

# crit_inference_executable(target SOURCES ... [INCLUDES ...] [LINK ...])
function(crit_inference_executable target)
  cmake_parse_arguments(ARG "" "" "SOURCES;INCLUDES;LINK" ${ARGN})
  add_executable(${target} ${ARG_SOURCES})
  target_compile_features(${target} PRIVATE cxx_std_20)
  if(ARG_INCLUDES)
    target_include_directories(${target} PRIVATE ${ARG_INCLUDES})
  endif()
  if(ARG_LINK)
    target_link_libraries(${target} PRIVATE ${ARG_LINK})
  endif()
  crit_inference_warnings(${target})
endfunction()

# crit_inference_tests(target TIMEOUT seconds CASES case...): one CTest per named case.
function(crit_inference_tests target prefix timeout)
  foreach(_case ${ARGN})
    add_test(NAME ${prefix}_${_case} COMMAND ${target} ${_case})
    set_tests_properties(${prefix}_${_case} PROPERTIES TIMEOUT ${timeout})
  endforeach()
endfunction()

function(crit_inference_test name timeout)
  add_test(NAME ${name} COMMAND ${ARGN})
  set_tests_properties(${name} PROPERTIES TIMEOUT ${timeout})
endfunction()

if(NOT BUILD_TESTING)
  return()
endif()

# Host examples and fixtures.
crit_inference_executable(crit_quantized_tests SOURCES ${FIXTURES}/quantized_tests.cpp
  INCLUDES ${INFERENCE}/techniques/projections/quantized-projections/code)
crit_inference_tests(crit_quantized_tests crit_quantized 45 fp8_block)

crit_inference_executable(crit_nvfp4_tests SOURCES ${FIXTURES}/nvfp4_tests.cpp INCLUDES ${INFERENCE}/techniques/projections/quantized-projections/code)
crit_inference_test(crit_nvfp4 30 crit_nvfp4_tests ${FIXTURES}/e4m3_scale_vectors.txt)
# Host ISA files: AVX2 and AVX-512 sources compiled with their own flags and chosen at run time through host_isa.h.
# One condition builds them for every library, so host_isa's supported() answers for all of them.
set(_host_nvfp4 ${INFERENCE}/techniques/experts/host-expert-arithmetic/code)
set(CRIT_INFERENCE_HOST_ISA OFF)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(AMD64|amd64|x86_64|X86_64)$")
  set(CRIT_INFERENCE_HOST_ISA ON)
endif()
add_library(crit_host_isa STATIC ${_host_nvfp4}/host_isa.cpp)
target_include_directories(crit_host_isa PUBLIC ${_host_nvfp4})
target_compile_features(crit_host_isa PUBLIC cxx_std_20)
crit_inference_warnings(crit_host_isa)
if(CRIT_INFERENCE_HOST_ISA)
  target_compile_definitions(crit_host_isa PRIVATE CRIT_HOST_ISA_FILES=1)
endif()
# crit_inference_isa_sources(target avx2_source avx512_source): links host_isa and, where the ISA files are built,
# adds them with their flags and defines CRIT_HOST_ISA_FILES for the target's dispatch.
function(crit_inference_isa_sources target avx2 avx512)
  target_link_libraries(${target} PUBLIC crit_host_isa)
  if(NOT CRIT_INFERENCE_HOST_ISA)
    return()
  endif()
  target_sources(${target} PRIVATE ${avx2} ${avx512})
  target_compile_definitions(${target} PRIVATE CRIT_HOST_ISA_FILES=1)
  if(MSVC)
    set_source_files_properties(${avx2} PROPERTIES COMPILE_OPTIONS /arch:AVX2)
    set_source_files_properties(${avx512} PROPERTIES COMPILE_OPTIONS /arch:AVX512)
  else()
    set_source_files_properties(${avx2} PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
    set_source_files_properties(${avx512} PROPERTIES COMPILE_OPTIONS "-mavx512f;-mfma")
  endif()
endfunction()

# Host NVFP4 rows in a declared order: no file may contract a multiply and an add into an FMA the order does not
# declare.
add_library(crit_host_nvfp4 STATIC ${_host_nvfp4}/host_nvfp4.cpp ${_host_nvfp4}/host_int.cpp ${_host_nvfp4}/host_capi.cpp)
target_include_directories(crit_host_nvfp4 PUBLIC ${_host_nvfp4} ${INFERENCE}/techniques/projections/quantized-projections/code)
target_compile_features(crit_host_nvfp4 PUBLIC cxx_std_20)
crit_inference_warnings(crit_host_nvfp4)
if(MSVC)
  target_compile_options(crit_host_nvfp4 PRIVATE /fp:precise)
else()
  target_compile_options(crit_host_nvfp4 PRIVATE -ffp-contract=off)
endif()
crit_inference_isa_sources(crit_host_nvfp4 ${_host_nvfp4}/host_nvfp4_avx2.cpp ${_host_nvfp4}/host_nvfp4_avx512.cpp)
# The integer rows (host_int.h): AVX2 and AVX-512 with BW and VNNI, beside the NVFP4 files.
if(CRIT_INFERENCE_HOST_ISA)
  target_sources(crit_host_nvfp4 PRIVATE ${_host_nvfp4}/host_int_avx2.cpp ${_host_nvfp4}/host_int_avx512.cpp)
  if(MSVC)
    set_source_files_properties(${_host_nvfp4}/host_int_avx2.cpp PROPERTIES COMPILE_OPTIONS /arch:AVX2)
    set_source_files_properties(${_host_nvfp4}/host_int_avx512.cpp PROPERTIES COMPILE_OPTIONS /arch:AVX512)
  else()
    set_source_files_properties(${_host_nvfp4}/host_int_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
    set_source_files_properties(${_host_nvfp4}/host_int_avx512.cpp PROPERTIES
                                COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vnni;-mfma")
  endif()
endif()
crit_inference_executable(crit_host_nvfp4_tests SOURCES ${FIXTURES}/host_nvfp4_tests.cpp LINK crit_host_nvfp4 Threads::Threads)
crit_inference_test(crit_host_nvfp4 60 crit_host_nvfp4_tests)
crit_inference_executable(crit_host_int_tests SOURCES ${FIXTURES}/host_int_tests.cpp LINK crit_host_nvfp4 Threads::Threads)
crit_inference_test(crit_host_int 60 crit_host_int_tests)
# The host's worker pool, which is Core's (crit_plugins.host builds it on the machine at first use): its source is
# compiled here for the fixtures that run on it.
set(_core_host ${CMAKE_CURRENT_SOURCE_DIR}/core/python/crit_plugins/host)
add_library(crit_host_pool STATIC ${_core_host}/pool.cpp)
target_include_directories(crit_host_pool PUBLIC ${_core_host})
target_link_libraries(crit_host_pool PUBLIC Threads::Threads)
target_compile_features(crit_host_pool PUBLIC cxx_std_20)
crit_inference_warnings(crit_host_pool)
# A CPU expert tier over the integer rows: one layer's CPU routes on the host's pool.
set(_host_tier ${INFERENCE}/techniques/experts/host-expert-tier/code)
add_library(crit_host_tier STATIC ${_host_tier}/host_tier.cpp ${_host_tier}/tier_capi.cpp)
target_include_directories(crit_host_tier PUBLIC ${_host_tier})
target_link_libraries(crit_host_tier PUBLIC crit_host_nvfp4 crit_host_pool Threads::Threads)
target_compile_features(crit_host_tier PUBLIC cxx_std_20)
crit_inference_warnings(crit_host_tier)
if(MSVC)
  target_compile_options(crit_host_tier PRIVATE /fp:precise)
else()
  target_compile_options(crit_host_tier PRIVATE -ffp-contract=off)
endif()
crit_inference_executable(crit_host_expert_tier_tests SOURCES ${FIXTURES}/host_expert_tier_tests.cpp LINK crit_host_tier)
crit_inference_test(crit_host_expert_tier 120 crit_host_expert_tier_tests)


if(NOT CRIT_INFERENCE_CUDA)
  return()
endif()

# CUDA reference kernels (CUDA C++20). Strict warnings apply to the C++ sources;
# nvcc compiles the .cu reference kernels separately.
find_package(CUDAToolkit 13.1 REQUIRED)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()
enable_language(CUDA)

# crit_inference_kernels(target code_dir source...): a static CUDA library exporting its code folder.
function(crit_inference_kernels target code source)
  set(_sources ${code}/${source})
  foreach(_more IN LISTS ARGN)
    list(APPEND _sources ${code}/${_more})
  endforeach()
  add_library(${target} STATIC ${_sources})
  target_include_directories(${target} PUBLIC ${code})
  target_compile_features(${target} PUBLIC cxx_std_20 cuda_std_20)
  target_link_libraries(${target} PUBLIC CUDA::cudart)
endfunction()

# token-sampling's law is crit's kernel token-sampling (kernels/techniques/sampling/token-sampling): its fixtures build
# its sampling.cu with the kernel core's headers (crit_kcore/radix.cuh) and the default knobs (no crit_tuning.h).
crit_inference_kernels(crit_token_sampling ${CMAKE_CURRENT_SOURCE_DIR}/../kernels/techniques/sampling/token-sampling/sampling_cuda sampling.cu)
target_include_directories(crit_token_sampling PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../kernels/core/include)
# The fixtures' host oracle of Core's sampling rules (tests/inference/host_sampler.h).
add_library(crit_host_sampler STATIC ${FIXTURES}/host_sampler.cpp)
target_include_directories(crit_host_sampler PUBLIC ${FIXTURES})
target_link_libraries(crit_host_sampler PUBLIC crit::core)
crit_inference_warnings(crit_host_sampler)
target_link_libraries(crit_token_sampling PUBLIC crit_host_sampler)
crit_inference_executable(crit_token_sampling_tests SOURCES ${FIXTURES}/token_sampling_tests.cpp LINK crit_token_sampling)
crit_inference_test(crit_token_sampling 120 crit_token_sampling_tests)
crit_inference_executable(crit_token_law_tests SOURCES ${FIXTURES}/token_law_tests.cpp LINK crit_token_sampling)
crit_inference_test(crit_token_law 300 crit_token_law_tests)

# The host's handoffs, Core's (crit_plugins.host, inference/core/python/crit_plugins/host/README.md): its CUDA fixture.
crit_inference_kernels(crit_host_mailbox ${_core_host} handoff.cu)
add_library(crit_host_mailbox_fixture STATIC ${FIXTURES}/host_mailbox_fixture.cu)
target_link_libraries(crit_host_mailbox_fixture PUBLIC crit_host_mailbox)
crit_inference_executable(crit_host_mailbox_tests SOURCES ${FIXTURES}/host_mailbox_tests.cpp LINK crit_host_mailbox_fixture)
crit_inference_test(crit_host_mailbox 300 crit_host_mailbox_tests)
