cmake_minimum_required(VERSION 3.28)
project(crit_inference VERSION 0.6.1 LANGUAGES CXX)
# Inference's native examples and fixtures. Inference uses Core only through its public SDK
# (crit::core, crit::cuda); Core never builds or includes Inference code.
find_package(Threads REQUIRED)
include(CTest)
option(CRIT_INFERENCE_CUDA "Build Inference's CUDA reference kernels and their fixtures; needs Core's CUDA backend" OFF)
option(CRIT_INFERENCE_INSTALLED_CORE "Use an installed Core SDK (CMAKE_PREFIX_PATH) instead of this checkout's core/" OFF)
if(NOT TARGET crit::core)
  if(CRIT_INFERENCE_INSTALLED_CORE)
    find_package(critCore 0.6.1 CONFIG REQUIRED)
  else()
    if(CRIT_INFERENCE_CUDA)
      set(CRIT_CORE_CUDA ON CACHE BOOL "Build the explicit CUDA backend services" FORCE)
    endif()
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../core" "${CMAKE_CURRENT_BINARY_DIR}/core")
  endif()
endif()
if(CRIT_INFERENCE_CUDA AND NOT TARGET crit::cuda)
  message(FATAL_ERROR "CRIT_INFERENCE_CUDA needs Core's CUDA backend (crit::cuda); build or install Core with CRIT_CORE_CUDA=ON")
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
crit_inference_executable(crit_recurrent_recovery_example SOURCES ${INFERENCE}/recurrent-state-recovery/code/example.cpp LINK crit::core)
crit_inference_test(crit_recurrent_recovery_example 30 crit_recurrent_recovery_example)
crit_inference_executable(crit_recurrent_recovery_tests SOURCES ${FIXTURES}/recurrent_recovery_tests.cpp
  INCLUDES ${INFERENCE}/recurrent-state-recovery/code LINK crit::core)
crit_inference_test(crit_recurrent_recovery 30 crit_recurrent_recovery_tests)

crit_inference_executable(crit_grouped_example SOURCES ${INFERENCE}/quantized-projections/code/grouped_example.cpp
  INCLUDES ${INFERENCE}/quantized-projections/code ${INFERENCE}/attention-state/code LINK Threads::Threads)
add_test(NAME crit_grouped_example COMMAND crit_grouped_example)
crit_inference_executable(crit_encoded_example SOURCES ${INFERENCE}/attention-state/code/encoded_example.cpp
  INCLUDES ${INFERENCE}/quantized-projections/code ${INFERENCE}/attention-state/code LINK crit::core)
add_test(NAME crit_encoded_example COMMAND crit_encoded_example)
crit_inference_executable(crit_quantized_tests SOURCES ${FIXTURES}/quantized_tests.cpp
  INCLUDES ${INFERENCE}/quantized-projections/code ${INFERENCE}/attention-state/code LINK crit::core)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(AMD64|amd64|x86_64|X86_64)$")
  target_compile_definitions(crit_quantized_tests PRIVATE CRIT_GROUPED_INT_AVX2=1)
  if(MSVC)
    target_compile_options(crit_quantized_tests PRIVATE /arch:AVX2)
  else()
    target_compile_options(crit_quantized_tests PRIVATE -mavx2)
  endif()
endif()
crit_inference_tests(crit_quantized_tests crit_quantized 45 arithmetic failures kv)

crit_inference_executable(crit_sampled_example SOURCES ${INFERENCE}/mtp-decode/code/sampled_example.cpp)
add_test(NAME crit_sampled_example COMMAND crit_sampled_example)
crit_inference_executable(crit_sampled_tests SOURCES ${FIXTURES}/sampled_tests.cpp INCLUDES ${INFERENCE}/mtp-decode/code)
crit_inference_tests(crit_sampled_tests crit_sampled 60 arithmetic statistical transactions adaptive)

crit_inference_executable(crit_constraint_example SOURCES ${INFERENCE}/token-constraints/code/example.cpp LINK crit::core)
add_test(NAME crit_constraint_example COMMAND crit_constraint_example)
crit_inference_executable(crit_constraint_tests SOURCES ${FIXTURES}/constraint_tests.cpp INCLUDES ${INFERENCE}/token-constraints/code
  LINK crit::core)
crit_inference_tests(crit_constraint_tests crit_constraint 60 arithmetic transactions sampled json_sampled grammar)

crit_inference_executable(crit_inference_io_codec_tests SOURCES ${FIXTURES}/inference_io_codec_tests.cpp
  INCLUDES ${INFERENCE}/quantized-projections/code ${INFERENCE}/ssd-expert-materialization/code LINK crit::core Threads::Threads)
crit_inference_test(crit_inference_io_codec 30 crit_inference_io_codec_tests "${CMAKE_CURRENT_BINARY_DIR}/gather-fixture")

crit_inference_executable(crit_nvfp4_tests SOURCES ${FIXTURES}/nvfp4_tests.cpp INCLUDES ${INFERENCE}/quantized-projections/code)
crit_inference_test(crit_nvfp4 30 crit_nvfp4_tests)
# Host ISA files: AVX2 and AVX-512 sources compiled with their own flags and chosen at run time through host_isa.h.
# One condition builds them for every library, so host_isa's supported() answers for all of them.
set(_host_nvfp4 ${INFERENCE}/host-expert-arithmetic/code)
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
add_library(crit_host_nvfp4 STATIC ${_host_nvfp4}/host_nvfp4.cpp)
target_include_directories(crit_host_nvfp4 PUBLIC ${_host_nvfp4} ${INFERENCE}/quantized-projections/code)
target_compile_features(crit_host_nvfp4 PUBLIC cxx_std_20)
crit_inference_warnings(crit_host_nvfp4)
if(MSVC)
  target_compile_options(crit_host_nvfp4 PRIVATE /fp:precise)
else()
  target_compile_options(crit_host_nvfp4 PRIVATE -ffp-contract=off)
endif()
crit_inference_isa_sources(crit_host_nvfp4 ${_host_nvfp4}/host_nvfp4_avx2.cpp ${_host_nvfp4}/host_nvfp4_avx512.cpp)
crit_inference_executable(crit_host_nvfp4_tests SOURCES ${FIXTURES}/host_nvfp4_tests.cpp LINK crit_host_nvfp4 Threads::Threads)
crit_inference_test(crit_host_nvfp4 60 crit_host_nvfp4_tests)

crit_inference_executable(crit_selection_tests SOURCES ${FIXTURES}/selection_tests.cpp INCLUDES ${INFERENCE}/recorded-selection/code)
crit_inference_test(crit_selection 30 crit_selection_tests)

crit_inference_executable(crit_speculative_example SOURCES ${INFERENCE}/mtp-decode/code/speculative_example.cpp)
crit_inference_test(crit_speculative_example 30 crit_speculative_example)
crit_inference_executable(crit_speculative_tests SOURCES ${FIXTURES}/speculative_tests.cpp
  INCLUDES ${INFERENCE}/mtp-decode/code ${INFERENCE}/attention-state/code LINK crit::core)
crit_inference_tests(crit_speculative_tests crit_speculative 45 transactions failures copy attention core)

crit_inference_executable(crit_attention_example SOURCES ${INFERENCE}/attention-state/code/attention_example.cpp LINK crit::core)
crit_inference_test(crit_attention_example 30 crit_attention_example)
crit_inference_executable(crit_paged_batch_example SOURCES ${INFERENCE}/attention-state/code/batch_example.cpp LINK crit::core)
crit_inference_test(crit_paged_batch_example 30 crit_paged_batch_example)
crit_inference_executable(crit_recovery_scheduler_tests SOURCES ${FIXTURES}/recovery_scheduler_tests.cpp
  INCLUDES ${INFERENCE}/attention-state/code LINK crit::core)
crit_inference_test(crit_recovery_scheduler 30 crit_recovery_scheduler_tests)
crit_inference_tests(crit_recovery_scheduler_tests crit_recovery_scheduler 30
  cancel_suspended cancel_restore suspend_failure restore_failure quarantine)
crit_inference_executable(crit_dependent_steps_tests SOURCES ${FIXTURES}/dependent_steps_tests.cpp
  INCLUDES ${INFERENCE}/concurrent-request-execution/code LINK crit::core)
crit_inference_tests(crit_dependent_steps_tests crit_dependent_steps 30 compare ahead cancel recovery stale commit quarantine immovable)
add_library(crit_native_scheduler_fixture SHARED ${FIXTURES}/native_scheduler_fixture.cpp)
target_include_directories(crit_native_scheduler_fixture PRIVATE ${INFERENCE}/concurrent-request-execution/code ${INFERENCE}/attention-state/code)
target_link_libraries(crit_native_scheduler_fixture PRIVATE crit::core)
crit_inference_warnings(crit_native_scheduler_fixture)
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
  add_test(NAME crit_native_scheduler_serving COMMAND "${Python3_EXECUTABLE}" -B
    "${FIXTURES}/native_serving_check.py" $<TARGET_FILE:crit_native_scheduler_fixture>)
  set_tests_properties(crit_native_scheduler_serving PROPERTIES TIMEOUT 90)
endif()
crit_inference_executable(crit_attention_tests SOURCES ${FIXTURES}/attention_tests.cpp
  INCLUDES ${INFERENCE}/attention-state/code ${CMAKE_CURRENT_SOURCE_DIR}/../kernels/examples/openblas
  LINK crit::core ${CMAKE_DL_LIBS})
crit_inference_tests(crit_attention_tests crit_attention 45 batches arithmetic rotary integration failures)

crit_inference_executable(crit_state_pages_tests SOURCES ${FIXTURES}/state_pages_tests.cpp
  INCLUDES ${INFERENCE}/attention-state/code LINK crit::core)
crit_inference_tests(crit_state_pages_tests crit_state_pages 30 transactions prefixes numerics failures randomized)
crit_inference_executable(crit_paged_kv_example SOURCES ${INFERENCE}/attention-state/code/paged_example.cpp LINK crit::core)
crit_inference_test(crit_paged_kv_example 30 crit_paged_kv_example)

crit_inference_executable(crit_settled_slots_tests SOURCES ${FIXTURES}/settled_slots_tests.cpp INCLUDES ${INFERENCE}/expert-residency/code)
crit_inference_test(crit_settled_slots 30 crit_settled_slots_tests)
crit_inference_executable(crit_cache_loan_example SOURCES ${INFERENCE}/expert-residency/code/loan_example.cpp LINK crit::core)
crit_inference_test(crit_cache_loan_example 30 crit_cache_loan_example)

crit_inference_executable(crit_scheduling_tests SOURCES ${FIXTURES}/scheduling_tests.cpp INCLUDES ${INFERENCE}/session-scheduling/code)
add_test(NAME crit_scheduling_policy COMMAND crit_scheduling_tests)
crit_inference_executable(crit_physical_batches_tests SOURCES ${FIXTURES}/physical_batches_tests.cpp
  INCLUDES ${INFERENCE}/session-scheduling/code LINK Threads::Threads)
crit_inference_test(crit_physical_batches 30 crit_physical_batches_tests)
crit_inference_executable(crit_physical_speculative_example SOURCES ${INFERENCE}/mtp-decode/code/physical_batches_example.cpp)
crit_inference_test(crit_physical_speculative_example 30 crit_physical_speculative_example)
crit_inference_executable(crit_pipeline_tests SOURCES ${FIXTURES}/pipeline_tests.cpp
  INCLUDES ${INFERENCE}/concurrent-request-execution/code LINK crit::core)
crit_inference_test(crit_inference_pipeline 30 crit_pipeline_tests)
crit_inference_executable(crit_prompt_copy_tests SOURCES ${FIXTURES}/prompt_copy_tests.cpp INCLUDES ${INFERENCE}/mtp-decode/code)
add_test(NAME crit_prompt_copy COMMAND crit_prompt_copy_tests)

crit_inference_executable(crit_core_projection_example SOURCES ${INFERENCE}/examples/core-projection/main.cpp
  INCLUDES ${INFERENCE}/quantized-projections/code LINK crit::core)
crit_inference_test(crit_core_projection_example 30 crit_core_projection_example)

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
add_executable(crit_cuda_ahead_example ${INFERENCE}/concurrent-request-execution/code/cuda_ahead_example.cu)
target_compile_features(crit_cuda_ahead_example PRIVATE cxx_std_20 cuda_std_20)
target_link_libraries(crit_cuda_ahead_example PRIVATE crit::cuda)
crit_inference_test(crit_cuda_ahead 60 crit_cuda_ahead_example)
include(CheckLinkerFlag)
check_linker_flag(CXX "LINKER:--wrap=crit_link_probe" CRIT_INFERENCE_HAS_LINKER_WRAP)

# crit_inference_kernels(target code_dir source): a static CUDA library exporting its code folder.
function(crit_inference_kernels target code source)
  add_library(${target} STATIC ${code}/${source})
  target_include_directories(${target} PUBLIC ${code})
  target_compile_features(${target} PUBLIC cxx_std_20 cuda_std_20)
  target_link_libraries(${target} PUBLIC CUDA::cudart)
endfunction()

set(_round_code ${INFERENCE}/device-speculative-round/code)
crit_inference_kernels(crit_device_round_kernels ${_round_code} greedy_accept.cu)
add_library(crit_device_round STATIC ${_round_code}/device_round.cpp)
target_link_libraries(crit_device_round PUBLIC crit::cuda crit_device_round_kernels)
crit_inference_warnings(crit_device_round)
add_library(crit_device_round_fixture STATIC ${FIXTURES}/device_round_fixture.cu)
target_link_libraries(crit_device_round_fixture PUBLIC crit_device_round_kernels)
crit_inference_executable(crit_device_round_tests SOURCES ${FIXTURES}/device_round_tests.cpp
  INCLUDES ${INFERENCE}/mtp-decode/code ${FIXTURES} LINK crit_device_round crit_device_round_fixture)
crit_inference_tests(crit_device_round_tests crit_device_round 60 accept graphs readback budget)

set(_recurrent_code ${INFERENCE}/recurrent-state-recovery/code)
crit_inference_kernels(crit_recurrent_cuda ${_recurrent_code} recurrent_cuda.cu)
crit_inference_executable(crit_recurrent_cuda_tests SOURCES ${FIXTURES}/recurrent_cuda_tests.cpp LINK crit::cuda crit_recurrent_cuda)
crit_inference_tests(crit_recurrent_cuda_tests crit_recurrent_cuda 120 oracle deferred rejects)

crit_inference_kernels(crit_draft_subset ${INFERENCE}/draft-vocabulary-subset/code subset_argmax.cu)
crit_inference_executable(crit_draft_subset_tests SOURCES ${FIXTURES}/draft_subset_tests.cpp LINK crit::cuda crit_draft_subset)
crit_inference_test(crit_draft_subset 60 crit_draft_subset_tests)

crit_inference_kernels(crit_rotated_kv ${INFERENCE}/rotated-kv-quantization/code rotated_kv.cu)
crit_inference_executable(crit_rotated_kv_tests SOURCES ${FIXTURES}/rotated_kv_tests.cpp LINK crit::cuda crit_rotated_kv)
crit_inference_test(crit_rotated_kv 60 crit_rotated_kv_tests)

crit_inference_kernels(crit_device_expert_cache ${INFERENCE}/device-expert-cache/code device_cache.cu)
crit_inference_executable(crit_device_expert_cache_tests SOURCES ${FIXTURES}/device_expert_cache_tests.cpp LINK crit_device_expert_cache)
crit_inference_test(crit_device_expert_cache 120 crit_device_expert_cache_tests)

crit_inference_kernels(crit_token_sampling ${INFERENCE}/token-sampling/code sampling.cu)
target_link_libraries(crit_token_sampling PUBLIC crit::core)
crit_inference_executable(crit_token_sampling_tests SOURCES ${FIXTURES}/token_sampling_tests.cpp LINK crit_token_sampling)
crit_inference_test(crit_token_sampling 120 crit_token_sampling_tests)

crit_inference_kernels(crit_host_mailbox ${INFERENCE}/host-work-mailbox/code mailbox.cu)
add_library(crit_host_mailbox_fixture STATIC ${FIXTURES}/host_mailbox_fixture.cu)
target_link_libraries(crit_host_mailbox_fixture PUBLIC crit_host_mailbox)
crit_inference_executable(crit_host_mailbox_tests SOURCES ${FIXTURES}/host_mailbox_tests.cpp LINK crit_host_mailbox_fixture)
crit_inference_test(crit_host_mailbox 300 crit_host_mailbox_tests)

crit_inference_kernels(crit_paged_attention ${INFERENCE}/paged-attention/code paged_attention.cu)
target_link_libraries(crit_paged_attention PUBLIC crit::core)
crit_inference_executable(crit_paged_attention_tests SOURCES ${FIXTURES}/paged_attention_tests.cpp
  INCLUDES ${INFERENCE}/attention-state/code LINK crit_paged_attention)
crit_inference_tests(crit_paged_attention_tests crit_paged_attention 300 decode invariance graph rejects)

crit_inference_kernels(crit_batched_engine ${INFERENCE}/batched-reference-engine/code synthetic_model.cu)
target_link_libraries(crit_batched_engine PUBLIC crit_paged_attention crit::cuda)
crit_inference_executable(crit_batched_engine_tests SOURCES ${FIXTURES}/batched_engine_tests.cpp LINK crit_batched_engine)
crit_inference_tests(crit_batched_engine_tests crit_batched_engine 300 kernels batched e4m3)
crit_inference_executable(crit_batched_engine_throughput SOURCES ${INFERENCE}/batched-reference-engine/code/throughput.cpp LINK crit_batched_engine)
add_library(crit_batched_engine_library SHARED ${INFERENCE}/batched-reference-engine/code/engine_library.cpp)
target_link_libraries(crit_batched_engine_library PRIVATE crit_batched_engine)
crit_inference_warnings(crit_batched_engine_library)
if(Python3_Interpreter_FOUND)
  add_test(NAME crit_batched_engine_serving COMMAND "${Python3_EXECUTABLE}" -B
    "${FIXTURES}/batched_engine_serving_check.py" $<TARGET_FILE:crit_batched_engine_library>)
  set_tests_properties(crit_batched_engine_serving PROPERTIES TIMEOUT 300)
endif()

crit_inference_kernels(crit_gguf_projections ${INFERENCE}/gguf-projections/code gguf_cuda.cu)
crit_inference_executable(crit_gguf_projections_tests SOURCES ${FIXTURES}/gguf_projections_tests.cpp LINK crit_gguf_projections crit::core)
crit_inference_test(crit_gguf_projections 120 crit_gguf_projections_tests)

set(_offload ${INFERENCE}/dense-layer-offload/code)
add_library(crit_dense_offload STATIC ${_offload}/host_dense.cpp ${_offload}/offload_exec.cpp ${_offload}/synthetic_offload.cpp)
target_include_directories(crit_dense_offload PUBLIC ${_offload})
target_link_libraries(crit_dense_offload PUBLIC crit_batched_engine Threads::Threads)
target_compile_features(crit_dense_offload PUBLIC cxx_std_20)
crit_inference_warnings(crit_dense_offload)
crit_inference_isa_sources(crit_dense_offload ${_offload}/host_dense_avx2.cpp ${_offload}/host_dense_avx512.cpp)
crit_inference_executable(crit_dense_offload_tests SOURCES ${FIXTURES}/dense_offload_tests.cpp LINK crit_dense_offload)
crit_inference_tests(crit_dense_offload_tests crit_dense_offload 300 plan host plans)
crit_inference_executable(crit_dense_offload_bench SOURCES ${_offload}/offload_bench.cpp LINK crit_dense_offload)

add_executable(crit_host_nvfp4_cuda_tests ${FIXTURES}/host_nvfp4_cuda_tests.cu)
target_compile_features(crit_host_nvfp4_cuda_tests PRIVATE cxx_std_20 cuda_std_20)
target_link_libraries(crit_host_nvfp4_cuda_tests PRIVATE crit_host_nvfp4 CUDA::cudart)
crit_inference_test(crit_host_nvfp4_cuda 120 crit_host_nvfp4_cuda_tests)

add_library(crit_device_round_toy STATIC ${_round_code}/toy_model.cu)
target_link_libraries(crit_device_round_toy PUBLIC crit_device_round_kernels crit_recurrent_cuda)
crit_inference_executable(crit_toy_round_example SOURCES ${_round_code}/toy_example.cpp LINK crit_device_round crit_device_round_toy)
crit_inference_test(crit_toy_round_example 120 crit_toy_round_example)

if(CRIT_INFERENCE_HAS_LINKER_WRAP)
  target_compile_definitions(crit_device_round_tests PRIVATE CRIT_TEST_LINKER_WRAP=1)
  target_link_options(crit_device_round_tests PRIVATE -Wl,--wrap=cudaMemcpyAsync
    -Wl,--wrap=cudaStreamSynchronize -Wl,--wrap=cudaEventSynchronize -Wl,--wrap=cudaDeviceSynchronize)
  target_compile_definitions(crit_toy_round_example PRIVATE CRIT_TOY_COUNT_COPIES=1)
  target_link_options(crit_toy_round_example PRIVATE -Wl,--wrap=cudaMemcpyAsync)
else()
  message(STATUS "Inference CUDA fixtures enabled; linker-injected failure fixtures require --wrap and are omitted.")
endif()
