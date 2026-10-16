cmake_minimum_required(VERSION 3.28)
project(crit_core_projection LANGUAGES CXX)

# Build Core's public SDK from this checkout; no installed SDK is required.
get_filename_component(CRIT_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
add_subdirectory("${CRIT_ROOT}/core" "${CMAKE_CURRENT_BINARY_DIR}/core")

add_executable(crit_core_projection main.cpp)
target_link_libraries(crit_core_projection PRIVATE crit::core)
target_include_directories(crit_core_projection PRIVATE
  "${CRIT_ROOT}/inference/quantized-projections/code")
if(MSVC)
  target_compile_options(crit_core_projection PRIVATE /W4 /WX /permissive-)
else()
  target_compile_options(crit_core_projection PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

enable_testing()
add_test(NAME core_projection COMMAND crit_core_projection)
set_tests_properties(core_projection PROPERTIES TIMEOUT 30)
