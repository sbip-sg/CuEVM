cmake_minimum_required(VERSION 3.22)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_VERBOSE_MAKEFILE ON)

project("CGBNTests"
    VERSION 0.0.1
    DESCRIPTION "CGBNTests: Test suite for CGBN library"
    LANGUAGES CUDA CXX
)

enable_language(CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CUDA_COMPUTE_CAPABILITY "50" CACHE STRING "CUDA Compute Capability")
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})

find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
# get the library
add_subdirectory(CGBN)


find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

file(GLOB TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cu")
# Add test files
# set(TEST_SOURCES
#     tests/test_main.cu
#     tests/test_keccak.cu
#     tests/test_sha256.cu
#     tests/test_ripemd160.cu
#     tests/test_blake2.cu
# )
# Create a single executable for all test files
add_executable(${PROJECT_NAME} ${TEST_SOURCES} $<TARGET_OBJECTS:CGBN>)
set_target_properties(${PROJECT_NAME}
   PROPERTIES
       CUDA_SEPARABLE_COMPILATION ON
       CUDA_RESOLVE_DEVICE_SYMBOLS ON
       POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(${PROJECT_NAME} PRIVATE CGBN)
# Include CuCrypto headers
#
# Include CuCrypto headers
# target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/CuCrypto/include)

target_link_libraries(${PROJECT_NAME} PRIVATE ${CUDART_LIBRARY})
target_link_libraries(${PROJECT_NAME} PUBLIC ${GTEST_LIBRARIES} ${CUDA_LIBRARIES})


# Add the test executable to CTest
enable_testing()
add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})

