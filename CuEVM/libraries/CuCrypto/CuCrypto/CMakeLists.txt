cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_VERBOSE_MAKEFILE ON)

project("CuCrypto"
    VERSION 0.0.1
    DESCRIPTION "CuCrypto: CUDA Crypto Static Library"
    LANGUAGES CUDA CXX
)


# where to find our CMake modules
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
#message(STATUS "CMake module path: ${CMAKE_MODULE_PATH}")

if (NOT CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    message(STATUS "This project has a top-level one called [${CMAKE_PROJECT_NAME}]")
else()
    message(STATUS "This project is a top-level one")
endif()

enable_language(CUDA)


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CUDA_COMPUTE_CAPABILITY "50" CACHE STRING "CUDA Compute Capability")
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})


# Add CuCrypto source files
# set(CUCRYPTO_SOURCES
#     src/keccak.cu
#     src/sha256.cu
#     src/ripemd160.cu
#     src/blake2.cu
# )
# Automatically find all .cu source files in the src directory
file(GLOB CUCRYPTO_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu")

# target_sources(${PROJECT_NAME}
#     PRIVATE
#     ${CUCRYPTO_SOURCES}
# )
add_library(${PROJECT_NAME} OBJECT ${CUCRYPTO_SOURCES})


# set target properties
set_target_properties(${PROJECT_NAME}
    PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        POSITION_INDEPENDENT_CODE ON
)

target_include_directories(${PROJECT_NAME}
    PUBLIC
        # where top-level project will look for the library's public headers
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        # where the library will look for its private headers
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)


set(MYCUDALIB ${CMAKE_CURRENT_BINARY_DIR}/CuCrypto.lib PARENT_SCOPE)