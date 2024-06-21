cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

set(namespace "sdcioc")
project("CuCrypto"
    VERSION 0.0.1
    DESCRIPTION "CuCrypto: CUDA Crypto Library"
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

add_library(${PROJECT_NAME} STATIC)

# set target properties
set_target_properties(${PROJECT_NAME}
    PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        POSITION_INDEPENDENT_CODE ON
)

target_sources(${PROJECT_NAME}
    PRIVATE
        src/keccak.cu
        src/sha256.cu
        src/ripemd160.cu
        src/blake2.cu
)

target_include_directories(${PROJECT_NAME}
    PUBLIC
        # where top-level project will look for the library's public headers
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        # where external projects will look for the library's public headers
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        # where the library will look for its private headers
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# totally optional listing of include directories
get_property(inclds
    TARGET ${PROJECT_NAME}
    PROPERTY INCLUDE_DIRECTORIES
)
message(STATUS "Listing include directories for ${PROJECT_NAME}...")
# "dumb" listing with "raw" generator expressions on configuration
foreach(pth ${inclds})
    message(STATUS "  ${pth}")
endforeach()

# without it public headers won't get installed
set(public_headers
    include/CuCrypto/utils.cuh
    include/CuCrypto/keccak.cuh
    include/CuCrypto/sha256.cuh
    include/CuCrypto/ripemd160.cuh
    include/CuCrypto/blake2.cuh
)

include(Installing)
