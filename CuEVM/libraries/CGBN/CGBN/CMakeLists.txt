cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_VERBOSE_MAKEFILE ON)

project("CGBN"
    VERSION 0.0.1
    DESCRIPTION "CGBN: CUDA Generic Bignum Library"
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
find_package(CUDAToolkit REQUIRED)
find_package(GMP REQUIRED)
include_directories(${GMP_INCLUDE_DIR})


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CUDA_COMPUTE_CAPABILITY "50" CACHE STRING "CUDA Compute Capability")
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})


file(GLOB CGBN_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu")

add_library(${PROJECT_NAME} OBJECT ${CGBN_SOURCES})

# set target properties
set_target_properties(${PROJECT_NAME}
    PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(${PROJECT_NAME} gmp)
# target_sources(${PROJECT_NAME}
#     PRIVATE
#         src/utils.cu
#         src/cgbn_mpz.cu
#         src/cgbn_cuda.cu
# )

target_include_directories(${PROJECT_NAME}
    PUBLIC
        # where top-level project will look for the library's public headers
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        # where the library will look for its private headers
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

