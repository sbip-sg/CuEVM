cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_VERBOSE_MAKEFILE ON)

project("CuEVM"
    VERSION 0.1.0
    DESCRIPTION "A CUDA EVM implementation"
    LANGUAGES CUDA CXX
)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
if (NOT CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    message(STATUS "This project has a top-level one called [${CMAKE_PROJECT_NAME}]")
else()
    message(STATUS "This project is a top-level one")
endif()

enable_language(CUDA)


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
if (NOT CUDA_COMPUTE_CAPABILITY)
    set(CUDA_COMPUTE_CAPABILITY "50" CACHE STRING "CUDA Compute Capability")
endif()
if (NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
endif()
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})


find_package(GMP REQUIRED)
find_package(cJSON REQUIRED) # Assuming cJSON is available as a find package
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(${GMP_INCLUDE_DIR})

add_subdirectory(libraries)

# Automatically find all .cu source files in the src directory
file(GLOB_RECURSE CUEVM_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu")

add_library(${PROJECT_NAME} OBJECT ${CUEVM_SOURCES})
#  $<TARGET_OBJECTS:CuCrypto> $<TARGET_OBJECTS:CGBN> $<TARGET_OBJECTS:CuBigInt>


set_target_properties(${PROJECT_NAME}
    PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        POSITION_INDEPENDENT_CODE ON
)
# set_target_properties(${PROJECT_NAME}  PROPERTIES CUDA_RUNTIME_LIBRARY Shared)

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

# Link the library to its dependencies
# our own libraries first
target_link_libraries(${PROJECT_NAME} PRIVATE CGBN CuCrypto CuBigInt)
# then the external ones
target_link_libraries(${PROJECT_NAME} PUBLIC gmp cjson ${CUDA_LIBRARIES})


# Add specific NVCC flags using target_compile_options (if necessary)
target_compile_options(${PROJECT_NAME}  PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --std=c++20 -rdc=true --expt-relaxed-constexpr>)


