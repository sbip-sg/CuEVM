cmake_minimum_required(VERSION 3.18 FATAL_ERROR) # 3.24 for selecting native CUDA_ARCHITECTURES
option(WITH_COMPILE "Doing the nvcc compilation" ON)
option(WITH_DOCS "Generating the documentation for the code" OFF)
option(ONLY_CPU "Generating the CPU version of cuEVM " OFF)
option(ENABLE_TRACING "Enable tracer" ON)
option(BUILD_LIBRARY "Build as a shared library" OFF) # Added option for building as a library
option(ENABLE_COMPLEX_TRACER "Enable complex tracer" OFF)
set(CUDA_COMPUTE_CAPABILITY "89" CACHE STRING "CUDA Compute Capability")

project(cuevm
        VERSION 0.1.0
        DESCRIPTION "A CUDA EVM implementation"
        LANGUAGES CUDA CXX)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})


if(WITH_DOCS)
        add_subdirectory("${PROJECT_SOURCE_DIR}/docs")
endif()

if(WITH_COMPILE)
        add_subdirectory("${PROJECT_SOURCE_DIR}/src")
endif()

if(ENABLE_TRACING)
    target_compile_definitions(${PROJECT_NAME} PRIVATE TRACER)
endif()

if(ENABLE_COMPLEX_TRACER)
    target_compile_definitions(${PROJECT_NAME} PRIVATE COMPLEX_TRACER)
endif()

if(ONLY_CPU)
    target_compile_definitions(${PROJECT_NAME} PRIVATE ONLY_CPU)
endif()
