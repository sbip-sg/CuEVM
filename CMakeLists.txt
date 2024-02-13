cmake_minimum_required(VERSION 3.18 FATAL_ERROR) # 3.24 for selecting native CUDA_ARCHITECTURES 
option(WITH_COMPILE "Doing the nvcc compilation" ON)
option(WITH_DOCS "Geenrating the documentation for the code" OFF)

set(CUDA_COMPUTE_CAPABILITY "89" CACHE STRING "CUDA Compute Capability")
project(cuevm
        VERSION 0.1.0
        DESCRIPTION "A CUDA EVM implementation"
        LANGUAGES CUDA CXX)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

include_directories("${PROJECT_SOURCE_DIR}/CGBN/include")

if(WITH_DOCS)
        add_subdirectory("${PROJECT_SOURCE_DIR}/docs")
endif()
if(WITH_COMPILE)
        add_subdirectory("${PROJECT_SOURCE_DIR}/src")
endif()
if(ENABLE_TRACING)
    target_compile_definitions(${PROJECT_NAME} PRIVATE TRACER)
endif()

if(ONLY_CPU)
    target_compile_definitions(${PROJECT_NAME} PRIVATE ONLY_CPU)
endif()
