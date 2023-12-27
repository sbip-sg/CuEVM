cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(cuevm
        VERSION 0.1.0
        DESCRIPTION "A CUDA EVM implementation"
        LANGUAGES CUDA CXX)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

include_directories("${PROJECT_SOURCE_DIR}/CGBN/include")

add_subdirectory("${PROJECT_SOURCE_DIR}/docs")
add_subdirectory("${PROJECT_SOURCE_DIR}/src")