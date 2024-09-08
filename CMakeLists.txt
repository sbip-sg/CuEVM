cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

option(WITH_COMPILE "Doing the nvcc compilation" ON)
option(WITH_DOCS "Generating the documentation for the code" OFF)
option(ONLY_CPU "Generating the CPU version of cuEVM " OFF)
option(ENABLE_TRACING "Enable tracer" ON)
option(ENABLE_OPTIONAL_TRACER "Enable complex tracer" OFF)

project("cuevm_interpreter"
        VERSION 0.1.0
        DESCRIPTION "A CUDA EVM Interpreter"
        LANGUAGES CUDA CXX)

enable_language(CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CUDA_COMPUTE_CAPABILITY "50" CACHE STRING "CUDA Compute Capability")
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

find_package(cJSON REQUIRED) # Assuming cJSON is available as a find package
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

if(WITH_DOCS)
        add_subdirectory("${PROJECT_SOURCE_DIR}/docs")
endif()

if(WITH_COMPILE)
    add_subdirectory("${PROJECT_SOURCE_DIR}/CuEVM")
    add_executable(${PROJECT_NAME}
        "${PROJECT_SOURCE_DIR}/interpreter/interpreter.cu"
        $<TARGET_OBJECTS:CuEVM>
        $<TARGET_OBJECTS:CuCrypto>
        $<TARGET_OBJECTS:CGBN>
        $<TARGET_OBJECTS:CuBigInt>
    )
    
    set_target_properties(${PROJECT_NAME}
        PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
            POSITION_INDEPENDENT_CODE ON
    )
    
    # Link the library to its dependencies
    # our own libraries first
    target_link_libraries(${PROJECT_NAME} PRIVATE CGBN CuCrypto CuBigInt CuEVM)
    # then the external ones
    target_link_libraries(${PROJECT_NAME} PUBLIC gmp cjson ${CUDA_LIBRARIES})
    
    target_compile_options(${PROJECT_NAME}  PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --std=c++20 -rdc=true --expt-relaxed-constexpr>)
    
    
    # if(ENABLE_TRACING)
    #     target_compile_definitions(${PROJECT_NAME} PRIVATE TRACER)
    # endif()
    
    # if(ENABLE_COMPLEX_TRACER)
    #     target_compile_definitions(${PROJECT_NAME} PRIVATE COMPLEX_TRACER)
    # endif()
    
    if(ONLY_CPU)
        target_compile_definitions(${PROJECT_NAME} PRIVATE ONLY_CPU)
    endif()    
endif()
