cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

option(DOCS "Generating the documentation for the code" OFF)
option(ENABLE_EIP_3155 "Enable EIP_3155 tracer" ON)
option(ENABLE_PAIRING_CODE "Enable ec pairing precompile code" OFF)
option(BUILD_LIBRARY "Build Python library" OFF)
set(EVM_VERSION "SHANGHAI" CACHE STRING "EVM version to use")
project("cuevm"
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

if(DOCS)
        add_subdirectory("${PROJECT_SOURCE_DIR}/docs")
endif()

if(BUILD_LIBRARY)
    find_package(PythonLibs REQUIRED)
    include_directories(${PYTHON_INCLUDE_DIRS})
endif()

add_subdirectory("${PROJECT_SOURCE_DIR}/CuEVM")
if(BUILD_LIBRARY)
    target_compile_definitions(CuEVM PRIVATE BUILD_LIBRARY)
endif()
if(ENABLE_EIP_3155)
    target_compile_definitions(CuEVM PRIVATE EIP_3155)
endif()
if(ENABLE_EIP_3155_OPTIONAL)
    target_compile_definitions(CuEVM PRIVATE EIP_3155_OPTIONAL)
endif()
if(ENABLE_PAIRING_CODE)
    target_compile_definitions(CuEVM PRIVATE ENABLE_PAIRING_CODE)
endif()
target_compile_definitions(CuEVM PRIVATE EVM_VERSION=${EVM_VERSION})


if(BUILD_LIBRARY)
    # Find Python libraries
    # find_package(PythonLibs REQUIRED)
    # find_package(PythonLibs REQUIRED)
    # include_directories(${PYTHON_INCLUDE_DIRS})

    # Build as a shared library
    add_library(${PROJECT_NAME} SHARED
        "${PROJECT_SOURCE_DIR}/interpreter/libcuevm.cu"
        "${PROJECT_SOURCE_DIR}/interpreter/interpreter.cu"
        "${PROJECT_SOURCE_DIR}/interpreter/python_utils.cu"
        $<TARGET_OBJECTS:CuEVM>
        $<TARGET_OBJECTS:CuCrypto>
        $<TARGET_OBJECTS:CGBN>
        $<TARGET_OBJECTS:CuBigInt>
    )

    # Apply flags for library
    target_compile_options(${PROJECT_NAME} PRIVATE  $<$<COMPILE_LANGUAGE:CUDA>:--shared -Xcompiler '-fPIC'>)
    if(CMAKE_BUILD_TYPE MATCHES Debug)
        target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g>)
    endif()
    # set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME "cuevm")
    set_target_properties(${PROJECT_NAME}
    PROPERTIES
        OUTPUT_NAME "cuevm"
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_RUNTIME_LIBRARY Shared  # Ensure CUDA runtime library is shared
    )
    message(STATUS "Python include dirs: ${PYTHON_INCLUDE_DIRS}")
    message(STATUS "Python libraries: ${PYTHON_LIBRARIES}")
    # Link the library to its dependencies
    # our own libraries first
    target_link_libraries(${PROJECT_NAME} PRIVATE CuCrypto CuBigInt CuEVM)
    # then the external ones
    target_link_libraries(${PROJECT_NAME} PUBLIC gmp cjson ${PYTHON_LIBRARIES} ${CUDA_LIBRARIES})

    target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --std=c++20 -rdc=true --expt-relaxed-constexpr>)
    if(ENABLE_EIP_3155)
        target_compile_definitions(${PROJECT_NAME} PRIVATE EIP_3155)
    endif()

    target_compile_definitions(${PROJECT_NAME} PRIVATE EVM_VERSION=${EVM_VERSION})

    target_compile_definitions(${PROJECT_NAME} PRIVATE BUILD_LIBRARY)

else()

    set(GPU_INTERPRETER ${PROJECT_NAME}_GPU)
    # GPU
    add_executable(${GPU_INTERPRETER}
        "${PROJECT_SOURCE_DIR}/interpreter/interpreter.cu"
        $<TARGET_OBJECTS:CuEVM>
        $<TARGET_OBJECTS:CuCrypto>
        $<TARGET_OBJECTS:CGBN>
        $<TARGET_OBJECTS:CuBigInt>
    )

    set_target_properties(${GPU_INTERPRETER}
        PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
            POSITION_INDEPENDENT_CODE ON
    )

    # Link the library to its dependencies
    # our own libraries first
    target_link_libraries(${GPU_INTERPRETER} PRIVATE CuCrypto CuBigInt CuEVM)
    # then the external ones
    target_link_libraries(${GPU_INTERPRETER} PUBLIC gmp cjson ${CUDA_LIBRARIES})

    target_compile_options(${GPU_INTERPRETER} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --std=c++20 -rdc=true --expt-relaxed-constexpr>)
    if(ENABLE_EIP_3155)
        target_compile_definitions(${GPU_INTERPRETER} PRIVATE EIP_3155)
    endif()
    target_compile_definitions(${GPU_INTERPRETER} PRIVATE EVM_VERSION=${EVM_VERSION})

endif()


