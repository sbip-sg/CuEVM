cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

option(DOCS "Generating the documentation for the code" OFF)
option(CPU "Generating the CPU version of CuEVM " ON)
option(GPU "Generating the GPU version of CuEVM " OFF)
option(ENABLE_EIP_3155 "Enable EIP_3155 tracer" ON)
option(ENABLE_PAIRING_CODE "Enable ec pairing precompile code" OFF)
option(ENABLE_EIP_3155_OPTIONAL "Enable EIP_3155 optional tracer" ON)
option(TESTS "Enable tests" ON)
option(TESTS_GPU "Enable GPU tests" OFF)
option(BUILD_LIBRARY "Build Python library" OFF)
set(EVM_VERSION "SHANGHAI" CACHE STRING "EVM version to use")
set(CGBN_TPI "32" CACHE STRING "Thread per instance for CGBN")
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

if (CPU OR GPU OR TESTS OR BUILD_LIBRARY)
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
endif()

if(CPU)
    set(CPU_INTERPRETER ${PROJECT_NAME}_CPU)
    add_executable(${CPU_INTERPRETER}
        "${PROJECT_SOURCE_DIR}/interpreter/interpreter.cu"
        $<TARGET_OBJECTS:CuEVM>
        $<TARGET_OBJECTS:CuCrypto>
        $<TARGET_OBJECTS:CGBN>
        $<TARGET_OBJECTS:CuBigInt>
    )

    set_target_properties(${CPU_INTERPRETER}
        PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
            POSITION_INDEPENDENT_CODE ON
    )

    # Link the library to its dependencies
    # our own libraries first
    target_link_libraries(${CPU_INTERPRETER} PRIVATE CGBN CuCrypto CuBigInt CuEVM)
    # then the external ones
    target_link_libraries(${CPU_INTERPRETER} PUBLIC gmp cjson ${CUDA_LIBRARIES})

    target_compile_options(${CPU_INTERPRETER}  PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --std=c++20 -rdc=true --expt-relaxed-constexpr>)
    if(ENABLE_EIP_3155)
        target_compile_definitions(${CPU_INTERPRETER} PRIVATE EIP_3155)
    endif()
    if(ENABLE_EIP_3155_OPTIONAL)
        target_compile_definitions(${CPU_INTERPRETER} PRIVATE EIP_3155_OPTIONAL)
    endif()
    target_compile_definitions(${CPU_INTERPRETER} PRIVATE EVM_VERSION=${EVM_VERSION})
    
    target_compile_definitions(${CPU_INTERPRETER} PRIVATE CGBN_TPI=${CGBN_TPI})
endif()

if(GPU)

    if(BUILD_LIBRARY)
        # Find Python libraries
        find_package(PythonLibs REQUIRED)
        include_directories(${PYTHON_INCLUDE_DIRS})
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

        # Link the library to its dependencies
        # our own libraries first
        target_link_libraries(${PROJECT_NAME} PRIVATE CGBN CuCrypto CuBigInt CuEVM)
        # then the external ones
        target_link_libraries(${PROJECT_NAME} PUBLIC gmp cjson ${PYTHON_LIBRARIES} ${CUDA_LIBRARIES})

        target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --std=c++20 -rdc=true --expt-relaxed-constexpr>)
        if(ENABLE_EIP_3155)
            target_compile_definitions(${PROJECT_NAME} PRIVATE EIP_3155)
        endif()
        if(ENABLE_EIP_3155_OPTIONAL)
            target_compile_definitions(${PROJECT_NAME} PRIVATE EIP_3155_OPTIONAL)
        endif()
        target_compile_definitions(${PROJECT_NAME} PRIVATE EVM_VERSION=${EVM_VERSION})

        target_compile_definitions(${PROJECT_NAME} PRIVATE GPU)

        target_compile_definitions(${PROJECT_NAME} PRIVATE CGBN_TPI=${CGBN_TPI})
        
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
        target_link_libraries(${GPU_INTERPRETER} PRIVATE CGBN CuCrypto CuBigInt CuEVM)
        # then the external ones
        target_link_libraries(${GPU_INTERPRETER} PUBLIC gmp cjson ${CUDA_LIBRARIES})

        target_compile_options(${GPU_INTERPRETER} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --std=c++20 -rdc=true --expt-relaxed-constexpr>)
        if(ENABLE_EIP_3155)
            target_compile_definitions(${GPU_INTERPRETER} PRIVATE EIP_3155)
        endif()
        if(ENABLE_EIP_3155_OPTIONAL)
            target_compile_definitions(${GPU_INTERPRETER} PRIVATE EIP_3155_OPTIONAL)
        endif()
        target_compile_definitions(${GPU_INTERPRETER} PRIVATE EVM_VERSION=${EVM_VERSION})

        target_compile_definitions(${GPU_INTERPRETER} PRIVATE GPU)

        target_compile_definitions(${GPU_INTERPRETER} PRIVATE CGBN_TPI=${CGBN_TPI})
    endif() 

endif()

if(TESTS)
    find_package(GTest REQUIRED)
    include_directories(${GTEST_INCLUDE_DIRS})
    find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    #add_subdirectory("${PROJECT_SOURCE_DIR}/tests")
    file(GLOB_RECURSE TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cu")
    set(TEST_EXECUTABLE ${PROJECT_NAME}_tests)
    add_executable(${TEST_EXECUTABLE}
        ${TEST_SOURCES}
        $<TARGET_OBJECTS:CuEVM>
        $<TARGET_OBJECTS:CuCrypto>
        $<TARGET_OBJECTS:CGBN>
        $<TARGET_OBJECTS:CuBigInt>
    )
    set_target_properties(${TEST_EXECUTABLE}
    PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        POSITION_INDEPENDENT_CODE ON
    )
    # Link the library to its dependencies
    # our own libraries first
    target_link_libraries(${TEST_EXECUTABLE} PRIVATE CGBN CuCrypto CuBigInt CuEVM)
    # then the external ones
    target_link_libraries(${TEST_EXECUTABLE} PUBLIC gmp cjson ${CUDA_LIBRARIES})
    target_link_libraries(${TEST_EXECUTABLE} PUBLIC ${GTEST_LIBRARIES} ${CUDART_LIBRARY})

    target_compile_options(${TEST_EXECUTABLE}  PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --std=c++20 -rdc=true --expt-relaxed-constexpr>)
    if(ENABLE_EIP_3155)
        target_compile_definitions(${TEST_EXECUTABLE} PRIVATE EIP_3155)
    endif()
    if(ENABLE_EIP_3155_OPTIONAL)
        target_compile_definitions(${TEST_EXECUTABLE} PRIVATE EIP_3155_OPTIONAL)
    endif()
    target_compile_definitions(${TEST_EXECUTABLE} PRIVATE EVM_VERSION=${EVM_VERSION})
    if (TESTS_GPU)
        target_compile_definitions(${TEST_EXECUTABLE} PRIVATE GPU)
    endif()
    # Add the test executable to CTest
    enable_testing()
    add_test(NAME ${TEST_EXECUTABLE} COMMAND ${TEST_EXECUTABLE})
endif()
