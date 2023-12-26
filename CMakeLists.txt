cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(cuevm
        VERSION 0.1.0
        DESCRIPTION "A CUDA EVM implementation"
        LANGUAGES CUDA CXX)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})


set(PROJECT_SOURCE_FILES_DIR "src")

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
find_package(GMP REQUIRED)
set(CMAKE_CXX_STANDARD 20)
list(APPEND CUDA_NVCC_FLAGS -lineinfo,--std=c++20,-rdc=true,-lcudadevrt,-lcjson,-lstdc++,-lm,-lgmp)

add_executable(cuevm src/interpreter.cu)
target_compile_features(cuevm PUBLIC cxx_std_20)


include_directories(./CGBN/include)
target_link_libraries(${PROJECT_NAME} CUDA::cudart gmp libgmp cjson libcjson)


set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

# Doxygen

# look for Doxygen package
find_package(Doxygen)

if (DOXYGEN_FOUND)
    # set input and output files
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs_doxygen/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile.out)

    # request to configure the file
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
    message("Doxygen build started")

    # Note: do not put "ALL" - this builds docs together with application EVERY TIME!
    add_custom_target( docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM )
else (DOXYGEN_FOUND)
  message("Doxygen need to be installed to generate the doxygen documentation")
endif (DOXYGEN_FOUND)