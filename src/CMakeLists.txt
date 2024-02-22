enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
find_package(GMP REQUIRED)
find_package(cJSON REQUIRED) # Assuming cJSON is available as a find package
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/CGBN/include)
include_directories(${GMP_INCLUDE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/include)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
# Set this to your CUDA device's compute capability
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})


# Check if building as a library or an executable
if(BUILD_LIBRARY)

    # Build as a shared library
    add_library(${PROJECT_NAME} SHARED libcuevm.cu)

    # Apply flags for library
    target_compile_options(${PROJECT_NAME}  PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--shared -Xcompiler '-fPIC'>)
    if(CMAKE_BUILD_TYPE MATCHES Debug)
        target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g>)
    endif()
    set_target_properties(${PROJECT_NAME}  PROPERTIES OUTPUT_NAME "cuevm")
    target_link_libraries(${PROJECT_NAME}  gmp cjson ${PYTHON_LIBRARIES})
else()
    # Build as an executable
    add_executable(${PROJECT_NAME} interpreter.cu)

    # Add specific NVCC flags using target_compile_options (if necessary)
    target_compile_options(${PROJECT_NAME}  PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --std=c++20 -rdc=true>)
    if(CMAKE_BUILD_TYPE MATCHES Debug)
        target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g>)
    endif()
    # Link libraries
    # target_link_libraries(cuevm CUDA::cudart ${GMP_LIBRARIES} ${cJSON_LIBRARIES})
    target_link_libraries(${PROJECT_NAME}  gmp cjson)
endif()

set_target_properties(${PROJECT_NAME}  PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${PROJECT_NAME}  PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
