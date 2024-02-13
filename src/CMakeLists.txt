enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
find_package(GMP REQUIRED)
find_package(cJSON REQUIRED) # Assuming cJSON is available as a find package

include_directories(${GMP_INCLUDE_DIR})

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Set this to your CUDA device's compute capability
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})

add_executable(${PROJECT_NAME} interpreter.cu)
set_target_properties(${PROJECT_NAME}  PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${PROJECT_NAME}  PROPERTIES CUDA_RUNTIME_LIBRARY Shared)

# Add specific NVCC flags using target_compile_options (if necessary)
target_compile_options(${PROJECT_NAME}  PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --std=c++20 -rdc=true>)

# Link libraries
# target_link_libraries(cuevm CUDA::cudart ${GMP_LIBRARIES} ${cJSON_LIBRARIES})
target_link_libraries(${PROJECT_NAME}  gmp cjson)

