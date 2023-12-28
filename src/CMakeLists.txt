enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
find_package(GMP REQUIRED)
set(CMAKE_CXX_STANDARD 20)
list(APPEND CUDA_NVCC_FLAGS -lineinfo,--std=c++20,-rdc=true,-lcudadevrt,-lcjson,-lstdc++,-lm,-lgmp)

add_executable(${PROJECT_NAME} "interpreter.cu")
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)

target_link_libraries(${PROJECT_NAME} CUDA::cudart gmp libgmp cjson libcjson)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
