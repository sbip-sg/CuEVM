enable_language(CUDA)
find_package(GMP REQUIRED)
find_package(cJSON REQUIRED) # Assuming cJSON is available as a find package

# include_directories(${PROJECT_SOURCE_DIR}/CGBN/include)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(${PROJECT_SOURCE_DIR}/bigint)
include_directories(${GMP_INCLUDE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/include)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Set this to your CUDA device's compute capability
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})

add_executable(${PROJECT_NAME}
    core/block_info.cu
    core/byte_array.cu
    core/jump_destinations.cu
    core/memory.cu
    core/message.cu
    core/stack.cu
    core/transaction.cu
    operations/arithmetic.cu
    operations/bitwise.cu
    operations/compare.cu
    operations/environment.cu
    operations/flow.cu
    operations/log.cu
    operations/memory.cu
    operations/stack.cu
    operations/storage.cu
    operations/system.cu
    state/access_state.cu
    state/account.cu
    state/logs.cu
    state/storage.cu
    state/touch_state.cu
    state/world_state.cu
    uitls/arith.cu
    uitls/cuda_utils.cu
    uitls/evm_utils.cu
    evm_call_state.cu
    gas_cost.cu
    evm.cu
)

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${PROJECT_NAME}  PROPERTIES CUDA_RUNTIME_LIBRARY Shared)

#set_target_properties(${PROJECT_NAME}  PROPERTIES CUDA_RUNTIME_LIBRARY Shared)

# Add specific NVCC flags using target_compile_options (if necessary)
target_compile_options(${PROJECT_NAME}  PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --std=c++20 -rdc=true>)

# Link libraries
# target_link_libraries(cuevm CUDA::cudart ${GMP_LIBRARIES} ${cJSON_LIBRARIES})
target_link_libraries(${PROJECT_NAME} gmp cjson CGBN CuCrypto)
# target_link_libraries(${PROJECT_NAME} PRIVATE )

