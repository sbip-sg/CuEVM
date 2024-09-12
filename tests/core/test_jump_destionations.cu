// CuEVM: CUDA Ethereum Virtual Machine implementation
// Test file for jump_destinations_t class

#include <gtest/gtest.h>

#include <CuEVM/core/jump_destinations.cuh>
#include <CuEVM/utils/cuda_utils.cuh>
#include <CuEVM/utils/error_codes.cuh>
#include <CuEVM/utils/evm_defines.cuh>
#include <CuEVM/utils/opcodes.cuh>

__global__ void test_jump_destinations_kernel(CuEVM::byte_array_t* byte_code,
                                              uint32_t expected_size,
                                              uint32_t* expected_destinations,
                                              uint32_t* result) {
    int32_t instance =
        (blockIdx.x * blockDim.x + threadIdx.x) / CuEVM::cgbn_tpi;
    CuEVM::jump_destinations_t jump_destinations(byte_code[instance]);

    result[instance] = ERROR_SUCCESS;
    // Check has method
    for (uint32_t i = 0; i < expected_size; i++) {
        result[instance] |= jump_destinations.has(expected_destinations[i]);
    }

    result[instance] |=
        (jump_destinations.has(0xFFFFFFFF) == ERROR_INVALID_JUMP_DESTINATION
             ? ERROR_SUCCESS
             : ERROR_INVALID_JUMP_DESTINATION);
}

void run_test_GPU(uint8_t* bytecode_data, uint32_t bytecode_size,
                  uint32_t* expected_destinations, uint32_t expected_size) {
    CuEVM::byte_array_t* cpuArray = CuEVM::byte_array_t::get_cpu(2);
    CUDA_CHECK(cudaDeviceReset());
    // copy the bytecode data to the device
    for (int i = 0; i < 2; i++) {
        cpuArray[i].grow(bytecode_size, 1);
        memcpy(cpuArray[i].data, bytecode_data, bytecode_size);
    }

    CuEVM::byte_array_t* gpuArray =
        CuEVM::byte_array_t::gpu_from_cpu(cpuArray, 2);

    // Allocate memory for expected destinations on device
    uint32_t* d_expected_destinations;
    cudaMalloc(&d_expected_destinations, expected_size * sizeof(uint32_t));
    cudaMemcpy(d_expected_destinations, expected_destinations,
               expected_size * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // Allocate memory for result on device
    uint32_t* d_result;
    cudaMalloc(&d_result, 2 * sizeof(uint32_t));

    // Launch kernel to test jump_destinations_t
    test_jump_destinations_kernel<<<2, CuEVM::cgbn_tpi>>>(
        gpuArray, expected_size, d_expected_destinations, d_result);
    cudaDeviceSynchronize();

    CuEVM::byte_array_t::gpu_free(gpuArray, 2);
    CuEVM::byte_array_t::cpu_free(cpuArray, 2);

    // Copy result back to host
    uint32_t* h_result;
    h_result = (uint32_t*)malloc(2 * sizeof(uint32_t));
    cudaMemcpy(h_result, d_result, 2 * sizeof(uint32_t),
               cudaMemcpyDeviceToHost);

    for (int i = 0; i < 2; i++) {
        EXPECT_EQ(h_result[i], ERROR_SUCCESS);
    }

    // Free device memory
    cudaFree(d_expected_destinations);
    cudaFree(d_result);
    CUDA_CHECK(cudaDeviceReset());
}

void run_test_CPU(uint8_t* bytecode_data, uint32_t bytecode_size,
                  uint32_t* expected_destinations, uint32_t expected_size) {
    CuEVM::byte_array_t byte_code(bytecode_data, bytecode_size);

    CuEVM::jump_destinations_t jump_destinations(byte_code);

    // Check has method
    for (uint32_t i = 0; i < expected_size; i++) {
        EXPECT_EQ(jump_destinations.has(expected_destinations[i]),
                  ERROR_SUCCESS);
    }

    // Check invalid jump destination
    EXPECT_EQ(jump_destinations.has(0xFFFFFFFF),
              ERROR_INVALID_JUMP_DESTINATION);
}

void run_test(uint8_t* bytecode_data, uint32_t bytecode_size,
              uint32_t* expected_destinations, uint32_t expected_size) {
    run_test_CPU(bytecode_data, bytecode_size, expected_destinations,
                 expected_size);
    run_test_GPU(bytecode_data, bytecode_size, expected_destinations,
                 expected_size);
}

TEST(JumpDestinationsTest, BasicTest) {
    uint8_t bytecode_data[] = {0x60, 0x00, OP_JUMPDEST,
                               0x60, 0x00, OP_JUMPDEST};
    uint32_t expected_destinations[] = {2, 5};
    run_test(bytecode_data, sizeof(bytecode_data) / sizeof(bytecode_data[0]),
             expected_destinations,
             sizeof(expected_destinations) / sizeof(expected_destinations[0]));
}

TEST(JumpDestinationsTest, NoJumpDestTest) {
    uint8_t bytecode_data[] = {0x60, 0x00, 0x60, 0x01, 0x60, 0x02};
    uint32_t expected_destinations[] = {};
    run_test(bytecode_data, sizeof(bytecode_data) / sizeof(bytecode_data[0]),
             expected_destinations, 0);
}

TEST(JumpDestinationsTest, MultipleJumpDestTest) {
    uint8_t bytecode_data[] = {OP_JUMPDEST, 0x60, 0x00,       OP_JUMPDEST,
                               0x60,        0x00, OP_JUMPDEST};
    uint32_t expected_destinations[] = {0, 3, 6};
    run_test(bytecode_data, sizeof(bytecode_data) / sizeof(bytecode_data[0]),
             expected_destinations,
             sizeof(expected_destinations) / sizeof(expected_destinations[0]));
}