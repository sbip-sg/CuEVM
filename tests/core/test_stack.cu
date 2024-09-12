#include <CGBN/cgbn.h>
#include <gtest/gtest.h>

#include <CuEVM/core/stack.cuh>
#include <CuEVM/utils/arith.cuh>
#include <CuEVM/utils/error_codes.cuh>

class EvmStackTest : public ::testing::Test {
   protected:
    CuEVM::ArithEnv arith;
    CuEVM::evm_stack_t stack;

    EvmStackTest() : arith(cgbn_no_checks), stack() {}

    void SetUp() override {
        // Any additional setup can be done here
    }

    void TearDown() override {
        // Any cleanup can be done here
    }
};

// Test Initial Size
TEST_F(EvmStackTest, InitialSize) { EXPECT_EQ(stack.size(), 0); }

// Test Push Operation
TEST_F(EvmStackTest, PushOperation) {
    CuEVM::bn_t value;

    cgbn_set_ui32(arith.env, value, 42);
    int32_t error_code = stack.push(arith, value);
    EXPECT_EQ(error_code, ERROR_SUCCESS);
    EXPECT_EQ(stack.size(), 1);

    cgbn_set_ui32(arith.env, value, 84);
    error_code = stack.push(arith, value);
    EXPECT_EQ(error_code, ERROR_SUCCESS);
    EXPECT_EQ(stack.size(), 2);
}

// Test Pop Operation
TEST_F(EvmStackTest, PopOperation) {
    CuEVM::bn_t value, popped_value;
    uint32_t popped_ui32;

    // Push values onto the stack
    cgbn_set_ui32(arith.env, value, 42);
    stack.push(arith, value);
    cgbn_set_ui32(arith.env, value, 84);
    stack.push(arith, value);

    // Pop values from the stack
    int32_t error_code = stack.pop(arith, popped_value);
    EXPECT_EQ(error_code, ERROR_SUCCESS);
    popped_ui32 = cgbn_get_ui32(arith.env, popped_value);
    EXPECT_EQ(popped_ui32, 84);
    EXPECT_EQ(stack.size(), 1);

    error_code = stack.pop(arith, popped_value);
    EXPECT_EQ(error_code, ERROR_SUCCESS);
    popped_ui32 = cgbn_get_ui32(arith.env, popped_value);
    EXPECT_EQ(popped_ui32, 42);
    EXPECT_EQ(stack.size(), 0);
}

// Test Stack Underflow
TEST_F(EvmStackTest, StackUnderflow) {
    CuEVM::bn_t popped_value;

    // Attempt to pop from an empty stack
    int32_t error_code = stack.pop(arith, popped_value);
    EXPECT_EQ(error_code, ERROR_STACK_UNDERFLOW);
    EXPECT_EQ(stack.size(), 0);
}

// Test Pushx Operation
TEST_F(EvmStackTest, PushxOperation) {
    uint8_t src_byte_data[4] = {0x01, 0x02, 0x03, 0x04};
    CuEVM::bn_t popped_value;
    uint32_t popped_ui32;

    // Push 4 bytes onto the stack
    int32_t error_code = stack.pushx(arith, 4, src_byte_data, 4);
    EXPECT_EQ(error_code, ERROR_SUCCESS);
    EXPECT_EQ(stack.size(), 1);

    // Pop the value from the stack to verify it
    error_code = stack.pop(arith, popped_value);
    EXPECT_EQ(error_code, ERROR_SUCCESS);
    popped_ui32 = cgbn_get_ui32(arith.env, popped_value);
    EXPECT_EQ(popped_ui32, 0x01020304);
    EXPECT_EQ(stack.size(), 0);
}

// Test Dupx Operation
TEST_F(EvmStackTest, DupxOperation) {
    CuEVM::bn_t value, popped_value;
    uint32_t popped_ui32;

    // Push values onto the stack
    cgbn_set_ui32(arith.env, value, 42);
    stack.push(arith, value);
    cgbn_set_ui32(arith.env, value, 84);
    stack.push(arith, value);

    // Duplicate the value at index 0
    int32_t error_code = stack.dupx(arith, 1);
    EXPECT_EQ(error_code, ERROR_SUCCESS);
    EXPECT_EQ(stack.size(), 3);

    // Pop the duplicated value
    error_code = stack.pop(arith, popped_value);
    EXPECT_EQ(error_code, ERROR_SUCCESS);
    popped_ui32 = cgbn_get_ui32(arith.env, popped_value);
    EXPECT_EQ(popped_ui32, 84);
    EXPECT_EQ(stack.size(), 2);
}

// Test Get Index Operation
TEST_F(EvmStackTest, GetIndexOperation) {
    CuEVM::bn_t value, popped_value;
    uint32_t popped_ui32;

    // Push values onto the stack
    cgbn_set_ui32(arith.env, value, 42);
    stack.push(arith, value);
    cgbn_set_ui32(arith.env, value, 84);
    stack.push(arith, value);

    // Get the value at index 1
    int32_t error_code = stack.get_index(arith, 1, popped_value);
    EXPECT_EQ(error_code, ERROR_SUCCESS);
    popped_ui32 = cgbn_get_ui32(arith.env, popped_value);
    EXPECT_EQ(popped_ui32, 84);
    EXPECT_EQ(stack.size(), 2);
}

// Test Swap Operation
TEST_F(EvmStackTest, SwapOperation) {
    CuEVM::bn_t value, popped_value;
    uint32_t popped_ui32;

    // Push values onto the stack
    cgbn_set_ui32(arith.env, value, 42);
    stack.push(arith, value);
    cgbn_set_ui32(arith.env, value, 84);
    stack.push(arith, value);

    // Swap the values at index 0 and 1 using swapx
    int32_t error_code = stack.swapx(arith, 1);
    EXPECT_EQ(error_code, ERROR_SUCCESS);

    // Get the value at index 0
    error_code = stack.get_index(arith, 1, popped_value);
    EXPECT_EQ(error_code, ERROR_SUCCESS);
    popped_ui32 = cgbn_get_ui32(arith.env, popped_value);
    EXPECT_EQ(popped_ui32, 42);

    // Get the value at index 1
    error_code = stack.get_index(arith, 2, popped_value);
    EXPECT_EQ(error_code, ERROR_SUCCESS);
    popped_ui32 = cgbn_get_ui32(arith.env, popped_value);
    EXPECT_EQ(popped_ui32, 84);
}

__global__ void testKernel(CuEVM::evm_stack_t* gpuStack, uint32_t count,
                           uint32_t* result) {
    int32_t instance =
        (blockIdx.x * blockDim.x + threadIdx.x) / CuEVM::cgbn_tpi;
    if (instance >= count) return;
    result[instance] = ERROR_SUCCESS;

    CuEVM::ArithEnv arith(cgbn_no_checks);
    CuEVM::bn_t value;
    cgbn_set_ui32(arith.env, value, 42);
    result[instance] |= gpuStack[instance].push(arith, value);
    result[instance] |=
        gpuStack[instance].size() == 1 ? ERROR_SUCCESS : __LINE__;

    CuEVM::bn_t popped_value;
    result[instance] |= gpuStack[instance].pop(arith, popped_value);
    result[instance] |=
        cgbn_get_ui32(arith.env, popped_value) == 42 ? ERROR_SUCCESS : __LINE__;

    result[instance] |=
        gpuStack[instance].size() == 0 ? ERROR_SUCCESS : __LINE__;
    result[instance] |=
        gpuStack[instance].pop(arith, popped_value) == ERROR_STACK_UNDERFLOW
            ? ERROR_SUCCESS
            : __LINE__;

    result[instance] |=
        gpuStack[instance].size() == 0 ? ERROR_SUCCESS : __LINE__;
    uint8_t src_byte_data[4] = {0x01, 0x02, 0x03, 0x04};
    result[instance] |= gpuStack[instance].pushx(arith, 4, src_byte_data, 4);
    result[instance] |=
        gpuStack[instance].size() == 1 ? ERROR_SUCCESS : __LINE__;
    result[instance] |= gpuStack[instance].pop(arith, popped_value);
    result[instance] |= cgbn_get_ui32(arith.env, popped_value) == 0x01020304
                            ? ERROR_SUCCESS
                            : __LINE__;

    result[instance] |=
        gpuStack[instance].size() == 0 ? ERROR_SUCCESS : __LINE__;
    cgbn_set_ui32(arith.env, value, 42);
    result[instance] |= gpuStack[instance].push(arith, value);
    cgbn_set_ui32(arith.env, value, 84);
    result[instance] |= gpuStack[instance].push(arith, value);
    result[instance] |=
        gpuStack[instance].size() == 2 ? ERROR_SUCCESS : __LINE__;

    result[instance] |= gpuStack[instance].dupx(arith, 1);
    result[instance] |=
        gpuStack[instance].size() == 3 ? ERROR_SUCCESS : __LINE__;
    result[instance] |= gpuStack[instance].pop(arith, popped_value);
    result[instance] |=
        cgbn_get_ui32(arith.env, popped_value) == 84 ? ERROR_SUCCESS : __LINE__;
    result[instance] |=
        gpuStack[instance].size() == 2 ? ERROR_SUCCESS : __LINE__;

    result[instance] |= gpuStack[instance].get_index(arith, 1, popped_value);
    result[instance] |=
        cgbn_get_ui32(arith.env, popped_value) == 84 ? ERROR_SUCCESS : __LINE__;
    result[instance] |=
        gpuStack[instance].size() == 2 ? ERROR_SUCCESS : __LINE__;
    result[instance] |= gpuStack[instance].swapx(arith, 1);
    result[instance] |= gpuStack[instance].get_index(arith, 1, popped_value);
    result[instance] |=
        cgbn_get_ui32(arith.env, popped_value) == 42 ? ERROR_SUCCESS : __LINE__;
    result[instance] |= gpuStack[instance].get_index(arith, 2, popped_value);
    result[instance] |=
        cgbn_get_ui32(arith.env, popped_value) == 84 ? ERROR_SUCCESS : __LINE__;
    result[instance] |=
        gpuStack[instance].size() == 2 ? ERROR_SUCCESS : __LINE__;

    result[instance] |= gpuStack[instance].pop(arith, popped_value);
    result[instance] |=
        cgbn_get_ui32(arith.env, popped_value) == 42 ? ERROR_SUCCESS : __LINE__;
    result[instance] |=
        gpuStack[instance].size() == 1 ? ERROR_SUCCESS : __LINE__;
}

// Test push Operation on GPU
TEST_F(EvmStackTest, PushOperationGPU) {
    CuEVM::evm_stack_t* cpuStack = CuEVM::evm_stack_t::get_cpu(2);
    CUDA_CHECK(cudaDeviceReset());
    CuEVM::evm_stack_t* gpuStack =
        CuEVM::evm_stack_t::gpu_from_cpu(cpuStack, 2);
    uint32_t* d_result;
    cudaMalloc(&d_result, 2 * sizeof(uint32_t));
    testKernel<<<2, CuEVM::cgbn_tpi>>>(gpuStack, 2, d_result);
    CUDA_CHECK(cudaDeviceSynchronize());
    CuEVM::evm_stack_t* results = CuEVM::evm_stack_t::cpu_from_gpu(gpuStack, 2);
    for (int i = 0; i < 2; i++) {
        EXPECT_EQ(results[i].size(), 1);
        EXPECT_EQ(results[i].stack_base[0], 84);
    }

    uint32_t* h_result;
    h_result = (uint32_t*)malloc(2 * sizeof(uint32_t));
    CUDA_CHECK(cudaMemcpy(h_result, d_result, 2 * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    for (int i = 0; i < 2; i++) {
        EXPECT_EQ(h_result[i], ERROR_SUCCESS);
    }
    free(h_result);
    CUDA_CHECK(cudaFree(d_result));
    CuEVM::evm_stack_t::cpu_free(cpuStack, 2);
    CuEVM::evm_stack_t::cpu_free(results, 2);
    CUDA_CHECK(cudaDeviceReset());
}