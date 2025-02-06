#include <CuEVM/core/memory_pool.cuh>

namespace CuEVM::memory_pool {
__device__ memory_pool_t* global_memory_pool;
__device__ evm_word_t* preallocated_stack_base;
__device__ uint8_t* preallocated_return_data_base;
__device__ SnapshotValue* preallocated_snapshot_values;
__device__ ValueStatus** preallocated_snapshot_restore_ptr;
__device__ uint8_t* preallocated_memory_base;
__device__ CuEVM::EccConstants* ecc_constants_ptr;
__host__ __host__ void create_memory_pool(uint32_t num_instances, uint32_t num_accounts) {
    memory_pool_t* memory_pool = new memory_pool_t();
    memory_pool->num_instances = num_instances;
    memory_pool->current_stack_page_size = memory_pool_stack_preallocate;
    memory_pool->count_words = 0;
    memory_pool->count_call_context = 0;
    // cuda malloc memory for words and call_context
    // memory_pool->words = new evm_word_t[num_instances * memory_pool_word_preallocate];
    // memory_pool->call_context = new evm_call_context_t[num_instances * memory_pool_call_context_preallocate];
    cudaMalloc(&memory_pool->stack_base, num_instances * memory_pool_stack_preallocate * sizeof(evm_word_t));
    printf("host: allocated stack base %p size %d\n", memory_pool->stack_base,
           num_instances * memory_pool_stack_preallocate);

    cudaMalloc(&memory_pool->call_context,
               num_instances * memory_pool_call_context_preallocate * sizeof(evm_call_context_t));

    cudaMalloc(&memory_pool->prealloc_stack_instances,
               num_instances * memory_pool_call_context_preallocate * sizeof(evm_stack_t));

    cudaMalloc(&memory_pool->prealloc_mem_instances,
               num_instances * memory_pool_call_context_preallocate * sizeof(evm_memory_t));

    cudaMalloc(&memory_pool->return_data_base, num_instances * memory_pool_return_data_preallocate * sizeof(uint8_t));
    printf("host: allocated return data base %p size %d\n", memory_pool->return_data_base,
           num_instances * memory_pool_return_data_preallocate);

    memory_pool_t* d_memory_pool;
    cudaMalloc(&d_memory_pool, sizeof(memory_pool_t));
    cudaMemcpy(d_memory_pool, memory_pool, sizeof(memory_pool_t), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(global_memory_pool, &d_memory_pool, sizeof(memory_pool_t*));

    SnapshotValue* d_preallocated_snapshot_values;
    cudaMalloc(&d_preallocated_snapshot_values,
               num_accounts * num_instances * memory_pool_snapshot_preallocate * sizeof(SnapshotValue));
    cudaMemset(d_preallocated_snapshot_values, 0,
               num_accounts * num_instances * memory_pool_snapshot_preallocate * sizeof(SnapshotValue));
    SnapshotValue** d_preallocated_snapshot_restore_ptr;
    cudaMalloc(&d_preallocated_snapshot_restore_ptr,
               num_accounts * num_instances * memory_pool_snapshot_preallocate * sizeof(ValueStatus*));
    cudaMemset(d_preallocated_snapshot_restore_ptr, 0,
               num_accounts * num_instances * memory_pool_snapshot_preallocate * sizeof(ValueStatus*));

    uint8_t* d_preallocated_memory_base;
    cudaMalloc(&d_preallocated_memory_base, num_instances * memory_prealloc_size * sizeof(uint8_t));
    cudaMemset(d_preallocated_memory_base, 0, num_instances * memory_prealloc_size * sizeof(uint8_t));
    // copy pointer to preallocated stack base
    cudaMemcpyToSymbol(preallocated_stack_base, &memory_pool->stack_base, sizeof(evm_word_t*));
    cudaMemcpyToSymbol(preallocated_return_data_base, &memory_pool->return_data_base, sizeof(uint8_t*));
    cudaMemcpyToSymbol(preallocated_snapshot_values, &d_preallocated_snapshot_values, sizeof(SnapshotValue*));
    cudaMemcpyToSymbol(preallocated_snapshot_restore_ptr, &d_preallocated_snapshot_restore_ptr, sizeof(ValueStatus**));
    cudaMemcpyToSymbol(preallocated_memory_base, &d_preallocated_memory_base, sizeof(uint8_t*));
    delete memory_pool;

    // ecc constants
    CuEVM::EccConstants* host_ecc_constants_ptr = new CuEVM::EccConstants();
    CuEVM::EccConstants* d_ecc_constants_ptr;
    cudaMalloc(&d_ecc_constants_ptr, sizeof(CuEVM::EccConstants));
    cudaMemcpy(d_ecc_constants_ptr, host_ecc_constants_ptr, sizeof(CuEVM::EccConstants), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(ecc_constants_ptr, &d_ecc_constants_ptr, sizeof(CuEVM::EccConstants*));
    delete host_ecc_constants_ptr;
}

__device__ evm_call_context_t* get_call_context(uint16_t depth) {
    if (depth < memory_pool_call_context_preallocate) {
        return &global_memory_pool->call_context[depth * global_memory_pool->num_instances + INSTANCE_GLOBAL_IDX];
    } else {
        return new evm_call_context_t();
    }
}

__device__ evm_stack_t* get_stack(uint16_t depth) {
    // printf(
    //     " depth %d get_stack instance %u index %u stack_base %p\n", depth, INSTANCE_GLOBAL_IDX,
    //     depth * global_memory_pool->num_instances + INSTANCE_GLOBAL_IDX,
    //     &global_memory_pool->prealloc_stack_instances[depth * global_memory_pool->num_instances +
    //     INSTANCE_GLOBAL_IDX]);
    if (depth < memory_pool_call_context_preallocate) {
        return &global_memory_pool
                    ->prealloc_stack_instances[depth * global_memory_pool->num_instances + INSTANCE_GLOBAL_IDX];
    } else {
        return new evm_stack_t();
    }
}

__device__ evm_memory_t* get_memory(uint16_t depth) {
    // printf(
    //     " depth %d get_memory instance %u index %u memory_base %p\n", depth, INSTANCE_GLOBAL_IDX,
    //     depth * global_memory_pool->num_instances + INSTANCE_GLOBAL_IDX,
    //     &global_memory_pool->prealloc_mem_instances[depth * global_memory_pool->num_instances +
    //     INSTANCE_GLOBAL_IDX]);
    if (depth < memory_pool_call_context_preallocate) {
        return &global_memory_pool
                    ->prealloc_mem_instances[depth * global_memory_pool->num_instances + INSTANCE_GLOBAL_IDX];
    } else {
        return new evm_memory_t();
    }
}

}  // namespace CuEVM::memory_pool
