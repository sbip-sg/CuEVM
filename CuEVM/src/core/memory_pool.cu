#include <assert.h>

#include <CuEVM/core/memory_pool.cuh>

namespace CuEVM::memory_pool {
__device__ memory_pool_t* global_memory_pool = nullptr;
__device__ evm_word_t* preallocated_stack_base = nullptr;
__device__ uint8_t* preallocated_return_data_base = nullptr;
__device__ SnapshotValue* preallocated_snapshot_values = nullptr;
__device__ ValueStatus** preallocated_snapshot_restore_ptr = nullptr;
__device__ uint8_t* preallocated_memory_base = nullptr;
__device__ CuEVM::EccConstants* ecc_constants_ptr = nullptr;
__host__ void create_memory_pool(uint32_t num_instances, uint32_t num_accounts, uint32_t num_devices) {
    for (int i = 0; i < num_devices; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        memory_pool_t* memory_pool = new memory_pool_t();
        memory_pool->num_instances = num_instances;

        cudaMalloc(&memory_pool->stack_base, num_instances * memory_pool_stack_preallocate * sizeof(evm_word_t));
        // printf("host: allocated stack base %p size %d\n", memory_pool->stack_base,
        //        num_instances * memory_pool_stack_preallocate);

        CUDA_CHECK(cudaMalloc(&memory_pool->call_context,
                              num_instances * memory_pool_call_context_preallocate * sizeof(evm_call_context_t)));

        CUDA_CHECK(cudaMalloc(&memory_pool->prealloc_stack_instances,
                              num_instances * memory_pool_call_context_preallocate * sizeof(evm_stack_t)));

        CUDA_CHECK(cudaMalloc(&memory_pool->prealloc_mem_instances,
                              num_instances * memory_pool_call_context_preallocate * sizeof(evm_memory_t)));

        // printf("host: allocated memory instances  %p size %d\n", &memory_pool->prealloc_mem_instances,
        //        num_instances * memory_pool_call_context_preallocate * sizeof(evm_memory_t));
        CUDA_CHECK(cudaMalloc(&memory_pool->return_data_base,
                              num_instances * memory_pool_return_data_preallocate * sizeof(uint8_t)));
        // printf("host: allocated return data base %p size %d\n", memory_pool->return_data_base,
        //        num_instances * memory_pool_return_data_preallocate);

        cudaMalloc(&memory_pool->snapshot_states_pool,
                   snapshot_account_pool_size * num_instances * sizeof(CuEVM::SnapshotState));
        cudaMalloc(&memory_pool->snapshot_account_counts, num_instances * sizeof(uint16_t));
        cudaMemset(memory_pool->snapshot_account_counts, 0, num_instances * sizeof(uint16_t));
        cudaMalloc(&memory_pool->snapshot_slot_counts, num_instances * sizeof(uint16_t));
        cudaMemset(memory_pool->snapshot_slot_counts, 0, num_instances * sizeof(uint16_t));

        memory_pool_t* d_memory_pool;
        CUDA_CHECK(cudaMalloc(&d_memory_pool, sizeof(memory_pool_t)));
        assert(d_memory_pool != nullptr);
        CUDA_CHECK(cudaMemcpy(d_memory_pool, memory_pool, sizeof(memory_pool_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyToSymbol(global_memory_pool, &d_memory_pool, sizeof(memory_pool_t*)));

        SnapshotValue* d_preallocated_snapshot_values;
        cudaMalloc(&d_preallocated_snapshot_values,
                   num_instances * memory_pool_snapshot_preallocate_slots * sizeof(SnapshotValue));
        cudaMemset(d_preallocated_snapshot_values, 0,
                   num_instances * memory_pool_snapshot_preallocate_slots * sizeof(SnapshotValue));
        SnapshotValue** d_preallocated_snapshot_restore_ptr;
        cudaMalloc(&d_preallocated_snapshot_restore_ptr,
                   num_instances * memory_pool_snapshot_preallocate_slots * sizeof(ValueStatus*));
        cudaMemset(d_preallocated_snapshot_restore_ptr, 0,
                   num_instances * memory_pool_snapshot_preallocate_slots * sizeof(ValueStatus*));

        uint8_t* d_preallocated_memory_base;
        cudaMalloc(&d_preallocated_memory_base, num_instances * memory_prealloc_size * sizeof(uint8_t));
        cudaMemset(d_preallocated_memory_base, 0, num_instances * memory_prealloc_size * sizeof(uint8_t));

        // printf("host: allocated memory instances  %p size %d\n", d_preallocated_memory_base,
        //        num_instances * memory_prealloc_size * sizeof(uint8_t));
        // copy pointer to preallocated stack base
        cudaMemcpyToSymbol(preallocated_stack_base, &memory_pool->stack_base, sizeof(evm_word_t*));
        cudaMemcpyToSymbol(preallocated_return_data_base, &memory_pool->return_data_base, sizeof(uint8_t*));
        cudaMemcpyToSymbol(preallocated_snapshot_values, &d_preallocated_snapshot_values, sizeof(SnapshotValue*));
        cudaMemcpyToSymbol(preallocated_snapshot_restore_ptr, &d_preallocated_snapshot_restore_ptr,
                           sizeof(ValueStatus**));
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
}

__host__ void free_memory_pool(uint32_t num_devices) {
    for (int i = 0; i < num_devices; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        // Get the memory_pool pointer from device
        memory_pool_t* d_memory_pool;
        CUDA_CHECK(cudaMemcpyFromSymbol(&d_memory_pool, global_memory_pool, sizeof(memory_pool_t*)));

        // Free memory allocated in memory_pool
        CUDA_CHECK(cudaFree(d_memory_pool->stack_base));
        CUDA_CHECK(cudaFree(d_memory_pool->call_context));
        CUDA_CHECK(cudaFree(d_memory_pool->prealloc_stack_instances));
        CUDA_CHECK(cudaFree(d_memory_pool->prealloc_mem_instances));
        CUDA_CHECK(cudaFree(d_memory_pool->return_data_base));
        CUDA_CHECK(cudaFree(d_memory_pool->snapshot_states_pool));
        CUDA_CHECK(cudaFree(d_memory_pool->snapshot_account_counts));
        CUDA_CHECK(cudaFree(d_memory_pool->snapshot_slot_counts));

        // Free the memory_pool struct itself
        CUDA_CHECK(cudaFree(d_memory_pool));

        // Free preallocated memory
        CUDA_CHECK(cudaFree(preallocated_stack_base));
        CUDA_CHECK(cudaFree(preallocated_return_data_base));
        CUDA_CHECK(cudaFree(preallocated_snapshot_values));
        CUDA_CHECK(cudaFree(preallocated_snapshot_restore_ptr));
        CUDA_CHECK(cudaFree(preallocated_memory_base));

        // Free ECC constants
        CuEVM::EccConstants* d_ecc_constants;
        CUDA_CHECK(cudaMemcpyFromSymbol(&d_ecc_constants, ecc_constants_ptr, sizeof(CuEVM::EccConstants*)));
        CUDA_CHECK(cudaFree(d_ecc_constants));
    }
}
__host__ void clear_memory_pool(uint32_t num_devices) {
    for (int i = 0; i < num_devices; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        // Get the memory_pool pointer from device
        memory_pool_t* d_memory_pool = global_memory_pool;
        CUDA_CHECK(cudaMemcpyFromSymbol(&d_memory_pool, global_memory_pool, sizeof(memory_pool_t*)));
        memory_pool_t* memory_pool = new memory_pool_t();
        CUDA_CHECK(cudaMemcpy(memory_pool, d_memory_pool, sizeof(memory_pool_t), cudaMemcpyDeviceToHost));
        // Clear memory allocated in memory_pool
        CUDA_CHECK(cudaMemset(memory_pool->stack_base, 0,
                              memory_pool->num_instances * memory_pool_stack_preallocate * sizeof(evm_word_t)));
        CUDA_CHECK(
            cudaMemset(memory_pool->call_context, 0,
                       memory_pool->num_instances * memory_pool_call_context_preallocate * sizeof(evm_call_context_t)));
        CUDA_CHECK(cudaMemset(memory_pool->prealloc_stack_instances, 0,
                              memory_pool->num_instances * memory_pool_call_context_preallocate * sizeof(evm_stack_t)));
        CUDA_CHECK(
            cudaMemset(memory_pool->prealloc_mem_instances, 0,
                       memory_pool->num_instances * memory_pool_call_context_preallocate * sizeof(evm_memory_t)));
        CUDA_CHECK(cudaMemset(memory_pool->return_data_base, 0,
                              memory_pool->num_instances * memory_pool_return_data_preallocate * sizeof(uint8_t)));
        CUDA_CHECK(cudaMemset(memory_pool->snapshot_states_pool, 0,
                              snapshot_account_pool_size * memory_pool->num_instances * sizeof(CuEVM::SnapshotState)));
        CUDA_CHECK(cudaMemset(memory_pool->snapshot_account_counts, 0, memory_pool->num_instances * sizeof(uint16_t)));
        CUDA_CHECK(cudaMemset(memory_pool->snapshot_slot_counts, 0, memory_pool->num_instances * sizeof(uint16_t)));

        // Clear preallocated memory
        SnapshotValue* d_preallocated_snapshot_values;
        ValueStatus** d_preallocated_snapshot_restore_ptr;
        uint8_t* d_preallocated_memory_base;

        CUDA_CHECK(cudaMemcpyFromSymbol(&d_preallocated_snapshot_values, preallocated_snapshot_values,
                                        sizeof(SnapshotValue*)));
        CUDA_CHECK(cudaMemcpyFromSymbol(&d_preallocated_snapshot_restore_ptr, preallocated_snapshot_restore_ptr,
                                        sizeof(ValueStatus**)));
        CUDA_CHECK(cudaMemcpyFromSymbol(&d_preallocated_memory_base, preallocated_memory_base, sizeof(uint8_t*)));

        CUDA_CHECK(
            cudaMemset(d_preallocated_snapshot_values, 0,
                       memory_pool->num_instances * memory_pool_snapshot_preallocate_slots * sizeof(SnapshotValue)));
        CUDA_CHECK(
            cudaMemset(d_preallocated_snapshot_restore_ptr, 0,
                       memory_pool->num_instances * memory_pool_snapshot_preallocate_slots * sizeof(ValueStatus*)));
        CUDA_CHECK(cudaMemset(d_preallocated_memory_base, 0,
                              memory_pool->num_instances * memory_prealloc_size * sizeof(uint8_t)));
    }
}

__device__ evm_call_context_t* get_call_context(uint16_t depth) {
    assert(global_memory_pool != nullptr);
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
#ifdef DEBUG_PERF
        printf("stack dynamic allocation\n");
#endif
        return new evm_stack_t();
    }
}
__device__ CuEVM::SnapshotState* get_snapshot_state() {
    // printf("get snapshot account instance %u counts %u\n", INSTANCE_GLOBAL_IDX,
    //        CuEVM::memory_pool::global_memory_pool->snapshot_counts[INSTANCE_GLOBAL_IDX]);
    uint32_t accounts_count = CuEVM::memory_pool::global_memory_pool->snapshot_account_counts[INSTANCE_GLOBAL_IDX]++;

    if (accounts_count < snapshot_account_pool_size) {
        // printf("get snapshot account instance %u counts %u\n", INSTANCE_GLOBAL_IDX, accounts_count);
        return &CuEVM::memory_pool::global_memory_pool
                    ->snapshot_states_pool[INSTANCE_GLOBAL_IDX + global_state_db_ptr->num_states * accounts_count];
    } else {
#ifdef DEBUG_PERF
        printf("snapshot account pool is full for instance %u, create a new one\n", INSTANCE_GLOBAL_IDX);
#endif
        SnapshotState* tmp = new CuEVM::SnapshotState();

        return tmp;
    }
}

__device__ uint32_t get_next_snapshot_offset() {
    return CuEVM::memory_pool::global_memory_pool->snapshot_slot_counts[INSTANCE_GLOBAL_IDX]++;
}

__device__ void reset_snapshot_slot_offset(uint32_t offset) {
    CuEVM::memory_pool::global_memory_pool->snapshot_slot_counts[INSTANCE_GLOBAL_IDX] = offset;
}

__device__ void reset_snapshot_account_offset(uint32_t offset) {
    CuEVM::memory_pool::global_memory_pool->snapshot_account_counts[INSTANCE_GLOBAL_IDX] = offset;
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
#ifdef DEBUG_PERF
        printf("memory dynamic allocation\n");
#endif
        return new evm_memory_t();
    }
}

}  // namespace CuEVM::memory_pool
