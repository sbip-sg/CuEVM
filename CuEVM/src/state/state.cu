// CuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation
// Programme) Author: Stefan-Dan Ciocirlan Data: 2023-11-30
// SPDX-License-Identifier: MIT

#include <CuCrypto/keccak.cuh>
#include <CuEVM/state/state.cuh>
#include <CuEVM/utils/error_codes.cuh>

namespace CuEVM {

__host__ __device__ state_t::state_t() { clear(); }

__host__ __device__ state_t::state_t(const state_t &other) : state_t() { duplicate(other); }

__host__ __device__ state_t &state_t::operator=(const state_t &other) {
    if (this != &other) {
        duplicate(other);
    }
    return *this;
}
__host__ __device__ void state_t::duplicate(const state_t &other) {
    __SHARED_MEMORY__ CuEVM::account_t *tmp_accounts;
    free();  // free the current state
    no_accounts = other.no_accounts;
    if (no_accounts > 0) {
        __ONE_GPU_THREAD_BEGIN__
        tmp_accounts = (CuEVM::account_t *)malloc(no_accounts * sizeof(CuEVM::account_t));
        __ONE_GPU_THREAD_END__
        for (uint32_t idx = 0; idx < no_accounts; idx++) {
            tmp_accounts[idx].clear();
            tmp_accounts[idx] = other.accounts[idx];
        }
    } else {
        tmp_accounts = nullptr;
    }
    accounts = tmp_accounts;
}

__host__ __device__ state_t::~state_t() { free(); }

__host__ __device__ void state_t::free() {
    if (accounts != nullptr && no_accounts > 0) {
        for (uint32_t idx = 0; idx < no_accounts; idx++) {
            accounts[idx].free();
        }
        __ONE_GPU_THREAD_BEGIN__
        std::free(accounts);
        __ONE_GPU_THREAD_END__
    }
    clear();
}

__host__ void state_t::free_managed() {
    if (accounts != nullptr && no_accounts > 0) {
        CUDA_CHECK(cudaFree(accounts));
    }
    clear();
}

__host__ __device__ void state_t::clear() {
    accounts = nullptr;
    no_accounts = 0;
}

__host__ __device__ int32_t state_t::get_account_index(ArithEnv &arith, const bn_t &address, uint32_t &index) {
    for (index = 0; index < no_accounts; index++) {
        // #ifdef __CUDA_ARCH__
        //     printf("get_account_index, %d , accounts[index] %p thread %d\n", index, &(accounts[index].address),
        //     threadIdx.x);
        // #endif
        if (accounts[index].has_address(arith, address)) {
            // #ifdef __CUDA_ARCH__
            //     printf("get_account_index, has adddress thread %d\n", threadIdx.x);
            // #endif
            return ERROR_SUCCESS;
        }
    }
    return ERROR_STATE_ADDRESS_NOT_FOUND;
}

__host__ __device__ int32_t state_t::get_account(ArithEnv &arith, const bn_t &address, CuEVM::account_t &account) {
    uint32_t index;
    if (get_account_index(arith, address, index) == ERROR_SUCCESS) {
        account = accounts[index];
        return ERROR_SUCCESS;
    }
    return ERROR_STATE_ADDRESS_NOT_FOUND;
}

__host__ __device__ int32_t state_t::get_account(ArithEnv &arith, const bn_t &address, CuEVM::account_t *&account_ptr) {
    uint32_t index;
    if (get_account_index(arith, address, index) == ERROR_SUCCESS) {
        account_ptr = &accounts[index];
        return ERROR_SUCCESS;
    }
    return ERROR_STATE_ADDRESS_NOT_FOUND;
}

__host__ __device__ int32_t state_t::add_account(const CuEVM::account_t &account) {
    __SHARED_MEMORY__ CuEVM::account_t *tmp_accounts;
    __ONE_GPU_THREAD_BEGIN__
    tmp_accounts = (CuEVM::account_t *)malloc((no_accounts + 1) * sizeof(CuEVM::account_t));
    memcpy(tmp_accounts, accounts, no_accounts * sizeof(CuEVM::account_t));
    __ONE_GPU_THREAD_END__
    tmp_accounts[no_accounts].clear();
    tmp_accounts[no_accounts] = account;
    if (accounts != nullptr) {
        __ONE_GPU_THREAD_BEGIN__
        std::free(accounts);
        __ONE_GPU_THREAD_END__
    }
    accounts = tmp_accounts;
    no_accounts++;
    return ERROR_SUCCESS;
}

__host__ __device__ int32_t state_t::set_account(ArithEnv &arith, const CuEVM::account_t &account) {
    bn_t target_address;
    cgbn_load(arith.env, target_address, (cgbn_evm_word_t_ptr) & (account.address));
    for (uint32_t idx = 0; idx < no_accounts; idx++) {
        if (accounts[idx].has_address(arith, target_address)) {
            accounts[idx] = account;
            return ERROR_SUCCESS;
        }
    }

    return add_account(account);
}

__host__ __device__ int32_t state_t::has_account(ArithEnv &arith, const bn_t &address) {
    for (uint32_t idx = 0; idx < no_accounts; idx++) {
        if (accounts[idx].has_address(arith, address)) {
            return ERROR_SUCCESS;
        }
    }
    return ERROR_STATE_ADDRESS_NOT_FOUND;
}

__host__ __device__ int32_t state_t::update_account(ArithEnv &arith, const CuEVM::account_t &account) {
    bn_t target_address;
    cgbn_load(arith.env, target_address, (cgbn_evm_word_t_ptr) & (account.address));
    for (uint32_t idx = 0; idx < no_accounts; idx++) {
        if (accounts[idx].has_address(arith, target_address)) {
            accounts[idx].update(arith, account);
            return ERROR_SUCCESS;
        }
    }
    return add_account(account);
}

// __host__ __device__ int32_t state_t::is_empty_account(ArithEnv &arith,
//                                                       const bn_t &address) {
//     int32_t error_code;
//     uint32_t index;
//     error_code = get_account_index(arith, address, index);
//     return (error_code == ERROR_SUCCESS) ? accounts[index].is_empty()
//                                          : error_code;
// }

__host__ int32_t state_t::from_json(const cJSON *state_json, int32_t managed) {
    free();
    // if (!cJSON_IsArray(state_json)) return 0;
    no_accounts = cJSON_GetArraySize(state_json);
    if (no_accounts == 0) return 1;
    if (managed) {
        CUDA_CHECK(cudaMallocManaged((void **)&(accounts), no_accounts * sizeof(CuEVM::account_t)));
    } else {
        accounts = (CuEVM::account_t *)malloc(no_accounts * sizeof(CuEVM::account_t));
    }
    // for (uint32_t idx = 0; idx < no_accounts; idx++) {
    //     accounts[idx].clear();
    // }
    uint32_t idx = 0;
    cJSON *account_json;
    cJSON_ArrayForEach(account_json, state_json) { accounts[idx++].from_json(account_json, managed); }
    return ERROR_SUCCESS;
}

__host__ __device__ void state_t::print() {
    __ONE_GPU_THREAD_WOSYNC_BEGIN__
    printf("no_accounts: %u\n", no_accounts);
    for (uint32_t idx = 0; idx < no_accounts; idx++) {
        printf("accounts[%u]:\n", idx);
        accounts[idx].print();
    }
    __ONE_GPU_THREAD_WOSYNC_END__
}

__host__ cJSON *state_t::to_json() {
    cJSON *state_json = nullptr;
    cJSON *account_json = nullptr;
    char *hex_string_ptr = new char[CuEVM::word_size * 2 + 3];
    state_json = cJSON_CreateObject();
    for (uint32_t idx = 0; idx < no_accounts; idx++) {
        accounts[idx].address.to_hex(hex_string_ptr, 0, 5);
        account_json = accounts[idx].to_json();
        cJSON_AddItemToObject(state_json, hex_string_ptr, account_json);
    }
    delete[] hex_string_ptr;
    hex_string_ptr = nullptr;
    return state_json;
}

__host__ state_t *state_t::get_cpu(uint32_t count) { return new state_t[count]; }

__host__ void state_t::cpu_free(state_t *cpu_states, uint32_t count) {
    if (cpu_states != nullptr) {
        for (uint32_t idx = 0; idx < count; idx++) {
            if (cpu_states[idx].no_accounts > 0) {
                delete[] cpu_states[idx].accounts;
            }
            cpu_states[idx].clear();
        }
        delete[] cpu_states;
    }
}

__host__ state_t *state_t::get_gpu_from_cpu(const state_t *cpu_states, uint32_t count) {
    state_t *gpu_states, *tmp_gpu_states;
    tmp_gpu_states = new state_t[count];
    for (uint32_t idx = 0; idx < count; idx++) {
        if (cpu_states[idx].no_accounts > 0) {
            tmp_gpu_states[idx].accounts =
                CuEVM::account_t::get_gpu_from_cpu(cpu_states[idx].accounts, cpu_states[idx].no_accounts);
            tmp_gpu_states[idx].no_accounts = cpu_states[idx].no_accounts;
        } else {
            tmp_gpu_states[idx].accounts = nullptr;
            tmp_gpu_states[idx].no_accounts = 0;
        }
    }
    CUDA_CHECK(cudaMalloc((void **)&gpu_states, count * sizeof(state_t)));
    CUDA_CHECK(cudaMemcpy(gpu_states, tmp_gpu_states, count * sizeof(state_t), cudaMemcpyHostToDevice));
    for (uint32_t idx = 0; idx < count; idx++) {
        tmp_gpu_states[idx].clear();
    }
    delete[] tmp_gpu_states;
    return gpu_states;
}

__host__ void state_t::gpu_free(state_t *gpu_states, uint32_t count) {
    state_t *tmp_gpu_states = new state_t[count];
    CUDA_CHECK(cudaMemcpy(tmp_gpu_states, gpu_states, count * sizeof(state_t), cudaMemcpyDeviceToHost));
    for (uint32_t idx = 0; idx < count; idx++) {
        if (tmp_gpu_states[idx].no_accounts > 0) {
            CuEVM::account_t::free_gpu(tmp_gpu_states[idx].accounts, tmp_gpu_states[idx].no_accounts);
        }
        tmp_gpu_states[idx].clear();
    }
    delete[] tmp_gpu_states;
    CUDA_CHECK(cudaFree(gpu_states));
}

__host__ state_t *state_t::get_cpu_from_gpu(state_t *gpu_states, uint32_t count) {
    state_t *cpu_states, *tmp_cpu_states, *tmp_gpu_states;
    tmp_cpu_states = new state_t[count];
    cpu_states = new state_t[count];
    CUDA_CHECK(cudaMemcpy(cpu_states, gpu_states, count * sizeof(state_t), cudaMemcpyDeviceToHost));
    for (uint32_t idx = 0; idx < count; idx++) {
        if (cpu_states[idx].no_accounts > 0) {
            CUDA_CHECK(cudaMalloc((void **)&(tmp_cpu_states[idx].accounts),
                                  cpu_states[idx].no_accounts * sizeof(CuEVM::account_t)));
            tmp_cpu_states[idx].no_accounts = cpu_states[idx].no_accounts;
        } else {
            tmp_cpu_states[idx].accounts = nullptr;
            tmp_cpu_states[idx].no_accounts = 0;
        }
    }
    CUDA_CHECK(cudaMalloc((void **)&tmp_gpu_states, count * sizeof(state_t)));
    CUDA_CHECK(cudaMemcpy(tmp_gpu_states, tmp_cpu_states, count * sizeof(state_t), cudaMemcpyHostToDevice));

    state_t_transfer_kernel<<<count, 1>>>(tmp_gpu_states, gpu_states, count);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(tmp_cpu_states, tmp_gpu_states, count * sizeof(state_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(gpu_states));
    CUDA_CHECK(cudaFree(tmp_gpu_states));

    for (uint32_t idx = 0; idx < count; idx++) {
        if (tmp_cpu_states[idx].no_accounts > 0) {
            cpu_states[idx].accounts =
                CuEVM::account_t::get_cpu_from_gpu(tmp_cpu_states[idx].accounts, tmp_cpu_states[idx].no_accounts);
            cpu_states[idx].no_accounts = tmp_cpu_states[idx].no_accounts;
        } else {
            cpu_states[idx].accounts = nullptr;
            cpu_states[idx].no_accounts = 0;
        }
        tmp_cpu_states[idx].clear();
    }
    delete[] tmp_cpu_states;
    return cpu_states;
}

__global__ void state_t_transfer_kernel(state_t *dst_instances, state_t *src_instances, uint32_t count) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        if (src_instances[idx].no_accounts > 0) {
            memcpy(dst_instances[idx].accounts, src_instances[idx].accounts,
                   src_instances[idx].no_accounts * sizeof(CuEVM::account_t));
            dst_instances[idx].no_accounts = src_instances[idx].no_accounts;
            for (uint32_t idx2 = 0; idx2 < src_instances[idx].no_accounts; idx2++) {
                src_instances[idx].accounts[idx2].byte_code.clear();
                src_instances[idx].accounts[idx2].storage.clear();
            }
            src_instances[idx].free();
        } else {
            dst_instances[idx].accounts = nullptr;
            dst_instances[idx].no_accounts = 0;
        }
    }
}

}  // namespace CuEVM
