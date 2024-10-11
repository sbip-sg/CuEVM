// CuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation
// Programme) Author: Stefan-Dan Ciocirlan Date: 2024-09-15
// SPDX-License-Identifier: MIT

#include <CuEVM/state/state_access.cuh>
#include <CuEVM/utils/error_codes.cuh>

namespace CuEVM {

__host__ __device__ state_access_t::state_access_t() { clear(); }

__host__ __device__ state_access_t::state_access_t(const state_access_t &other) : state_access_t() { duplicate(other); }

__host__ __device__ state_access_t::~state_access_t() { free(); }

__host__ __device__ void state_access_t::free() {
    __ONE_GPU_THREAD_BEGIN__
    if (flags != nullptr && no_accounts > 0) {
        std::free(flags);
    }
    __ONE_GPU_THREAD_END__
    state_t::free();
    clear();
}

__host__ void state_access_t::free_managed() {
    if (flags != nullptr && no_accounts > 0) {
        CUDA_CHECK(cudaFree(flags));
    }
    state_t::free_managed();
    clear();
}

__host__ __device__ void state_access_t::clear() {
    flags = nullptr;
    state_t::clear();
}

__host__ __device__ state_access_t &state_access_t::operator=(const state_access_t &other) {
    if (this != &other) {
        free();
        duplicate(other);
    }
    return *this;
}

__host__ __device__ void state_access_t::duplicate(const state_access_t &other) {
    __SHARED_MEMORY__ CuEVM::account_flags_t *tmp_flags;
    state_t::duplicate(other);
    if (no_accounts > 0) {
        __ONE_GPU_THREAD_BEGIN__
        tmp_flags = (CuEVM::account_flags_t *)malloc(sizeof(CuEVM::account_flags_t) * no_accounts);
        memcpy(tmp_flags, other.flags, no_accounts * sizeof(CuEVM::account_flags_t));
        __ONE_GPU_THREAD_END__
    } else {
        tmp_flags = nullptr;
    }
    flags = tmp_flags;
}

__host__ __device__ int32_t state_access_t::get_account(ArithEnv &arith, const bn_t &address, CuEVM::account_t &account,
                                                        const CuEVM::account_flags_t flag) {
    uint32_t index = 0;
    if (state_t::get_account_index(arith, address, index) == ERROR_SUCCESS) {
        flags[index].update(flag);
        account = accounts[index];
        return ERROR_SUCCESS;
    }
    return ERROR_STATE_ADDRESS_NOT_FOUND;
}

__host__ __device__ int32_t state_access_t::get_account(ArithEnv &arith, const bn_t &address,
                                                        CuEVM::account_t *&account_ptr,
                                                        const CuEVM::account_flags_t flag) {
    uint32_t index = 0;
    if (state_t::get_account_index(arith, address, index) == ERROR_SUCCESS) {
        flags[index].update(flag);
        account_ptr = &accounts[index];
        return ERROR_SUCCESS;
    }
    return ERROR_STATE_ADDRESS_NOT_FOUND;
}

__host__ __device__ int32_t state_access_t::add_account(const CuEVM::account_t &account,
                                                        const CuEVM::account_flags_t flag) {
    state_t::add_account(account);

    uint32_t index = no_accounts - 1;
    __SHARED_MEMORY__ CuEVM::account_flags_t *tmp_flags;
    __ONE_GPU_THREAD_BEGIN__
    // printf("no_accounts: %u\n", no_accounts);
    tmp_flags = (CuEVM::account_flags_t *)malloc(sizeof(CuEVM::account_flags_t) * no_accounts);
    memcpy(tmp_flags, flags, (no_accounts - 1) * sizeof(CuEVM::account_flags_t));

    if (flags != nullptr) {
        delete[] flags;
    }
    __ONE_GPU_THREAD_END__

    flags = tmp_flags;
    flags[index] = flag;

    // printf("after clear flags\n");
    return ERROR_SUCCESS;
}

__host__ __device__ int32_t state_access_t::add_duplicate_account(CuEVM::account_t *&account_ptr,
                                                                  CuEVM::account_t *&src_account_ptr,
                                                                  const CuEVM::account_flags_t flag) {
    CuEVM::account_flags_t no_storage_copy(ACCOUNT_NON_STORAGE_FLAG);
    __SHARED_MEMORY__ CuEVM::account_t *tmp_account_ptr;

    tmp_account_ptr = new CuEVM::account_t(src_account_ptr, no_storage_copy);

    // #ifdef __CUDA_ARCH__
    //     printf("TouchState::add_duplicate_account before add_account %d\n" ,threadIdx.x);
    // #endif
    int32_t error_code = add_account(*tmp_account_ptr, flag);
    // #ifdef __CUDA_ARCH__
    //     printf("TouchState::add_duplicate_account after add_account %d\n" ,threadIdx.x);
    // #endif
    // printf("after add_account\n");
    account_ptr = &accounts[no_accounts - 1];
    //  #ifdef __CUDA_ARCH__
    //     printf("TouchState::add_duplicate_account after assigning account_ptr %d, account_ptr %p\n" ,threadIdx.x,
    //     account_ptr);
    // #endif
    __ONE_GPU_THREAD_WOSYNC_BEGIN__
    delete tmp_account_ptr;
    __ONE_GPU_THREAD_WOSYNC_END__
    //  #ifdef __CUDA_ARCH__
    //     printf("TouchState::add_duplicate_account delete tmp_account_ptr %d, account_ptr %p\n" ,threadIdx.x,
    //     account_ptr);
    // #endif
    // printf("after delete tmp_account_ptr\n");
    return error_code;
}

__host__ __device__ int32_t state_access_t::add_new_account(ArithEnv &arith, const bn_t &address,
                                                            CuEVM::account_t *&account_ptr,
                                                            const CuEVM::account_flags_t flag) {
    __SHARED_MEMORY__ CuEVM::account_t *tmp_account_ptr;
    bn_t zero;
    cgbn_set_ui32(arith.env, zero, 0);
    // printf("before new CuEVM::account_t();\n");

    tmp_account_ptr = new CuEVM::account_t();
    tmp_account_ptr->set_address(arith, address);
    // default constructor did not set balance + nonce
    tmp_account_ptr->set_balance(arith, zero);
    tmp_account_ptr->set_nonce(arith, zero);

#ifdef __CUDA_ARCH__
    // printf("before add_account(*tmp_account_ptr, flag); %d\n", threadIdx.x);
#endif
    int32_t error_code = add_account(*tmp_account_ptr, flag);
    account_ptr = &accounts[no_accounts - 1];
    __ONE_GPU_THREAD_WOSYNC_BEGIN__
    delete tmp_account_ptr;
    __ONE_GPU_THREAD_WOSYNC_END__
    return error_code;
}

__host__ __device__ int32_t state_access_t::set_account(ArithEnv &arith, const CuEVM::account_t &account,
                                                        const CuEVM::account_flags_t flag) {
    if (update_account(arith, account, flag) != ERROR_SUCCESS) {
        return add_account(account, flag);
    } else {
        return ERROR_SUCCESS;
    }
}

__host__ __device__ int32_t state_access_t::update_account(ArithEnv &arith, const CuEVM::account_t &account,
                                                           const CuEVM::account_flags_t flag) {
    bn_t target_address;
    cgbn_load(arith.env, target_address, (cgbn_evm_word_t_ptr) & (account.address));
    uint32_t index = 0;
    if (state_t::get_account_index(arith, target_address, index) == ERROR_SUCCESS) {
        accounts[index].update(arith, account, flag);
        flags[index].update(flag);
        return ERROR_SUCCESS;
    }
    return ERROR_STATE_ADDRESS_NOT_FOUND;
}

__host__ __device__ int32_t state_access_t::update(ArithEnv &arith, const state_access_t &other) {
    int32_t error_code = ERROR_SUCCESS;
    for (uint32_t i = 0; i < other.no_accounts; i++) {
        // if update failed (not exist), add the account
        if (update_account(arith, other.accounts[i], other.flags[i]) != ERROR_SUCCESS) {
            error_code |= add_account(other.accounts[i], other.flags[i]);
        }
    }
    return error_code;
}

__host__ __device__ void state_access_t::print() {
    __ONE_GPU_THREAD_WOSYNC_BEGIN__
    printf("no_accounts: %u\n", no_accounts);
    for (uint32_t idx = 0; idx < no_accounts; idx++) {
        printf("accounts[%u]:\n", idx);
        accounts[idx].print();
        printf("flags[%u]:\n", idx);
        flags[idx].print();
    }
    __ONE_GPU_THREAD_WOSYNC_END__
}

__host__ cJSON *state_access_t::to_json() {
    cJSON *state_json = nullptr;
    cJSON *account_json = nullptr;
    char *hex_string_ptr = new char[CuEVM::word_size * 2 + 3];
    char *flag_string_ptr = new char[sizeof(uint32_t) * 2 + 3];
    state_json = cJSON_CreateObject();
    for (uint32_t idx = 0; idx < no_accounts; idx++) {
        accounts[idx].address.to_hex(hex_string_ptr, 0, 5);
        account_json = accounts[idx].to_json();
        cJSON_AddStringToObject(account_json, "flags", flags[idx].to_hex(flag_string_ptr));
        cJSON_AddItemToObject(state_json, hex_string_ptr, account_json);
    }
    delete[] hex_string_ptr;
    hex_string_ptr = nullptr;
    delete[] flag_string_ptr;
    flag_string_ptr = nullptr;
    return state_json;
}

__host__ int32_t state_access_t::get_account_index_evm(const evm_word_t &address, uint32_t &index) const {
    for (index = 0; index < no_accounts; index++) {
        if (accounts[index].address == address) {
            return ERROR_SUCCESS;
        }
    }
    return ERROR_STATE_ADDRESS_NOT_FOUND;
}

__host__ cJSON *state_access_t::merge_json(const state_t &state1, const state_access_t &state2) {
    cJSON *state_json = nullptr;
    cJSON *account_json = nullptr;
    cJSON *accounts_json = nullptr;

    state_json = cJSON_CreateObject();
    accounts_json = cJSON_CreateArray();
    cJSON_AddItemToObject(state_json, "accounts", accounts_json);
    uint8_t *writen_accounts;
    writen_accounts = new uint8_t[state2.no_accounts];
    memset(writen_accounts, 0, state2.no_accounts * sizeof(uint8_t));
    const CuEVM::account_t *account1_ptr = nullptr;
    const CuEVM::account_t *account2_ptr = nullptr;
    uint32_t jdx = 0;
    for (uint32_t idx = 0; idx < state1.no_accounts; idx++) {
        account1_ptr = &(state1.accounts[idx]);
        if (state2.get_account_index_evm(account1_ptr->address, jdx) == ERROR_SUCCESS) {
            account2_ptr = &(state2.accounts[jdx]);
            account_json = CuEVM::account_t::merge_json(account1_ptr, account2_ptr, state2.flags[jdx]);
            cJSON_AddItemToArray(accounts_json, account_json);
            writen_accounts[jdx] = 1;
        } else {
            account_json = CuEVM::account_t::merge_json(account1_ptr, account2_ptr, ACCOUNT_NONE_FLAG);
            cJSON_AddItemToArray(accounts_json, account_json);
        }
    }
    for (jdx = 0; jdx < state2.no_accounts; jdx++) {
        if (writen_accounts[jdx] == 0) {
            account2_ptr = &(state2.accounts[jdx]);
            account_json = CuEVM::account_t::merge_json(account1_ptr, account2_ptr, ACCOUNT_ALL_FLAG);
            cJSON_AddItemToArray(accounts_json, account_json);
        }
    }
    delete[] writen_accounts;
    return state_json;
}
__host__ state_access_t *state_access_t::get_cpu(uint32_t count) { return new state_access_t[count]; }

__host__ void state_access_t::cpu_free(state_access_t *cpu_states, uint32_t count) {
    if (cpu_states != nullptr) {
        for (uint32_t idx = 0; idx < count; idx++) {
            if (cpu_states[idx].no_accounts > 0) {
                delete[] cpu_states[idx].accounts;
                std::free(cpu_states[idx].flags);
            }
            cpu_states[idx].clear();
        }
        delete[] cpu_states;
    }
}

__host__ state_access_t *state_access_t::get_gpu_from_cpu(const state_access_t *cpu_states, uint32_t count) {
    state_access_t *gpu_states, *tmp_gpu_states;
    tmp_gpu_states = new state_access_t[count];
    for (uint32_t idx = 0; idx < count; idx++) {
        if (cpu_states[idx].no_accounts > 0) {
            tmp_gpu_states[idx].accounts =
                CuEVM::account_t::get_gpu_from_cpu(cpu_states[idx].accounts, cpu_states[idx].no_accounts);
            CUDA_CHECK(cudaMalloc((void **)&(tmp_gpu_states[idx].flags),
                                  cpu_states[idx].no_accounts * sizeof(CuEVM::account_flags_t)));
            CUDA_CHECK(cudaMemcpy(tmp_gpu_states[idx].flags, cpu_states[idx].flags,
                                  cpu_states[idx].no_accounts * sizeof(CuEVM::account_flags_t),
                                  cudaMemcpyHostToDevice));
            tmp_gpu_states[idx].no_accounts = cpu_states[idx].no_accounts;
        } else {
            tmp_gpu_states[idx].accounts = nullptr;
            tmp_gpu_states[idx].no_accounts = 0;
            tmp_gpu_states[idx].flags = nullptr;
        }
    }
    CUDA_CHECK(cudaMalloc((void **)&gpu_states, count * sizeof(state_access_t)));
    CUDA_CHECK(cudaMemcpy(gpu_states, tmp_gpu_states, count * sizeof(state_access_t), cudaMemcpyHostToDevice));
    for (uint32_t idx = 0; idx < count; idx++) {
        tmp_gpu_states[idx].clear();
    }
    delete[] tmp_gpu_states;
    return gpu_states;
}

__host__ void state_access_t::gpu_free(state_access_t *gpu_states, uint32_t count) {
    state_access_t *tmp_gpu_states = new state_access_t[count];
    CUDA_CHECK(cudaMemcpy(tmp_gpu_states, gpu_states, count * sizeof(state_access_t), cudaMemcpyDeviceToHost));
    for (uint32_t idx = 0; idx < count; idx++) {
        if (tmp_gpu_states[idx].no_accounts > 0) {
            CuEVM::account_t::free_gpu(tmp_gpu_states[idx].accounts, tmp_gpu_states[idx].no_accounts);
            CUDA_CHECK(cudaFree(tmp_gpu_states[idx].flags));
        }
        tmp_gpu_states[idx].clear();
    }
    delete[] tmp_gpu_states;
    CUDA_CHECK(cudaFree(gpu_states));
}

__host__ state_access_t *state_access_t::get_cpu_from_gpu(state_access_t *gpu_states, uint32_t count) {
    state_access_t *cpu_states, *tmp_cpu_states, *tmp_gpu_states;
    tmp_cpu_states = new state_access_t[count];
    cpu_states = new state_access_t[count];
    CUDA_CHECK(cudaMemcpy(cpu_states, gpu_states, count * sizeof(state_access_t), cudaMemcpyDeviceToHost));
    for (uint32_t idx = 0; idx < count; idx++) {
        if (cpu_states[idx].no_accounts > 0) {
            CUDA_CHECK(cudaMalloc((void **)&(tmp_cpu_states[idx].accounts),
                                  cpu_states[idx].no_accounts * sizeof(CuEVM::account_t)));
            tmp_cpu_states[idx].no_accounts = cpu_states[idx].no_accounts;
            CUDA_CHECK(cudaMalloc((void **)&(tmp_cpu_states[idx].flags),
                                  cpu_states[idx].no_accounts * sizeof(CuEVM::account_flags_t)));
        } else {
            tmp_cpu_states[idx].accounts = nullptr;
            tmp_cpu_states[idx].no_accounts = 0;
            tmp_cpu_states[idx].flags = nullptr;
        }
    }
    CUDA_CHECK(cudaMalloc((void **)&tmp_gpu_states, count * sizeof(state_access_t)));
    CUDA_CHECK(cudaMemcpy(tmp_gpu_states, tmp_cpu_states, count * sizeof(state_access_t), cudaMemcpyHostToDevice));

    state_access_t_transfer_kernel<<<count, 1>>>(tmp_gpu_states, gpu_states, count);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(tmp_cpu_states, tmp_gpu_states, count * sizeof(state_access_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(gpu_states));
    CUDA_CHECK(cudaFree(tmp_gpu_states));

    for (uint32_t idx = 0; idx < count; idx++) {
        if (tmp_cpu_states[idx].no_accounts > 0) {
            cpu_states[idx].flags =
                (CuEVM::account_flags_t *)malloc(tmp_cpu_states[idx].no_accounts * sizeof(CuEVM::account_flags_t));
            CUDA_CHECK(cudaMemcpy(cpu_states[idx].flags, tmp_cpu_states[idx].flags,
                                  tmp_cpu_states[idx].no_accounts * sizeof(CuEVM::account_flags_t),
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaFree(tmp_cpu_states[idx].flags));
            cpu_states[idx].accounts =
                CuEVM::account_t::get_cpu_from_gpu(tmp_cpu_states[idx].accounts, tmp_cpu_states[idx].no_accounts);
            cpu_states[idx].no_accounts = tmp_cpu_states[idx].no_accounts;
        } else {
            cpu_states[idx].accounts = nullptr;
            cpu_states[idx].no_accounts = 0;
            cpu_states[idx].flags = nullptr;
        }
        tmp_cpu_states[idx].clear();
    }
    delete[] tmp_cpu_states;
    return cpu_states;
}

__global__ void state_access_t_transfer_kernel(state_access_t *dst_instances, state_access_t *src_instances,
                                               uint32_t count) {
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
            memcpy(dst_instances[idx].flags, src_instances[idx].flags,
                   src_instances[idx].no_accounts * sizeof(CuEVM::account_flags_t));
            src_instances[idx].free();
        } else {
            dst_instances[idx].accounts = nullptr;
            dst_instances[idx].no_accounts = 0;
            dst_instances[idx].flags = nullptr;
        }
    }
}
}  // namespace CuEVM