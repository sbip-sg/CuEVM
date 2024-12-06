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
    CuEVM::account_t *tmp_accounts;
    free();  // free the current state
    no_accounts = other.no_accounts;
    if (no_accounts > 0) {
        tmp_accounts = (CuEVM::account_t *)malloc(no_accounts * sizeof(CuEVM::account_t));
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
        std::free(accounts);
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

__host__ __device__ int32_t state_t::get_account_index(const evm_word_t *address, uint32_t &index) {
    for (index = 0; index < no_accounts; index++) {
        if (accounts[index].address == *address) {
            return ERROR_SUCCESS;
        }
    }

    return ERROR_STATE_ADDRESS_NOT_FOUND;
}

__host__ __device__ int32_t state_t::get_account(const evm_word_t *address, CuEVM::account_t &account) {
    uint32_t index;
    if (get_account_index(address, index) == ERROR_SUCCESS) {
        account = accounts[index];
        return ERROR_SUCCESS;
    }
    return ERROR_STATE_ADDRESS_NOT_FOUND;
}

__host__ __device__ int32_t state_t::get_account(const evm_word_t *address, CuEVM::account_t *&account_ptr) {
    uint32_t index;
    if (get_account_index(address, index) == ERROR_SUCCESS) {
        account_ptr = &accounts[index];
        return ERROR_SUCCESS;
    }
    return ERROR_STATE_ADDRESS_NOT_FOUND;
}

__host__ __device__ int32_t state_t::add_account(const CuEVM::account_t &account) {
    CuEVM::account_t *tmp_accounts;

    tmp_accounts = (CuEVM::account_t *)malloc((no_accounts + 1) * sizeof(CuEVM::account_t));
    memcpy(tmp_accounts, accounts, no_accounts * sizeof(CuEVM::account_t));
    tmp_accounts[no_accounts].clear();
    tmp_accounts[no_accounts] = account;
    if (accounts != nullptr) {
        std::free(accounts);
    }
    accounts = tmp_accounts;
    no_accounts++;
    return ERROR_SUCCESS;
}

__host__ __device__ int32_t state_t::set_account(const CuEVM::account_t &account) {
    for (uint32_t idx = 0; idx < no_accounts; idx++) {
        if (accounts[idx].has_address(&(account.address))) {
            accounts[idx] = account;
            return ERROR_SUCCESS;
        }
    }

    return add_account(account);
}

__host__ __device__ int32_t state_t::update_account(const CuEVM::account_t &account,
                                                    const CuEVM::account_flags_t flag) {
    for (uint32_t idx = 0; idx < no_accounts; idx++) {
        if (accounts[idx].has_address(&(account.address))) {
            accounts[idx].update(account, flag);
            return ERROR_SUCCESS;
        }
    }
    return add_account(account);
}

// __host__ __device__ int32_t update(ArithEnv &arith, CuEVM::account_t *accounts, CuEVM::account_flags_t *flags,
__host__ __device__ int32_t state_t::update(const CuEVM::account_t *_accounts, const CuEVM::account_flags_t *_flags,
                                            uint32_t account_count) {
    int32_t error_code = ERROR_SUCCESS;
    for (uint32_t i = 0; i < account_count; i++) {
        // if update failed (not exist), add the account
        if (update_account(_accounts[i], _flags[i]) != ERROR_SUCCESS) {
            error_code |= add_account(_accounts[i]);
        }
    }
    return error_code;
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

}  // namespace CuEVM
