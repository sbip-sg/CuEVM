#pragma once
#include <CuEVM/core/evm_word.cuh>
namespace CuEVM {
constexpr uint32_t worldstate_addresses_size = 32;
constexpr uint32_t worldstate_storage_values_size = 1024;

constexpr uint32_t account_prealloc_keys_size = 4;  // configurable keys per account
constexpr uint32_t value_page_size = 32;
constexpr uint32_t account_page_size = 16;
// heuristic size for the bytecode hex string to keep everything within 1MB
constexpr uint32_t byte_code_hex_size = 32 * max_code_size;

constexpr uint32_t dynamic_pool_base_size = 16;  // multiply by 2 every time
// store both the value and the status of the key
struct ValueStatus {
    evm_word_t original_value;
    evm_word_t value;
    bool is_warm = false;

    __host__ __device__ ValueStatus() {}
    // Constructor to initialize with a value and default is_warm to false
    __device__ void set_value(const evm_word_t *val, bool warm_status) {
        value = *val;
        is_warm = warm_status;
    }

    // the only time we set the original value is when we read from the pre-state db
    __host__ __device__ void from_hex(const char *hex_str) {
        value.from_hex(hex_str);
        original_value = value;
        is_warm = false;
    }
    // Assignment operator to assign value and reset is_warm to false
    __host__ __device__ ValueStatus &operator=(const evm_word_t &val) {
        value = val;
        is_warm = false;
        return *this;
    }
    __host__ __device__ ValueStatus &operator=(const ValueStatus &val) {
        value = val.value;
        is_warm = val.is_warm;
        return *this;
    }
    __host__ __device__ void print() {
        value.print();
        printf(" is_warm: %d\n", is_warm);
    }
};

struct KeyOffset {
    evm_word_t key;
    uint32_t offset;
};

struct SnapshotValue {
    evm_word_t value;
    bool is_warm = false;
};

struct SnapshotStoragePage {
    evm_word_t *keys[snapshot_page_size];
    SnapshotValue *values[snapshot_page_size];
    SnapshotStoragePage *next_page = nullptr;
};

struct StateDbStoragePage {
    evm_word_t *keys[value_page_size];
    ValueStatus *values[value_page_size];
    StateDbStoragePage *next_page = nullptr;
};

namespace memory_pool {

extern __device__ evm_word_t *preallocated_stack_base;
extern __device__ uint8_t *preallocated_return_data_base;
extern __device__ SnapshotValue *preallocated_snapshot_values;
extern __device__ ValueStatus **preallocated_snapshot_restore_ptr;  // store the original value to restore
}  // namespace memory_pool
}  // namespace CuEVM
