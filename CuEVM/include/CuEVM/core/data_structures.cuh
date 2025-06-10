#pragma once
#include <CuEVM/core/evm_word.cuh>
#include <CuEVM/utils/ecc.cuh>
#include <CuEVM/utils/evm_defines.cuh>
#include <vector>
namespace CuEVM {

constexpr uint32_t worldstate_addresses_size = 32;
constexpr uint32_t worldstate_storage_values_size = 1024;
#ifdef BUILD_GO_LIBRARY
constexpr uint32_t account_prealloc_keys_size = 128;  // configurable keys per account for "pre state"  
#else 
constexpr uint32_t account_prealloc_keys_size = 32;  // configurable keys per account for "pre state"
#endif
constexpr uint32_t memory_prealloc_size = 8096;          // memory (bytes) preallocated for each instance
constexpr uint32_t memory_pool_stack_preallocate = 128;  // number of stack elements preallocated for each instance

constexpr uint32_t memory_pool_call_context_preallocate = 8;     // number of call contexts for each instance
constexpr uint32_t memory_pool_return_data_preallocate = 128;    // number of return data bytes for each instance
constexpr uint32_t memory_pool_snapshot_preallocate_slots = 32;  // number of snapshot slots for each instance

constexpr uint32_t value_page_size = 16; // number of storage slots per page when expanding storage
constexpr uint32_t snapshot_account_pool_size = 32;  // number of snapshot accounts for each instance
constexpr uint32_t snapshot_page_size = 8;
// each snapshot state keeps track of the touched accounts warming up in the context
constexpr uint32_t preallocated_touched_accounts_size = 4;
constexpr uint32_t preallocated_touched_storage_keys_size = account_prealloc_keys_size/2;



constexpr CONSTANT uint32_t serialized_worldstate_addresses_size = 32;
constexpr CONSTANT uint32_t serialized_worldstate_storage_slots = 512;

// specific implementation constants
// constexpr CONSTANT uint32_t initial_storage_capacity = 8;

struct SnapshotValue {
    evm_word_t value;
    bool is_warm = false;
};
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
        original_value = val.original_value;
        is_warm = val.is_warm;
        return *this;
    }
    __host__ __device__ void print() {
        printf("value: ");
        value.print();
        printf("original value: ");
        original_value.print();
        printf(" is_warm: %d\n", is_warm);
    }
};

struct SnapshotStoragePage {
    // contiguous
    // size snapshot_page_size
    ValueStatus *restore_ptr[snapshot_page_size];
    SnapshotValue values[snapshot_page_size];
    SnapshotStoragePage *next_page = nullptr;
};

struct StateDbStoragePage {
    // contiguous
    // size value_page_size
    evm_word_t keys[value_page_size];
    ValueStatus values[value_page_size];
    StateDbStoragePage *next_page = nullptr;
};
// store balance and code change in subcall
struct SnapshotAccount {
    int32_t address_index;
    evm_word_t balance;
    uint32_t storage_size;
    uint32_t code_size = 0;
    uint8_t *code = nullptr;
    SnapshotAccount *next_account = nullptr;
};

struct SnapshotState {
    // store only the modified fields
    // uint16_t depth = 0;
    evm_word_t address;
    // evm_word_t balance;
    // uint32_t nonce = 0;
    uint32_t storage_size = 0;
    uint32_t preallocated_offset = 0;  // the beginning offset of the preallocated storage
    // uint32_t code_size = 0;
    // uint8_t *code = nullptr;
    SnapshotStoragePage *storage_page = nullptr;
    SnapshotState *next_state = nullptr;
    SnapshotAccount *accounts = nullptr;
    // number of account changes balances and codes, recorded by accounts
    uint32_t diff_account_counts = 0;
    // number of account from cold to warm
    uint32_t touched_account_counts = 0;
    uint32_t touched_storage_counts = 0;
    // arrays to store the accounts list warmed during execution
    // positive value : address index in the address list;
    // negative value : index in dynamic account list
    int32_t preallocated_touched_accounts[preallocated_touched_accounts_size];  // store the list of account warmed up
                                                                                // in this state
    int32_t *dynamic_touched_accounts = nullptr;
    // arrays to store the key from blank to warm during execution
    evm_word_t preallocated_touched_storage_keys[preallocated_touched_storage_keys_size];
    evm_word_t *dynamic_touched_storage_keys = nullptr;

    __host__ __device__ SnapshotState() : storage_size(0) {}

    __host__ __device__ ~SnapshotState();
    // revert this state and return the next state
    __device__ SnapshotState *revert();
    __device__ void set_blank_key(const evm_word_t *key);
    __device__ void set_touched_account(const int32_t address_index);
    __device__ void set_account(const evm_word_t *balance, const uint32_t nonce);
    __device__ int32_t find_dynamic_offset(ValueStatus *value);
    __device__ void set_storage(ValueStatus *value);
    __device__ void print() const;
    __device__ void clear();
};
namespace transaction {
class TransactionList {
   public:
    uint32_t size;
    // shared among all transactions (eth-tests)
    evm_word_t nonce;
    // allow different sender in fuzzing mode
#ifdef BUILD_GO_LIBRARY
    evm_word_t *sender;
#else
    evm_word_t sender;
#endif
    evm_word_t to;
    evm_word_t max_fee_per_gas;
    evm_word_t max_priority_fee_per_gas;
    evm_word_t gas_price;
    uint16_t type;
    // different for each transaction (eth-tests)
    evm_word_t *value;
    gas_t *gas_limit;
    uint8_t *call_data;
    uint32_t *call_data_offset;
    uint32_t *call_data_size;
    // TODO: access list

    // different for each transaction (go library)
#ifdef BUILD_GO_LIBRARY
    gas_t *block_number;
    gas_t *time_stamp;
#endif
    __host__ __device__ void print();
};
}  // namespace transaction

namespace memory {
// to change for making more optimal memory allocation current 1KB
// constexpr CONSTANT uint32_t page_size = 1024U;
__device__ void warp_cooperative_set(uint8_t *ptr1, const uint8_t *ptr2, uint32_t length);
__device__ void warp_cooperative_setzero(uint8_t *ptr1, uint32_t length);
/**
 * The memory data structure.
 */
struct evm_memory_t {
    uint32_t preallocated_base_offset;  // the start offset of the preallocated memory in the memory pool
    uint32_t size;                      /**< The size of the memory acceesed by now (YP: \f$32 \dot \mu_{i}\f$)*/
    gas_t memory_cost;                  /**< The memory cost (YP: \f$M(\mu_{i})\f$)*/
    uint8_t *dynamic_data = nullptr;    // pointer to the dynamic memory

    /**
     * The default constructor.
     */
    __host__ __device__ evm_memory_t() : preallocated_base_offset(0), size(0) { memory_cost = 0; }

    __device__ void init(uint32_t preallocated_base_offset) {
        // printf("init memory preallocated_base_offset %d\n", preallocated_base_offset);
        // printf("memory pointer %p\n", this);
        this->preallocated_base_offset = min(preallocated_base_offset, memory_prealloc_size);
        this->size = 0;
        this->memory_cost = 0;
        this->dynamic_data = nullptr;
    }

    /**
     * the destructor
     */
    __host__ __device__ ~evm_memory_t() {
        memory_cost = 0;
        size = 0;
    }

    /**
     * Print the memory data structure.
     */
    __host__ __device__ void print() const;

    /**
     * Get the json object from the memory data structure.
     * @return The json object.
     */
    __host__ cJSON *to_json() const;

    /**
     * Increase the memory cost.
     * @param[in] memory_expansion_cost The memory expansion cost.
     */
    __device__ void increase_memory_cost(gas_t memory_expansion_cost);

    /**
     * Increase the memory for the given offset if needed.
     * @param[in] new_size The new size of the memory.
     * @return 0 if success, otherwise the error code.
     */
    __device__ int32_t grow(uint32_t new_size);

    /**
     * Get the a pointer to the given memory data.
     * @param[in] index The index of the memory access.
     * @param[in] length The length of the memory access.
     * @param[out] data The pointer to the memory data.
     * @return 0 if success, otherwise the error code.
     */
    __device__ int32_t get(const uint32_t index, const uint32_t length, uint8_t *&data_);

    /**
     * Copy the given memory data.
     * @param[in] index The index of the memory access.
     * @param[in] length The length of the memory access.
     * @param[out] data The pointer to the memory data.
     * @return 0 if success, otherwise the error code.
     */
    __device__ int32_t copy(const uint32_t index, const uint32_t length, uint8_t *&data_);

    /**
     * Set the given memory data. Outside available_size is 0.
     * @param[in] data The data to be set.
     * @param[in] index The index of the memory access.
     * @param[in] length The length of the memory access.
     * @return 0 if success, otherwise the error code.
     */
    __device__ int32_t set(uint8_t *data_, uint32_t data_size, const uint32_t index, const uint32_t length);

    __device__ int32_t set_buffer_data(uint8_t *data_, uint32_t data_offset, uint32_t data_size, const uint32_t index,
                                       const uint32_t length);

    /**
     * Set the given memory data to zero.
     * @param[in] index The index of the memory access.
     * @param[in] length The length of the memory access.
     * @return 0 if success, otherwise the error code.
     */
    __device__ int32_t set_zero(const uint32_t index, const uint32_t length);
};
}  // namespace memory
namespace stack {
constexpr CONSTANT uint32_t max_size = CuEVM::max_stack_size; /**< The maximum stack size*/
struct evm_stack_t {
    evm_word_t *global_stack_base; /**< The stack YP: (YP: \f$\mu_{s}\f$)*/  // global memory store from X+1 element
    evm_word_t *shared_stack_base;  // shared memory for X elements from the top => becomes preallocated stackbase
    uint16_t stack_offset;          // offset of the current stack (its size) from it's base offset in shared memory
    uint32_t stack_base_offset;     // offset of the stack base in shared memory or global memory

    /**
     * The default constructor
     * Stack base offset of the child stack = parent stack offset  + 1
     */
    __device__ evm_stack_t(evm_word_t *shared_stack_base, uint32_t stack_base_offset = 0)
        : global_stack_base(nullptr),
          shared_stack_base(shared_stack_base),
          stack_base_offset(stack_base_offset),
          stack_offset(0) {
        // printf("stack constructor shared stack base %p, stack base offset %d, stack offset %d\n",
        // shared_stack_base,
        //        stack_base_offset, stack_offset);
    }
    __device__ evm_stack_t()
        : global_stack_base(nullptr), shared_stack_base(nullptr), stack_base_offset(0), stack_offset(0) {}
    __device__ void init(evm_word_t *shared_stack_base, uint32_t stack_base_offset = 0) {
        this->global_stack_base = nullptr;
        this->shared_stack_base = shared_stack_base;
        this->stack_base_offset = stack_base_offset;
        this->stack_offset = 0;
    }
    /**
     * The destructor
     */
    __host__ __device__ ~evm_stack_t();

    /**
     * The copy constructor
     * @param[in] other The other stack
     */
    // __host__ __device__ evm_stack_t(const evm_stack_t &other);

    /**
     * Free the memory
     */
    __host__ __device__ void free();

    /**
     * Clear the content
     */
    __host__ __device__ void clear();

    /**
     * Extract the stack data for tracing
     * @param[in] other The other stack
     */
    __host__ __device__ void extract_data(evm_word_t *other) const;

    /**
     * Get the size of the stack
     * @return The size of the stack
     */
    __host__ __device__ uint32_t size() const;

    /**
     * Reduce the size (remove items) of the stack without memory operations
     * @param[in] num_items The number of items to remove
     */
    __device__ void reduce_size(uint32_t num_items);

    /**
     * Get the top of the stack
     * @return The top of the stack pointer
     */
    __host__ __device__ evm_word_t *top();

    /**
     * Push a value to the stack
     * @param[in] arith The arithmetical environment
     * @param[in] value The value to be pushed
     * @return 0 if the value is pushed, error code otherwise
     */
    __host__ __device__ int32_t push(const evm_word_t &value);
    __host__ __device__ int32_t push_uint32(uint32_t value);
    __host__ __device__ int32_t push_uint64(uint64_t value);
    __host__ __device__ int32_t push_evm_word_t(const evm_word_t *value);
    /**
     * Pop a value from the stack
     * @param[in] arith The arithmetical environment
     * @param[out] y The value popped from the stack
     * @return 0 if the value is popped, error code otherwise
     */
    __host__ __device__ int32_t pop(evm_word_t &y);
    __host__ __device__ int32_t pop_evm_word(evm_word_t *&y);
    /**
     * Push a value to the stack from a byte array
     * @param[in] arith The arithmetical environment
     * @param[in] x the number of bytes of the value
     * @param[in] src_byte_data The source byte data
     * @param[in] src_byte_size The size of the source byte data
     * @return 0 if the value is pushed, error code otherwise
     */
    __host__ __device__ int32_t pushx(uint8_t x, const uint8_t __restrict__ *src_byte_data, uint8_t src_byte_size);

    /**
     * Get the value from the stack at the given index
     * @param[in] arith The arithmetical environment
     * @param[in] index The index of the value
     * @param[out] y The value at the given index
     * @return 0 if the value is popped, error code otherwise
     */
    __host__ __device__ int32_t get_index(uint32_t index, evm_word_t &y);

    __host__ __device__ evm_word_t *get_address_at_index(uint32_t index) const;
    /**
     * Duplicvate the value at the given index and push
     * it at the top of the stack.
     * @param[in] arith The arithmetical environment
     * @param[in] x The index of the value
     * @return 0 if the value is duplicated, error code otherwise
     */
    __host__ __device__ int32_t dupx(uint32_t x);

    /**
     * Swap the values at the given index with the top of the stack
     * @param[in] arith The arithmetical environment
     * @param[in] x The index of the value
     * @return 0 if the value is swapped, error code otherwise
     */
    __host__ __device__ int32_t swapx(uint32_t x);

    /**
     * Print the stack
     */
    __host__ __device__ void print() const;

    /**
     * Get the JSON object from the stack
     * @return The JSON object
     */
    __host__ cJSON *to_json();

    /**
     * Generate the stack gpu instances from the stack cpu instances
     * @param[in] cpu_instances The stack cpu instances
     * @param[in] count The number of instances
     * @return The stack gpu instances
     */
    __host__ static evm_stack_t *gpu_from_cpu(evm_stack_t *cpu_instances, uint32_t count);

    /**
     * Free the stack gpu instances
     * @param[in] gpu_instances The stack gpu instances
     * @param[in] count The number of instances
     */
    __host__ static void gpu_free(evm_stack_t *gpu_instances, uint32_t count);
};

}  // namespace stack

using evm_stack_t = stack::evm_stack_t;
using evm_memory_t = memory::evm_memory_t;

struct evm_call_context_t {
    bool static_env;                  /**< The static flag (STATICCALL) YP: \f$w\f$ */
    uint32_t depth;                   /**< The depth of the state */
    uint32_t pc;                      /**< The program counter */
    gas_t gas_used;                   /**< The gas */
    gas_t gas_refund;                 /**< The gas refund */
    gas_t gas_limit;                  /**< The gas limit */
    stack::evm_stack_t *stack_ptr;    /**< The stack */
    memory::evm_memory_t *memory_ptr; /**< The memory */

    evm_word_t from;
    evm_word_t to;

    evm_word_t storage_address;
    evm_word_t value;
    uint32_t call_type; /**< The call type internal has the opcode */
    uint8_t *call_data; /**< The data YP: \f$d\f$ */
    uint32_t call_data_size;
    uint8_t *byte_code; /**< The byte code YP: \f$b\f$ or \f$I_{b}\f$*/
    uint32_t byte_code_size;
    int32_t bytecode_offset;  // bytecode offset in the global `all_account_codes`

    uint32_t fixed_ret_size = 0;
    uint32_t fixed_ret_offset = 0;
    uint32_t dynamic_ret_size = 0;
    uint8_t *return_data = nullptr;  // == nullptr if return data size < preallocated global pool

    SnapshotState *snapshot_state = nullptr;
    evm_call_context_t *parent;
#ifdef EIP_3155
    uint32_t trace_idx; /**< The index in the trace */
#endif

    /**
     * The complete constructor of the evm_call_context_t
     */
    __device__ void initiate_values(uint32_t depth, gas_t gas_limit, CuEVM::evm_stack_t *stack_ptr,
                                    CuEVM::evm_memory_t *memory_ptr, evm_word_t from, evm_word_t to,
                                    evm_word_t storage_address, evm_word_t value, uint32_t call_type,
                                    uint8_t *call_data, uint32_t call_data_size, uint8_t *byte_code,
                                    uint32_t byte_code_size, int32_t bytecode_offset = -1,
                                    evm_call_context_t *parent = nullptr, bool static_env = false, gas_t gas_refund = 0

    );

    /**
     * The constructor with the parent's state
     **/
    __device__ void initiate_values(evm_call_context_t *parent, gas_t gas_limit, evm_word_t from, evm_word_t to,
                                    evm_word_t storage_address, evm_word_t value, uint32_t call_type,
                                    uint8_t *call_data, uint32_t call_data_size, uint8_t *byte_code,
                                    uint32_t byte_code_size, int32_t bytecode_offset = -1,
                                    uint32_t return_data_offset = 0, uint32_t return_data_size = 0,
                                    bool static_env = false, gas_t gas_refund = 0);

    __device__ void copy_return_data_to_memory(uint32_t memory_offset, uint32_t data_offset, uint32_t size);
    __device__ void copy_return_data(uint8_t *dest, uint32_t data_offset, uint32_t size);

    __device__ void set_parent_return_data(uint8_t *data, uint32_t size);
    __device__ void set_parent_return_data(uint32_t offset, uint32_t size);
    __device__ void print_return_data() const;
    __device__ evm_call_context_t() {};
    __device__ void clear();

    /**
     * The destructor of the evm_call_state_t
     */
    __device__ ~evm_call_context_t();

    __device__ void print() const;
    __device__ int32_t revert();
};

namespace memory_pool {
struct memory_pool_t {
    // preallocate data structures for call_depths
    evm_call_context_t *call_context;
    evm_stack_t *prealloc_stack_instances;
    evm_memory_t *prealloc_mem_instances;
    //
    evm_word_t *stack_base;
    uint8_t *return_data_base;
    uint32_t num_instances;
    uint16_t *snapshot_account_counts;  // store the current snapshot count for allocating new ones, per each instance
    uint16_t *snapshot_slot_counts;     // store the current snapshot slot count for allocating new ones, accross all
                                        // accoutns, per each instance
    CuEVM::SnapshotState *snapshot_states_pool;
    __host__ memory_pool_t() {};
};
extern __device__ memory_pool_t *global_memory_pool;
extern __device__ CuEVM::EccConstants *ecc_constants_ptr;
extern __device__ evm_word_t *preallocated_stack_base;
extern __device__ uint8_t *preallocated_return_data_base;
extern __device__ SnapshotValue *preallocated_snapshot_values;
extern __device__ ValueStatus **preallocated_snapshot_restore_ptr;  // store the original value to restore
extern __device__ uint8_t *preallocated_memory_base;

}  // namespace memory_pool

#ifdef EIP_3155
namespace utils {
extern __device__ char **global_trace_buffers;
extern __device__ size_t *global_trace_lengths;
}  // namespace utils
#endif
// typedef int32_t (*evm_operation_f)(CuEVM::evm_call_context_t* call_state);

/**
 * @brief Get the CPU EVM instances object
 * Get the evm instances from the json file
 * @param[in] test_json The json object
 * @param[out] evm_instances The evm instances
 * @param[out] num_instances The number of instances
 * @param[in] managed Whether the memory is managed
 * @return int32_t The error code, 0 if successful
 */
__host__ std::vector<CuEVM::transaction::TransactionList *> get_evm_instances(
    const cJSON *test_json, uint32_t &num_instances, uint32_t &num_account, uint32_t num_gpus = 1, uint32_t clones = 1);

__global__ void kernel_evm_multiple_instances(transaction::TransactionList *transaction_list_ptr, uint32_t count,
#ifdef EIP_3155
                                              char *d_buffer, size_t buffer_size,
#endif
                                              bool copy_state_data = true);
}  // namespace CuEVM
