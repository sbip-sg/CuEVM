
#include <CuEVM/core/evm_call_context.cuh>
#include <CuEVM/core/memory_pool.cuh>
#include <CuEVM/utils/error_codes.cuh>
#include <CuEVM/utils/opcodes.cuh>
namespace CuEVM {
__device__ cached_evm_call_context::cached_evm_call_context(evm_call_context_t* state) {  // copy from state to cache
    pc = state->pc;
    gas_used = state->gas_used;
    gas_limit = state->gas_limit;
    stack_ptr = state->stack_ptr;
    // printf("cached_evm_call_state stack ptr %p\n", stack_ptr);
    // printf("cached_evm_call_state stack ptr shared_stack_base %p\n", stack_ptr->shared_stack_base);
    // printf("cached_evm_call_state stack ptr stack_base_offset %d\n", stack_ptr->stack_base_offset);
    // printf("cached_evm_call_state stack ptr stack_offset %d\n", stack_ptr->stack_offset);
    // printf("message ptr %p\n", state->message_ptr);
    byte_code_size = state->byte_code_size;
    byte_code_data = state->byte_code;
}
__device__ void cached_evm_call_context::write_cache_to_state(evm_call_context_t* state) {
    state->pc = pc;
    state->gas_used = gas_used;
}  // copy from cache to state
__device__ void cached_evm_call_context::set_byte_code(const byte_array_t* byte_code) {
    byte_code_size = byte_code->size;
    byte_code_data = byte_code->data;
}
__device__ void cached_evm_call_context::set_byte_code(uint8_t* byte_code, uint32_t size) {
    byte_code_size = size;
    byte_code_data = byte_code;
}
__device__ void cached_evm_call_context::print() const {
    printf("Cached EVM Call State\n");
    printf("PC: %d\n", pc);
    printf("Gas Used: %lu\n", gas_used);
    printf("Gas Limit: %lu\n", gas_limit);
    printf("Byte Code Size: %d\n", byte_code_size);
    for (uint32_t i = 0; i < byte_code_size; i++) {
        printf("%02x", byte_code_data[i]);
    }
    printf("\n");
}

// __device__ evm_call_context_t::evm_call_context_t(CuEVM::evm_call_context_t* parent, uint32_t depth, uint32_t pc,
//                                                   gas_t gas_used, gas_t gas_refund, CuEVM::evm_stack_t* stack_ptr,
//                                                   CuEVM::evm_memory_t* memory_ptr,
//                                                   CuEVM::log_state_data_t* log_state_ptr) {
//     this->parent = parent;
//     this->depth = depth;
//     this->pc = pc;
//     this->gas_used = gas_used;
//     this->gas_refund = gas_refund;
//     this->stack_ptr = stack_ptr;
//     this->memory_ptr = memory_ptr;

//     // this->log_state_ptr = log_state_ptr;
//     // this->last_return_data_size = 0;
//     // this->last_return_data_offset = 0;
// #ifdef EIP_3155
//     this->trace_idx = 0;
//     // printf("evm_call_state_t constructor no parent %d\n", THREADIDX);
// #endif
// }

__device__ void evm_call_context_t::initiate_values(uint32_t depth, gas_t gas_limit, CuEVM::evm_stack_t* stack_ptr,
                                                    CuEVM::evm_memory_t* memory_ptr, evm_word_t from, evm_word_t to,
                                                    evm_word_t storage_address, evm_word_t value, uint32_t call_type,
                                                    uint8_t* call_data, uint32_t call_data_size, uint8_t* byte_code,
                                                    uint32_t byte_code_size, evm_call_context_t* parent,
                                                    CuEVM::jump_destinations_t* jump_destinations, bool static_env,
                                                    gas_t gas_refund
#ifdef EIP_3155
                                                    ,
                                                    uint32_t trace_idx
#endif
) {

    this->parent = parent;
    this->depth = depth;
    this->pc = pc;
    this->gas_used = gas_used;
    this->gas_refund = gas_refund;
    this->gas_limit = gas_limit;
    this->stack_ptr = stack_ptr;
    this->memory_ptr = memory_ptr;
    this->from = from;
    this->to = to;
    this->storage_address = storage_address;
    this->value = value;
    this->call_type = call_type;
    this->call_data = call_data;
    this->call_data_size = call_data_size;
    this->byte_code = byte_code;
    this->byte_code_size = byte_code_size;
    this->parent = parent;
    this->jump_destinations = jump_destinations;
    this->static_env = static_env;
    this->gas_refund = gas_refund;
#ifdef EIP_3155
    this->trace_idx = trace_idx;
#endif
}

/**
 * The constructor with the parent state and message call
 **/
__device__ void evm_call_context_t::initiate_values(evm_call_context_t* parent, gas_t gas_limit, evm_word_t from,
                                                    evm_word_t to, evm_word_t storage_address, evm_word_t value,
                                                    uint32_t call_type, uint8_t* call_data, uint32_t call_data_size,
                                                    uint8_t* byte_code, uint32_t byte_code_size,
                                                    evm_word_t* shared_stack_ptr,
                                                    CuEVM::jump_destinations_t* jump_destinations, bool static_env,
                                                    gas_t gas_refund
#ifdef EIP_3155
                                                    ,
                                                    uint32_t trace_idx
#endif
) {
    // printf("evm_call_state_t constructor with parent %d\n", THREADIDX);
    if (parent == nullptr) {
        printf("parent is nullptr\n");
        return;
    }
    this->parent = parent;
    this->depth = parent->depth + 1;
    this->pc = 0;
    this->gas_used = 0;
    this->gas_refund = parent->gas_refund;

    if (parent->stack_ptr != nullptr) {
        this->stack_ptr =
            new CuEVM::evm_stack_t(parent->stack_ptr->shared_stack_base +
                                       (parent->stack_ptr->stack_base_offset + parent->stack_ptr->stack_offset) *
                                           CuEVM::memory_pool::global_memory_pool->num_instances,
                                   parent->stack_ptr->stack_base_offset + parent->stack_ptr->stack_offset);
        // printf("parent stack found %p thread %d\n", parent->stack_ptr, THREADIDX);
    } else {
        this->stack_ptr = new CuEVM::evm_stack_t(shared_stack_ptr);
        // printf("parent stack not found %p thread %d\n", shared_stack_ptr, THREADIDX);
    }

    this->memory_ptr = new CuEVM::evm_memory_t();

#ifdef EIP_3155
    this->trace_idx = 0;
#endif
    // printf("evm_call_state_t constructor with parent %d\n", THREADIDX);
}

/**
 * The destructor of the evm_call_state_t
 */
__device__ evm_call_context_t::~evm_call_context_t() {
    if (parent != nullptr) {
        // delete stack_ptr;
        // delete memory_ptr;
        // delete last_return_data_ptr;
        // TODO delete touch_state_ptr;
    }
}
__device__ void evm_call_context_t::print() const {
    printf("EVM Call State\n");
    printf("Depth: %d\n", depth);
    printf("PC: %d\n", pc);
    printf("Gas Used: %lu\n", gas_used);
    printf("Gas Refund: %lu\n", gas_refund);
    printf("Gas Limit: %lu\n", gas_limit);
    printf("From: ");
    from.print();
    printf("\nTo: ");
    to.print();
    printf("\nValue: ");
    value.print();
    printf("\nStorage Address: ");
    storage_address.print();
    printf("\nCall Type: %d\n", call_type);
    printf("Call Data Size: %d\n", call_data_size);
    for (uint32_t i = 0; i < call_data_size; i++) {
        printf("%02x", call_data[i]);
    }
    printf("\nByte Code Size: %d\n", byte_code_size);
    for (uint32_t i = 0; i < byte_code_size; i++) {
        printf("%02x", byte_code[i]);
    }
    // printf("\nReturn Data Offset: ");
    // return_data_offset.print();
    // printf("\nReturn Data Size: %d\n", return_data_size);
    printf("Static Env: %d\n", static_env);
}

__device__ int32_t evm_call_context_t::revert() {}

}  // namespace CuEVM