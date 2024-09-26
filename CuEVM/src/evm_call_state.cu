
#include <CuEVM/evm_call_state.cuh>
#include <CuEVM/utils/error_codes.cuh>
#include <CuEVM/utils/opcodes.cuh>

namespace CuEVM {
__host__ __device__ evm_call_state_t::evm_call_state_t(
    ArithEnv& arith, CuEVM::evm_call_state_t* parent, uint32_t depth,
    uint32_t pc, bn_t gas_used, bn_t gas_refund,
    CuEVM::evm_message_call_t* message_ptr, CuEVM::evm_stack_t* stack_ptr,
    CuEVM::evm_memory_t* memory_ptr, CuEVM::log_state_data_t* log_state_ptr,
    CuEVM::TouchState touch_state,
    CuEVM::evm_return_data_t* last_return_data_ptr) {
    this->parent = parent;
    this->depth = depth;
    this->pc = pc;
    cgbn_set(arith.env, this->gas_used, gas_used);
    cgbn_set(arith.env, this->gas_refund, gas_refund);
    this->message_ptr = message_ptr;
    this->message_ptr->get_gas_limit(arith, this->gas_limit);
    this->stack_ptr = stack_ptr;
    this->memory_ptr = memory_ptr;
    this->log_state_ptr = log_state_ptr;
    this->touch_state = touch_state;
    this->last_return_data_ptr = last_return_data_ptr;
#ifdef EIP_3155
    this->trace_idx = 0;
#endif
}

/**
 * The constructor with the parent state and message call
 */
__host__ __device__ evm_call_state_t::evm_call_state_t(
    ArithEnv& arith, CuEVM::evm_call_state_t* parent,
    CuEVM::evm_message_call_t* message_ptr)
    : touch_state(new CuEVM::state_access_t(), &parent->touch_state) {
    this->parent = parent;
    this->depth = parent->depth + 1;
    this->pc = 0;
    cgbn_set_ui32(arith.env, this->gas_used, 0);
    cgbn_set(arith.env, this->gas_refund, parent->gas_refund);
    this->message_ptr = message_ptr;
    this->message_ptr->get_gas_limit(arith, this->gas_limit);
    this->stack_ptr = new CuEVM::evm_stack_t();
    this->memory_ptr = new CuEVM::evm_memory_t();
    this->log_state_ptr = new CuEVM::log_state_data_t();
    this->last_return_data_ptr = new CuEVM::evm_return_data_t();
#ifdef EIP_3155
    this->trace_idx = 0;
#endif
}

/**
 * The constructor with no parent state and message call
 */
__host__ __device__ evm_call_state_t::evm_call_state_t(
    ArithEnv& arith, CuEVM::WorldState* word_state_ptr,
    CuEVM::evm_stack_t* stack_ptr, CuEVM::evm_memory_t* memory_ptr,
    CuEVM::log_state_data_t* log_state_ptr,
    CuEVM::state_access_t* state_access_ptr,
    CuEVM::evm_return_data_t* last_return_data_ptr)
    : touch_state(state_access_ptr, word_state_ptr) {
    this->parent = nullptr;
    this->depth = 0;
    this->pc = 0;
    cgbn_set_ui32(arith.env, this->gas_used, 0);
    cgbn_set_ui32(arith.env, this->gas_refund, 0);
    this->message_ptr = nullptr;
    cgbn_set_ui32(arith.env, this->gas_limit, 0);
    this->stack_ptr = stack_ptr;
    this->memory_ptr = memory_ptr;
    this->log_state_ptr = log_state_ptr;
    this->last_return_data_ptr = last_return_data_ptr;
#ifdef EIP_3155
    this->trace_idx = 0;
#endif
}

/**
 * The destructor of the evm_call_state_t
 */
__host__ __device__ evm_call_state_t::~evm_call_state_t() {
    if (parent != nullptr) {
        delete message_ptr;
        delete stack_ptr;
        delete memory_ptr;
        delete log_state_ptr;
        delete last_return_data_ptr;
    }
}

__host__ __device__ int32_t evm_call_state_t::update(ArithEnv& arith,
                                                     evm_call_state_t& other) {
    uint32_t error_code = ERROR_SUCCESS;
    // printf("\n\ntouch state update \n");
    // printf("this touch state \n");
    // this->touch_state.print();
    // printf("other touch state \n");
    // other.touch_state.print();
    error_code |= this->touch_state.update(arith, &other.touch_state);
    // printf("touch state update done \n");
    // printf("this touch state \n");
    // this->touch_state.print();
    // printf("end touch state update\n\n");
    error_code |= this->log_state_ptr->update(arith, *other.log_state_ptr);
    return error_code;
}

}  // namespace CuEVM