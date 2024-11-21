
#include <CuEVM/evm_call_state.cuh>
#include <CuEVM/utils/error_codes.cuh>
#include <CuEVM/utils/opcodes.cuh>

namespace CuEVM {
__host__ __device__ cached_evm_call_state::cached_evm_call_state(ArithEnv& arith,
                                                                 evm_call_state_t* state) {  // copy from state to cache
    pc = state->pc;
    cgbn_set(arith.env, gas_used, state->gas_used);
    cgbn_set(arith.env, gas_limit, state->gas_limit);
    stack_ptr = state->stack_ptr;
    byte_code_size = state->message_ptr->byte_code->size;
    byte_code_data = state->message_ptr->byte_code->data;
}
__host__ __device__ void cached_evm_call_state::write_cache_to_state(ArithEnv& arith, evm_call_state_t* state) {
    state->pc = pc;
    cgbn_set(arith.env, state->gas_used, gas_used);
}  // copy from cache to state
__host__ __device__ void cached_evm_call_state::set_byte_code(const byte_array_t* byte_code) {
    byte_code_size = byte_code->size;
    byte_code_data = byte_code->data;
}
__host__ __device__ evm_call_state_t::evm_call_state_t(ArithEnv& arith, CuEVM::evm_call_state_t* parent, uint32_t depth,
                                                       uint32_t pc, bn_t gas_used, bn_t gas_refund,
                                                       CuEVM::evm_message_call_t* message_ptr,
                                                       CuEVM::evm_stack_t* stack_ptr, CuEVM::evm_memory_t* memory_ptr,
                                                       CuEVM::log_state_data_t* log_state_ptr,
                                                       CuEVM::TouchState* touch_state_ptr,
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
    this->touch_state_ptr = touch_state_ptr;
    this->last_return_data_ptr = last_return_data_ptr;
#ifdef EIP_3155
    this->trace_idx = 0;
    printf("evm_call_state_t constructor no parent %d\n", THREADIDX);
#endif
}

// /**
//  * The constructor with the parent state and message call
//  */
// __host__ __device__ evm_call_state_t::evm_call_state_t(ArithEnv& arith, CuEVM::evm_call_state_t* parent,
//                                                        CuEVM::evm_message_call_t* message_ptr)
//     : touch_state(new CuEVM::state_access_t(), &parent->touch_state) {
//     this->parent = parent;
//     this->depth = parent->depth + 1;
//     this->pc = 0;
//     cgbn_set_ui32(arith.env, this->gas_used, 0);
//     cgbn_set(arith.env, this->gas_refund, parent->gas_refund);
//     this->message_ptr = message_ptr;
//     this->message_ptr->get_gas_limit(arith, this->gas_limit);
//     this->stack_ptr = new CuEVM::evm_stack_t();
//     this->memory_ptr = new CuEVM::evm_memory_t();
//     this->log_state_ptr = new CuEVM::log_state_data_t();
//     this->last_return_data_ptr = new CuEVM::evm_return_data_t();
// #ifdef EIP_3155
//     this->trace_idx = 0;
// #endif
// }

/**
 * The constructor with the parent state and message call
 */
__host__ __device__ evm_call_state_t::evm_call_state_t(ArithEnv& arith, CuEVM::evm_call_state_t* parent,
                                                       CuEVM::evm_message_call_t* shared_message_ptr,
                                                       CuEVM::evm_message_call_t_shadow* shadow_message_ptr,
                                                       evm_word_t* shared_stack_ptr)
// : touch_state(new CuEVM::state_access_t(), &parent->touch_state) {
{
    // printf("evm_call_state_t constructor with parent %d\n", THREADIDX);
    __SHARED_MEMORY__ CuEVM::TouchState* touch_state_ptr[CGBN_IBP];
    __ONE_GPU_THREAD_WOSYNC_BEGIN__
    touch_state_ptr[INSTANCE_IDX_PER_BLOCK] =
        new CuEVM::TouchState(new CuEVM::state_access_t(), parent->touch_state_ptr);
    __ONE_GPU_THREAD_END__
    this->touch_state_ptr = touch_state_ptr[INSTANCE_IDX_PER_BLOCK];
    this->parent = parent;
    this->depth = parent->depth + 1;
    this->pc = 0;
    cgbn_set_ui32(arith.env, this->gas_used, 0);
    cgbn_set(arith.env, this->gas_refund, parent->gas_refund);
    this->message_ptr = shared_message_ptr;
    this->message_ptr_copy = shadow_message_ptr;  // point to global memory, deallocate in destructor
    this->message_ptr->get_gas_limit(arith, this->gas_limit);
    __SHARED_MEMORY__ evm_stack_t* stack_ptr[CGBN_IBP];
    __ONE_GPU_THREAD_WOSYNC_BEGIN__
    if (parent->stack_ptr != nullptr) {
        stack_ptr[INSTANCE_IDX_PER_BLOCK] =
            new CuEVM::evm_stack_t(parent->stack_ptr->shared_stack_base + parent->stack_ptr->stack_offset,
                                   parent->stack_ptr->stack_base_offset + parent->stack_ptr->stack_offset);
        // printf("parent stack found %p thread %d\n", parent->stack_ptr, THREADIDX);
    } else {
        stack_ptr[INSTANCE_IDX_PER_BLOCK] = new CuEVM::evm_stack_t(shared_stack_ptr);
        // printf("parent stack not found %p thread %d\n", parent->stack_ptr, THREADIDX);
    }
    // printf("Stack constructor done stack base %p stack base offset %d stack offset %d\n",
    // stack_ptr->shared_stack_base,
    //        stack_ptr->stack_base_offset, stack_ptr->stack_offset);
    // stack_ptr = new CuEVM::evm_stack_t(shared_stack_ptr);
    __ONE_GPU_THREAD_END__
    this->stack_ptr = stack_ptr[INSTANCE_IDX_PER_BLOCK];

    // printf("evm_call_state_t initialized stack pointer %p thread %d\n", stack_ptr, THREADIDX);

    __SHARED_MEMORY__ evm_memory_t* memory_ptr[CGBN_IBP];
    __ONE_GPU_THREAD_WOSYNC_BEGIN__
    memory_ptr[INSTANCE_IDX_PER_BLOCK] = new CuEVM::evm_memory_t();
    __ONE_GPU_THREAD_END__

    printf("evm_call_state_t initialized memory pointer %p thread %d\n", memory_ptr, THREADIDX);

    this->memory_ptr = memory_ptr[INSTANCE_IDX_PER_BLOCK];
    // printf("memory size %d  idx %d\n", memory_ptr->size, THREADIDX);
    __SHARED_MEMORY__ log_state_data_t* log_state_ptr[CGBN_IBP];
    __ONE_GPU_THREAD_WOSYNC_BEGIN__
    log_state_ptr[INSTANCE_IDX_PER_BLOCK] = new CuEVM::log_state_data_t();
    __ONE_GPU_THREAD_END__
    this->log_state_ptr = log_state_ptr[INSTANCE_IDX_PER_BLOCK];

    // printf("evm_call_state_t initialized log state pointer %p thread %d\n", log_state_ptr, THREADIDX);

    __SHARED_MEMORY__ evm_return_data_t* last_return_data_ptr[CGBN_IBP];
    __ONE_GPU_THREAD_WOSYNC_BEGIN__
    last_return_data_ptr[INSTANCE_IDX_PER_BLOCK] = new CuEVM::evm_return_data_t();
    __ONE_GPU_THREAD_END__
    this->last_return_data_ptr = last_return_data_ptr[INSTANCE_IDX_PER_BLOCK];

    // printf("evm_call_state_t initialized return data pointer %p thread %d\n", last_return_data_ptr, THREADIDX);
    // this->memory_ptr = new CuEVM::evm_memory_t();
    // this->log_state_ptr = new CuEVM::log_state_data_t();
    // this->last_return_data_ptr = new CuEVM::evm_return_data_t();
#ifdef EIP_3155
    this->trace_idx = 0;
#endif
    // printf("evm_call_state_t constructor with parent %d\n", THREADIDX);
}
__host__ __device__ evm_call_state_t::evm_call_state_t(ArithEnv& arith, CuEVM::evm_call_state_t* other) {
    this->parent = other->parent;
    this->depth = other->depth;
    this->pc = other->pc;
    cgbn_set(arith.env, this->gas_used, other->gas_used);
    cgbn_set(arith.env, this->gas_refund, other->gas_refund);
    this->message_ptr = other->message_ptr;
    this->message_ptr_copy = other->message_ptr_copy;
    this->stack_ptr = other->stack_ptr;
    this->memory_ptr = other->memory_ptr;
    this->log_state_ptr = other->log_state_ptr;
    this->touch_state_ptr = other->touch_state_ptr;
    this->last_return_data_ptr = other->last_return_data_ptr;
}

/**
 * The constructor with no parent state and message call
 */
__host__ __device__ evm_call_state_t::evm_call_state_t(ArithEnv& arith, CuEVM::WorldState* word_state_ptr,
                                                       CuEVM::evm_stack_t* stack_ptr, CuEVM::evm_memory_t* memory_ptr,
                                                       CuEVM::log_state_data_t* log_state_ptr,
                                                       CuEVM::state_access_t* state_access_ptr,
                                                       CuEVM::evm_return_data_t* last_return_data_ptr)
// : touch_state(state_access_ptr, word_state_ptr) {
{
    __SHARED_MEMORY__ CuEVM::TouchState* touch_state_ptr[CGBN_IBP];
    __ONE_GPU_THREAD_WOSYNC_BEGIN__
    touch_state_ptr[INSTANCE_IDX_PER_BLOCK] = new CuEVM::TouchState(state_access_ptr, word_state_ptr);
    __ONE_GPU_THREAD_END__
    this->touch_state_ptr = touch_state_ptr[INSTANCE_IDX_PER_BLOCK];
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
        // delete message_ptr; TODO: fix this, currently using shared mem for message_ptr
        // printf("evm_call_state_t destructor thread %d\n", THREADIDX);
        // __ONE_GPU_THREAD_WOSYNC_BEGIN__
        // printf(
        //     "evm_call_state_t destructor thread %d message ptr %p stack ptr %p memory ptr %p log state ptr %p last "
        //     "return data ptr %p\n",
        //     THREADIDX, message_ptr, stack_ptr, memory_ptr, log_state_ptr, last_return_data_ptr);
        // __ONE_GPU_THREAD_WOSYNC_END__
        __ONE_GPU_THREAD_WOSYNC_BEGIN__
        delete message_ptr_copy;
        delete stack_ptr;
        delete memory_ptr;
        delete log_state_ptr;
        delete last_return_data_ptr;
        // TODO delete touch_state_ptr;
        __ONE_GPU_THREAD_WOSYNC_END__
        // printf("evm_call_state_t done destructor thread %d\n", THREADIDX);
    }
}

__host__ __device__ void evm_call_state_t::print(ArithEnv& arith) const {
    __ONE_GPU_THREAD_WOSYNC_BEGIN__
    printf("EVM Call State\n");
    printf("Depth: %d\n", depth);
    printf("PC: %d\n", pc);
    printf("Gas Used: \n");
    printf("Gas Refund: \n");
    printf("Gas Limit: \n");
    __ONE_GPU_THREAD_WOSYNC_END__
    print_bnt(arith, gas_used);
    print_bnt(arith, gas_refund);
    print_bnt(arith, gas_limit);

    __ONE_GPU_THREAD_WOSYNC_BEGIN__
    printf("Message pointer: %p\n", message_ptr);
    printf("Stack pointer: %p\n", stack_ptr);
    printf("Memory pointer: %p\n", memory_ptr);
    printf("Log state pointer: %p\n", log_state_ptr);
    // printf("Touch state\n");
    // touch_state.print();
    printf("Last return data pointer: %p\n", last_return_data_ptr);
    printf("\n");
    __ONE_GPU_THREAD_WOSYNC_END__
}

__host__ __device__ int32_t evm_call_state_t::update(ArithEnv& arith, evm_call_state_t& other) {
    uint32_t error_code = ERROR_SUCCESS;
    // printf("\n\ntouch state update \n");
    // printf("this touch state \n");
    // this->touch_state.print();
    // // printf("\n------------------\n\n");
    // printf("other touch state \n");
    // other.touch_state.print();
    // printf("\n------------------\n\n");

    error_code |= this->touch_state_ptr->update(arith, other.touch_state_ptr);
    // printf("touch state update done \n");
    // printf("this touch state \n");
    // this->touch_state.print();
    // printf("\n------------------\n\n");
    // printf("end touch state update\n\n");
#ifdef ENABLE_LOGS
    error_code |= this->log_state_ptr->update(arith, *other.log_state_ptr);
#endif
    return error_code;
}

}  // namespace CuEVM