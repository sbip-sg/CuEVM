
#include <CuEVM/gas_cost.cuh>
#include <CuEVM/operations/system.cuh>
#include <CuEVM/utils/error_codes.cuh>
#include <CuEVM/utils/evm_utils.cuh>
#include <CuEVM/utils/opcodes.cuh>

namespace CuEVM::operations {
/**
 * Make a generic call.
 * @param[in] arith The arithmetical environment.
 * @param[in] current_state The current state.
 * @param[out] new_state_ptr The new state pointer.
 * @return 0 if the operation is successful, otherwise the error code.
 */
__host__ __device__ int32_t generic_CALL(ArithEnv &arith, const bn_t &args_offset, const bn_t &args_size,
                                         CuEVM::evm_call_state_t &current_state,
                                         CuEVM::evm_call_state_t *&new_state_ptr) {
    // try to send value in call
    bn_t value;
    new_state_ptr->message_ptr->get_value(arith, value);
    int32_t error_code =
        ((new_state_ptr->message_ptr->get_static_env() && (cgbn_compare_ui32(arith.env, value, 0) != 0) &&
          (new_state_ptr->message_ptr->get_call_type() == OP_CALL))
             ? ERROR_STATIC_CALL_CONTEXT_CALL_VALUE
             : ERROR_SUCCESS);

    // charge the gas for the call
    // #ifdef __CUDA_ARCH__
    //     printf("Before memory_grow_cost %d\n", threadIdx.x);
    //     __SYNC_THREADS__
    //     print_bnt(arith, current_state.gas_used);
    // #endif
    // memory call data
    bn_t memory_expansion_cost_args;

    // replace gas_used, throw away after the call
    // because we did not increase_memory_cost between expansions
    bn_t temp_memory_gas_used;
    // reset to 0;
    cgbn_set_ui32(arith.env, temp_memory_gas_used, 0);

    error_code |= CuEVM::gas_cost::memory_grow_cost(arith, *current_state.memory_ptr, args_offset, args_size,
                                                    memory_expansion_cost_args, temp_memory_gas_used);
    // memory return data
    bn_t ret_offset, ret_size;
    new_state_ptr->message_ptr->get_return_data_offset(arith, ret_offset);
    new_state_ptr->message_ptr->get_return_data_size(arith, ret_size);
    bn_t memory_expansion_cost_ret;
    error_code |= CuEVM::gas_cost::memory_grow_cost(arith, *current_state.memory_ptr, ret_offset, ret_size,
                                                    memory_expansion_cost_ret, temp_memory_gas_used);

    // compute the total memory expansion cost
    bn_t memory_expansion_cost;
    if (cgbn_compare(arith.env, memory_expansion_cost_args, memory_expansion_cost_ret) > 0) {
        cgbn_set(arith.env, memory_expansion_cost, memory_expansion_cost_args);
    } else {
        cgbn_set(arith.env, memory_expansion_cost, memory_expansion_cost_ret);
    }
    cgbn_add(arith.env, current_state.gas_used, current_state.gas_used, memory_expansion_cost);

    // #ifdef __CUDA_ARCH__
    //     printf("after memory_grow_cost %d\n", threadIdx.x);
    //     __SYNC_THREADS__
    //     print_bnt(arith, current_state.gas_used);
    // #endif

    // adress warm call
    bn_t contract_address;
    new_state_ptr->message_ptr->get_contract_address(arith, contract_address);
    CuEVM::gas_cost::access_account_cost(arith, current_state.gas_used, current_state.touch_state, contract_address);
    // positive value call cost (except delegate call)
    // empty account call cost
    // #ifdef __CUDA_ARCH__
    //     printf("After access_account_cost cost %d\n", threadIdx.x);
    //     __SYNC_THREADS__
    //     print_bnt(arith, current_state.gas_used);
    // #endif

    bn_t gas_stippend;
    cgbn_set_ui32(arith.env, gas_stippend, 0);
    if (new_state_ptr->message_ptr->get_call_type() != OP_DELEGATECALL) {
        if (cgbn_compare_ui32(arith.env, value, 0) > 0) {
            cgbn_add_ui32(arith.env, current_state.gas_used, current_state.gas_used, GAS_CALL_VALUE);
            cgbn_set_ui32(arith.env, gas_stippend, GAS_CALL_STIPEND);
            // If the empty account is called
            // only for call opcode
            if ((new_state_ptr->touch_state.is_empty_account(arith, contract_address)) &&
                (new_state_ptr->message_ptr->get_call_type() == OP_CALL)) {
                cgbn_add_ui32(arith.env, current_state.gas_used, current_state.gas_used, GAS_NEW_ACCOUNT);
            };
        }
    }
    // max gas call, gas_sent_with_call
    bn_t gas_capped;
    CuEVM::gas_cost::max_gas_call(arith, gas_capped, current_state.gas_limit, current_state.gas_used);

    // limit the gas to the gas capped
    if (cgbn_compare(arith.env, new_state_ptr->gas_limit, gas_capped) > 0) {
        cgbn_set(arith.env, new_state_ptr->gas_limit, gas_capped);
    }
    // add the the gas sent to the gas used
    cgbn_add(arith.env, current_state.gas_used, current_state.gas_used, new_state_ptr->gas_limit);

    // Gas stipen 2300 is added to the total gas limit but not gas used
    // add the gas stippend to gas limit of the child call
    cgbn_add(arith.env, new_state_ptr->gas_limit, new_state_ptr->gas_limit, gas_stippend);

    error_code |= CuEVM::gas_cost::has_gas(arith, current_state.gas_limit, current_state.gas_used);

    if (error_code == ERROR_SUCCESS) {
        // increase the memory cost
        current_state.memory_ptr->increase_memory_cost(arith, memory_expansion_cost);
        // set the byte code
        // FIX: MAke the warm up later for the contract in START_CALL
        // CuEVM::account_t *contract=nullptr;
        // error_code |= access_state.get_account(arith, contract_address,
        // contract, ACCOUNT_NONE_FLAG);
        // new_state_ptr->message_ptr->set_byte_code(
        //     contract->byte_code);

        // get/set the call data
        error_code |= current_state.memory_ptr->get(arith, args_offset, args_size, new_state_ptr->message_ptr->data);
    }
    return error_code;
}

/**
 * Make a generic create call.
 * @param[in] arith The arithmetical environment.
 * @param[in] current_state The current state.
 * @param[out] new_state_ptr The new state pointer.
 * @return 0 if the operation is successful, otherwise the error code.
 */
__host__ __device__ int32_t generic_CREATE(ArithEnv &arith, CuEVM::evm_call_state_t &current_state,
                                           CuEVM::evm_call_state_t *&new_state_ptr, const uint32_t opcode) {
    bn_t value, memory_offset, length;
    int32_t error_code = current_state.stack_ptr->pop(arith, value);
    error_code |= current_state.stack_ptr->pop(arith, memory_offset);
    error_code |= current_state.stack_ptr->pop(arith, length);
    // create cost
    cgbn_add_ui32(arith.env, current_state.gas_used, current_state.gas_used, GAS_CREATE);

    // compute the memory cost
    bn_t memory_expansion_cost;
    error_code |= CuEVM::gas_cost::memory_grow_cost(arith, *current_state.memory_ptr, memory_offset, length,
                                                    memory_expansion_cost, current_state.gas_used);

    // compute the initcode gas cost
    CuEVM::gas_cost::initcode_cost(arith, current_state.gas_used, length);

    bn_t salt;
    if (opcode == OP_CREATE2) {
        error_code |= current_state.stack_ptr->pop(arith, salt);
        // compute the keccak gas cost
        CuEVM::gas_cost::keccak_cost(arith, current_state.gas_used, length);
    }
    // #ifdef __CUDA_ARCH__
    //     printf("Before has_gas %d error code %d\n", threadIdx.x, error_code);
    //     print_bnt(arith, current_state.gas_limit);
    //     print_bnt(arith, current_state.gas_used);
    // #endif

    error_code |= CuEVM::gas_cost::has_gas(arith, current_state.gas_limit, current_state.gas_used);

    if (error_code == ERROR_SUCCESS) {
        // increase the memory cost
        current_state.memory_ptr->increase_memory_cost(arith, memory_expansion_cost);
        // #ifdef __CUDA_ARCH__
        //         printf("loading initialisation_code %d:\n", threadIdx.x);
        //         print_bnt(arith, memory_offset);
        //         print_bnt(arith, length);
        //         current_state.memory_ptr->print();
        // #endif
        // get the initialisation code
        CuEVM::byte_array_t initialisation_code;
        current_state.memory_ptr->get(arith, memory_offset, length, initialisation_code);
#ifdef __CUDA_ARCH__
        printf("initialisation_code %d:\n", threadIdx.x);
        initialisation_code.print();
#endif
        bn_t sender_address;
        current_state.message_ptr->get_recipient(arith, sender_address);
        bn_t contract_address;

        // // warm up the contract address
        // error_code |=
        //     current_state.touch_state.set_warm_account(arith,
        //     contract_address);

        CuEVM::account_t *sender_account = nullptr;
        current_state.touch_state.get_account(arith, sender_address, sender_account, ACCOUNT_NON_STORAGE_FLAG);
        // Do not get_account after this to reuse sender_account
        if (opcode == OP_CREATE2) {
            error_code |= CuEVM::utils::get_contract_address_create2(arith, contract_address, sender_address, salt,
                                                                     initialisation_code);
        } else {
            bn_t sender_nonce;
            sender_account->get_nonce(arith, sender_nonce);
            error_code |=
                CuEVM::utils::get_contract_address_create(arith, contract_address, sender_address, sender_nonce);
        }
        if (!current_state.touch_state.is_empty_account_create(arith, contract_address)) {
            // corner collision case: must set warm for the contract address
            current_state.touch_state.set_warm_account(arith, contract_address);
            error_code |= ERROR_MESSAGE_CALL_CREATE_CONTRACT_EXISTS;
        }
        // gas capped limit
        bn_t gas_capped;
        CuEVM::gas_cost::max_gas_call(arith, gas_capped, current_state.gas_limit, current_state.gas_used);
        // add the gas sent to the gas used
        cgbn_add(arith.env, current_state.gas_used, current_state.gas_used, gas_capped);
        // the return data offset and size
        bn_t ret_offset, ret_size;
        cgbn_set_ui32(arith.env, ret_offset, 0);
        cgbn_set_ui32(arith.env, ret_size, 0);
        CuEVM::byte_array_t call_data;
        // create the new evm call state
        new_state_ptr = new CuEVM::evm_call_state_t(
            arith, &current_state,
            new CuEVM::evm_message_call_t(arith, sender_address, contract_address, contract_address, gas_capped, value,
                                          current_state.message_ptr->get_depth() + 1, opcode, contract_address,
                                          call_data, initialisation_code, ret_offset, ret_size,
                                          current_state.message_ptr->get_static_env()));
        error_code |=
            (current_state.message_ptr->get_static_env() ? ERROR_STATIC_CALL_CONTEXT_CREATE :
#ifdef EIP_3860
                                                         (cgbn_compare_ui32(arith.env, length, max_initcode_size) > 0
                                                              ? ERROR_CREATE_INIT_CODE_SIZE_EXCEEDED
                                                              : ERROR_SUCCESS)
#else
                                                         ERROR_SUCCESS
#endif
            );

        if (sender_account->is_contract()) {
            bn_t sender_nonce;
            sender_account->get_nonce(arith, sender_nonce);
            cgbn_add_ui32(arith.env, sender_nonce, sender_nonce, 1);
            sender_account->set_nonce(arith, sender_nonce);
            // propagate to child state
            new_state_ptr->touch_state.set_nonce(arith, sender_address, sender_nonce);
        }
    } else
        cgbn_bitwise_mask_and(arith.env, current_state.gas_used, current_state.gas_used, 64);

    // printf("generic_CREATE error_code: %d\n", error_code);
    return error_code;
}

/**
 * The STOP operation.
 * @param[out] return_data The return data.
 * @return return error code.
 */
__host__ __device__ int32_t STOP(CuEVM::evm_return_data_t &return_data) {
    return_data = CuEVM::evm_return_data_t();
    return ERROR_RETURN;
}

/**
 * The CREATE operation. gives the new evm call state
 * @param[in] arith The arithmetical environment.
 * @param[in] current_state The current state.
 * @param[out] new_state_ptr The new state pointer.
 * @return 0 if the operation is successful, otherwise the error code.
 */
__host__ __device__ int32_t CREATE(ArithEnv &arith, CuEVM::evm_call_state_t &current_state,
                                   CuEVM::evm_call_state_t *&new_state_ptr) {
    return generic_CREATE(arith, current_state, new_state_ptr, OP_CREATE);
}

/**
 * The CALL operation. gives the new evm call state
 * @param[in] arith The arithmetical environment.
 * @param[in] current_state The current state.
 * @param[out] new_state_ptr The new state pointer.
 * @return 0 if the operation is successful, otherwise the error code.
 */
__host__ __device__ int32_t CALL(ArithEnv &arith, CuEVM::evm_call_state_t &current_state,
                                 CuEVM::evm_call_state_t *&new_state_ptr) {
    bn_t gas, address, value, args_offset, args_size, ret_offset, ret_size;
#ifdef __CUDA_ARCH__
    printf("opcode CALL %d\n", threadIdx.x);
#endif
    int32_t error_code = current_state.stack_ptr->pop(arith, gas);
    error_code |= current_state.stack_ptr->pop(arith, address);
    error_code |= current_state.stack_ptr->pop(arith, value);
    error_code |= current_state.stack_ptr->pop(arith, args_offset);
    error_code |= current_state.stack_ptr->pop(arith, args_size);
    error_code |= current_state.stack_ptr->pop(arith, ret_offset);
    error_code |= current_state.stack_ptr->pop(arith, ret_size);
#ifdef __CUDA_ARCH__
    printf("opcode CALL before error_code == ERROR_SUCCESS %d\n", threadIdx.x);
#endif
    if (error_code == ERROR_SUCCESS) {
        // clean the address
        CuEVM::evm_address_conversion(arith, address);
        bn_t sender;
        current_state.message_ptr->get_recipient(arith, sender);  // I_{a}
        bn_t recipient;
        cgbn_set(arith.env, recipient, address);  // t
        bn_t contract_address;
        cgbn_set(arith.env, contract_address, address);  // t
        bn_t storage_address;
        cgbn_set(arith.env, storage_address, address);  // t
        CuEVM::byte_array_t call_data;
        CuEVM::byte_array_t code;
        new_state_ptr = new CuEVM::evm_call_state_t(
            arith, &current_state,
            new CuEVM::evm_message_call_t(arith, sender, recipient, contract_address, gas, value,
                                          current_state.message_ptr->get_depth() + 1, OP_CALL, storage_address,
                                          call_data, code, ret_offset, ret_size,
                                          current_state.message_ptr->get_static_env()));
        // #ifdef __CUDA_ARCH__
        // printf("opcode CALL after constructing message call t  %d\n", threadIdx.x);
        // #endif
        error_code |= generic_CALL(arith, args_offset, args_size, current_state, new_state_ptr);
    }

#ifdef __CUDA_ARCH__
    printf("opcode CALL after cgeneric_CALL %d\n", threadIdx.x);
#endif
    return error_code;
}

/**
 * The CALLCODE operation. gives the new evm call state
 * @param[in] arith The arithmetical environment.
 * @param[in] current_state The current state.
 * @param[out] new_state_ptr The new state pointer.
 * @return 0 if the operation is successful, otherwise the error code.
 */
__host__ __device__ int32_t CALLCODE(ArithEnv &arith, CuEVM::evm_call_state_t &current_state,
                                     CuEVM::evm_call_state_t *&new_state_ptr) {
    bn_t gas, address, value, args_offset, args_size, ret_offset, ret_size;
    int32_t error_code = current_state.stack_ptr->pop(arith, gas);
    error_code |= current_state.stack_ptr->pop(arith, address);
    error_code |= current_state.stack_ptr->pop(arith, value);
    error_code |= current_state.stack_ptr->pop(arith, args_offset);
    error_code |= current_state.stack_ptr->pop(arith, args_size);
    error_code |= current_state.stack_ptr->pop(arith, ret_offset);
    error_code |= current_state.stack_ptr->pop(arith, ret_size);

    if (error_code == ERROR_SUCCESS) {
        // clean the address
        CuEVM::evm_address_conversion(arith, address);
        bn_t sender;
        current_state.message_ptr->get_recipient(arith, sender);  // I_{a}
        bn_t recipient;
        cgbn_set(arith.env, recipient, sender);  // I_{a}
        bn_t contract_address;
        cgbn_set(arith.env, contract_address, address);  // t
        bn_t storage_address;
        cgbn_set(arith.env, storage_address, sender);  // I_{a}
        CuEVM::byte_array_t call_data;
        CuEVM::byte_array_t code;

        new_state_ptr = new CuEVM::evm_call_state_t(
            arith, &current_state,
            new CuEVM::evm_message_call_t(arith, sender, recipient, contract_address, gas, value,
                                          current_state.message_ptr->get_depth() + 1, OP_CALLCODE, storage_address,
                                          call_data, code, ret_offset, ret_size,
                                          current_state.message_ptr->get_static_env()));

        error_code |= generic_CALL(arith, args_offset, args_size, current_state, new_state_ptr);
    }
    return error_code;
}

/**
 * The RETURN operation.
 * @param[in] arith The arithmetical environment.
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[in] stack The stack.
 * @param[in] memory The memory.
 * @param[out] return_data The return data.
 * @return ERROR_RETURN if the operation is successful, otherwise the error
 * code.
 */
__host__ __device__ int32_t RETURN(ArithEnv &arith, const bn_t &gas_limit, bn_t &gas_used, CuEVM::evm_stack_t &stack,
                                   CuEVM::evm_memory_t &memory, CuEVM::evm_return_data_t &return_data) {
    bn_t memory_offset, length;
    int32_t error_code = stack.pop(arith, memory_offset);
    error_code |= stack.pop(arith, length);

    bn_t memory_expansion_cost;
    error_code |=
        CuEVM::gas_cost::memory_grow_cost(arith, memory, memory_offset, length, memory_expansion_cost, gas_used);

    error_code |= CuEVM::gas_cost::has_gas(arith, gas_limit, gas_used);

    if (error_code == ERROR_SUCCESS) {
        memory.increase_memory_cost(arith, memory_expansion_cost);

        error_code |= memory.get(arith, memory_offset, length, return_data) | ERROR_RETURN;
    }

    return error_code;
}

/**
 * The DELEGATECALL operation. gives the new evm call state
 * @param[in] arith The arithmetical environment.
 * @param[in] current_state The current state.
 * @param[out] new_state_ptr The new state pointer.
 * @return 0 if the operation is successful, otherwise the error code.
 */
__host__ __device__ int32_t DELEGATECALL(ArithEnv &arith, CuEVM::evm_call_state_t &current_state,
                                         CuEVM::evm_call_state_t *&new_state_ptr) {
    bn_t gas, address, value, args_offset, args_size, ret_offset, ret_size;
    int32_t error_code = current_state.stack_ptr->pop(arith, gas);
    error_code |= current_state.stack_ptr->pop(arith, address);
    current_state.message_ptr->get_value(arith, value);
    error_code |= current_state.stack_ptr->pop(arith, args_offset);
    error_code |= current_state.stack_ptr->pop(arith, args_size);
    error_code |= current_state.stack_ptr->pop(arith, ret_offset);
    error_code |= current_state.stack_ptr->pop(arith, ret_size);

    if (error_code == ERROR_SUCCESS) {
        // clean the address
        CuEVM::evm_address_conversion(arith, address);
        bn_t sender;
        current_state.message_ptr->get_sender(arith, sender);  // keep the message call sender I_{s}
        bn_t recipient;
        current_state.message_ptr->get_recipient(arith, recipient);  // I_{a}
        bn_t contract_address;
        cgbn_set(arith.env, contract_address, address);  // t
        bn_t storage_address;
        cgbn_set(arith.env, storage_address, recipient);  // I_{a}
        CuEVM::byte_array_t call_data;
        CuEVM::byte_array_t code;

        new_state_ptr = new CuEVM::evm_call_state_t(
            arith, &current_state,
            new CuEVM::evm_message_call_t(arith, sender, recipient, contract_address, gas, value,
                                          current_state.message_ptr->get_depth() + 1, OP_DELEGATECALL, storage_address,
                                          call_data, code, ret_offset, ret_size,
                                          current_state.message_ptr->get_static_env()));

        error_code |= generic_CALL(arith, args_offset, args_size, current_state, new_state_ptr);
    }
    return error_code;
}

/**
 * The CREATE2 operation. gives the new evm call state
 * @param[in] arith The arithmetical environment.
 * @param[in] current_state The current state.
 * @param[out] new_state_ptr The new state pointer.
 * @return 0 if the operation is successful, otherwise the error code.
 */
__host__ __device__ int32_t CREATE2(ArithEnv &arith, CuEVM::evm_call_state_t &current_state,
                                    CuEVM::evm_call_state_t *&new_state_ptr) {
    return generic_CREATE(arith, current_state, new_state_ptr, OP_CREATE2);
}

/**
 * The STATICCALL operation. gives the new evm call state
 * @param[in] arith The arithmetical environment.
 * @param[in] current_state The current state.
 * @param[out] new_state_ptr The new state pointer.
 * @return 0 if the operation is successful, otherwise the error code.
 */
__host__ __device__ int32_t STATICCALL(ArithEnv &arith, CuEVM::evm_call_state_t &current_state,
                                       CuEVM::evm_call_state_t *&new_state_ptr) {
    bn_t gas, address, value, args_offset, args_size, ret_offset, ret_size;
    int32_t error_code = current_state.stack_ptr->pop(arith, gas);
    error_code |= current_state.stack_ptr->pop(arith, address);
    cgbn_set_ui32(arith.env, value, 0);
    error_code |= current_state.stack_ptr->pop(arith, args_offset);
    error_code |= current_state.stack_ptr->pop(arith, args_size);
    error_code |= current_state.stack_ptr->pop(arith, ret_offset);
    error_code |= current_state.stack_ptr->pop(arith, ret_size);

    if (error_code == ERROR_SUCCESS) {
        // clean the address
        CuEVM::evm_address_conversion(arith, address);
        bn_t sender;
        current_state.message_ptr->get_recipient(arith, sender);  //  I_{a}
        bn_t recipient;
        cgbn_set(arith.env, recipient, address);  // t
        bn_t contract_address;
        cgbn_set(arith.env, contract_address, address);  // t
        bn_t storage_address;
        cgbn_set(arith.env, storage_address, address);  // t
        CuEVM::byte_array_t call_data;
        CuEVM::byte_array_t code;

        new_state_ptr = new CuEVM::evm_call_state_t(
            arith, &current_state,
            new CuEVM::evm_message_call_t(arith, sender, recipient, contract_address, gas, value,
                                          current_state.message_ptr->get_depth() + 1, OP_STATICCALL, storage_address,
                                          call_data, code, ret_offset, ret_size, 1));

        error_code |= generic_CALL(arith, args_offset, args_size, current_state, new_state_ptr);
    }

    return error_code;
}

/**
 * The REVERT operation.
 * @param[in] arith The arithmetical environment.
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[in] stack The stack.
 * @param[in] memory The memory.
 * @param[out] return_data The return data.
 */
__host__ __device__ int32_t REVERT(ArithEnv &arith, const bn_t &gas_limit, bn_t &gas_used, CuEVM::evm_stack_t &stack,
                                   CuEVM::evm_memory_t &memory, CuEVM::evm_return_data_t &return_data) {
    bn_t memory_offset, length;
    int32_t error_code = stack.pop(arith, memory_offset);
    error_code |= stack.pop(arith, length);

    bn_t memory_expansion_cost;

    error_code |=
        CuEVM::gas_cost::memory_grow_cost(arith, memory, memory_offset, length, memory_expansion_cost, gas_used);

    error_code |= CuEVM::gas_cost::has_gas(arith, gas_limit, gas_used);

    if (error_code == ERROR_SUCCESS) {
        memory.increase_memory_cost(arith, memory_expansion_cost);

        error_code |= memory.get(arith, memory_offset, length, return_data) | ERROR_REVERT;
    }
    return error_code;
}

/**
 * The INVALID operation.
 * @return The error code.
 */
__host__ __device__ int32_t INVALID() { return ERROR_NOT_IMPLEMENTED; }

/**
 * The SELFDESTRUCT operation.
 * @param[in] arith The arithmetical environment.
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[inout] stack The stack.
 * @param[in] message The current context message call.
 * @param[inout] touch_state The touch state.
 * @param[out] return_data The return data.
 * @return 0 if the operation is successful, otherwise the error code.
 */
__host__ __device__ int32_t SELFDESTRUCT(ArithEnv &arith, const bn_t &gas_limit, bn_t &gas_used,
                                         CuEVM::evm_stack_t &stack, CuEVM::evm_message_call_t &message,
                                         CuEVM::TouchState &touch_state, CuEVM::evm_return_data_t &return_data) {
    int32_t error_code = ERROR_SUCCESS;
    if (message.get_static_env()) {
        error_code = ERROR_STATIC_CALL_CONTEXT_SELFDESTRUCT;
    } else {
        bn_t recipient;
        error_code |= stack.pop(arith, recipient);

        cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_SELFDESTRUCT);
        bn_t contract_address;
        message.get_contract_address(arith, contract_address);

        // custom logic, cannot use access_account_cost (no warm cost)
        if (!touch_state.is_warm_account(arith, recipient))
            cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_COLD_ACCOUNT_ACCESS);

        bn_t sender_balance;
        touch_state.get_balance(arith, contract_address, sender_balance);

        if (cgbn_compare_ui32(arith.env, sender_balance, 0) > 0) {
            if (touch_state.is_empty_account(arith, recipient)) {
                cgbn_add_ui32(arith.env, gas_used, gas_used, GAS_NEW_ACCOUNT);
            }
        }
        error_code |= CuEVM::gas_cost::has_gas(arith, gas_limit, gas_used);
        if (error_code == ERROR_SUCCESS) {
            bn_t recipient_balance;
            touch_state.get_balance(arith, recipient, recipient_balance);
            cgbn_add(arith.env, recipient_balance, recipient_balance, sender_balance);
            cgbn_set_ui32(arith.env, sender_balance, 0);
            touch_state.set_balance(arith, recipient, recipient_balance);
            touch_state.set_balance(arith, contract_address, sender_balance);
            // receiver = self => 0 balance
            return_data = CuEVM::evm_return_data_t();
            error_code |= ERROR_RETURN;
        }
    }
    return error_code;
}
}  // namespace CuEVM::operations
