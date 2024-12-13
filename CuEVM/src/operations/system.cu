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
__device__ int32_t generic_CALL(const evm_word_t &args_offset, const evm_word_t &args_size,
                                CuEVM::evm_call_state_t &current_state, CuEVM::evm_call_state_t *&new_state_ptr,
                                CuEVM::cached_evm_call_state &cached_state) {
    // try to send value in call
    evm_word_t value;
    new_state_ptr->message_ptr->get_value(value);
    // #ifdef __CUDA_ARCH__
    //     printf("generic_CALL message_ptr->get_value %d\n", threadIdx.x);
    // #endif
    int32_t error_code = ((new_state_ptr->message_ptr->get_static_env() && (uint256_cmp_word(&value, 0) != 0) &&
                           (new_state_ptr->message_ptr->get_call_type() == OP_CALL))
                              ? ERROR_STATIC_CALL_CONTEXT_CALL_VALUE
                              : ERROR_SUCCESS);

    // charge the gas for the call
    // #ifdef __CUDA_ARCH__
    //     printf("Before memory_grow_cost %d\n", threadIdx.x);
    //     __SYNC_THREADS__
    //     print_bnt(arith, cached_state.gas_used);
    // #endif
    // memory call data
    gas_t memory_expansion_cost_args;

    // replace gas_used, throw away after the call
    // because we did not increase_memory_cost between expansions
    gas_t temp_memory_gas_used;
    // reset to 0;
    temp_memory_gas_used = 0;

    error_code |= CuEVM::gas_cost::memory_grow_cost(*current_state.memory_ptr, args_offset, args_size,
                                                    memory_expansion_cost_args, temp_memory_gas_used);

    // printf("generic_CALL memory_expansion_cost_args idx %d error_code %d mempointer %p memsize %d\n", THREADIDX,
    //        error_code, current_state.memory_ptr, current_state.memory_ptr->size);
    // __SYNC_THREADS__
    // print_bnt(arith, temp_memory_gas_used);
    // print_bnt(arith, args_offset);
    // print_bnt(arith, args_size);

    // memory return data
    evm_word_t ret_offset, ret_size;
    new_state_ptr->message_ptr->get_return_data_offset(ret_offset);
    new_state_ptr->message_ptr->get_return_data_size(ret_size);
    gas_t memory_expansion_cost_ret;
    error_code |= CuEVM::gas_cost::memory_grow_cost(*current_state.memory_ptr, ret_offset, ret_size,
                                                    memory_expansion_cost_ret, temp_memory_gas_used);

    // #ifdef __CUDA_ARCH__
    //     printf("generic_CALL memory_expansion_cost_ret idx %d error_code %d mempointer %p memsize %d\n", threadIdx.x,
    //            error_code, current_state.memory_ptr, current_state.memory_ptr->size);
    //     __SYNC_THREADS__
    //     print_bnt(arith, temp_memory_gas_used);
    //     print_bnt(arith, ret_offset);
    //     print_bnt(arith, ret_size);
    // #endif
    // compute the total memory expansion cost
    gas_t memory_expansion_cost;
    if (memory_expansion_cost_args > memory_expansion_cost_ret) {
        memory_expansion_cost = memory_expansion_cost_args;
    } else {
        memory_expansion_cost = memory_expansion_cost_ret;
    }
    cached_state.gas_used += memory_expansion_cost;

    // #ifdef __CUDA_ARCH__
    //     printf("after memory_grow_cost %d\n", threadIdx.x);
    //     __SYNC_THREADS__
    //     print_bnt(arith, cached_state.gas_used);
    // #endif

    // adress warm call
    // bn_t contract_address;
    evm_word_t *contract_address_ptr = &new_state_ptr->message_ptr->contract_address;
    // new_state_ptr->message_ptr->get_contract_address(arith, contract_address);
    CuEVM::gas_cost::access_account_cost(cached_state.gas_used, current_state.state_db_ptr, contract_address_ptr);
    // positive value call cost (except delegate call)
    // empty account call cost
    // #ifdef __CUDA_ARCH__
    //     printf("After access_account_cost cost %d\n", threadIdx.x);
    //     __SYNC_THREADS__
    //     print_bnt(arith, cached_state.gas_used);
    // #endif

    gas_t gas_stippend;
    gas_stippend = 0;
    if (new_state_ptr->message_ptr->get_call_type() != OP_DELEGATECALL) {
        if (uint256_cmp_word(&value, 0) > 0) {
            cached_state.gas_used += GAS_CALL_VALUE;
            gas_stippend = GAS_CALL_STIPEND;
            // If the empty account is called
            // only for call opcode
            if ((current_state.state_db_ptr->is_empty_account(contract_address_ptr)) &&
                (new_state_ptr->message_ptr->get_call_type() == OP_CALL)) {
                cached_state.gas_used += GAS_NEW_ACCOUNT;
            };
        }
    }
    // max gas call, gas_sent_with_call
    gas_t gas_capped;
    CuEVM::gas_cost::max_gas_call(gas_capped, cached_state.gas_limit, cached_state.gas_used);

    // limit the gas to the gas capped
    if (new_state_ptr->gas_limit > gas_capped) {
        new_state_ptr->gas_limit = gas_capped;
    }
    // add the the gas sent to the gas used
    cached_state.gas_used += new_state_ptr->gas_limit;

    // Gas stipen 2300 is added to the total gas limit but not gas used
    // add the gas stippend to gas limit of the child call
    new_state_ptr->gas_limit += gas_stippend;

    error_code |= CuEVM::gas_cost::has_gas(cached_state.gas_limit, cached_state.gas_used);

    if (error_code == ERROR_SUCCESS) {
        // increase the memory cost
        current_state.memory_ptr->increase_memory_cost(memory_expansion_cost);
        // set the byte code
        // FIX: MAke the warm up later for the contract in START_CALL
        // CuEVM::account_t *contract=nullptr;
        // error_code |= access_state.get_account(arith, contract_address,
        // contract, ACCOUNT_NONE_FLAG);
        // new_state_ptr->message_ptr->set_byte_code(
        //     contract->byte_code);

        // get/set the call data
        error_code |= current_state.memory_ptr->get(args_offset, args_size, *new_state_ptr->message_ptr->data);
    }
    // #ifdef __CUDA_ARCH__
    //     printf("generic_CALL error_code: %d idx %d\n", error_code, threadIdx.x);
    // #endif
    return error_code;
}

/**
 * Make a generic create call.
 * @param[in] arith The arithmetical environment.
 * @param[in] current_state The current state.
 * @param[out] new_state_ptr The new state pointer.
 * @return 0 if the operation is successful, otherwise the error code.
 */
__device__ int32_t generic_CREATE(CuEVM::evm_call_state_t &current_state, CuEVM::evm_call_state_t *&new_state_ptr,
                                  const uint32_t opcode, cached_evm_call_state &cached_state) {
    evm_word_t value, memory_offset, length;
    int32_t error_code = cached_state.stack_ptr->pop(value);
    error_code |= cached_state.stack_ptr->pop(memory_offset);
    error_code |= cached_state.stack_ptr->pop(length);
    // create cost
    cached_state.gas_used += GAS_CREATE;

    // compute the memory cost
    gas_t memory_expansion_cost;
    error_code |= CuEVM::gas_cost::memory_grow_cost(*current_state.memory_ptr, memory_offset, length,
                                                    memory_expansion_cost, cached_state.gas_used);

    // compute the initcode gas cost
    CuEVM::gas_cost::initcode_cost(cached_state.gas_used, uint256_get_uint32_t(&length));

    evm_word_t salt;
    if (opcode == OP_CREATE2) {
        error_code |= cached_state.stack_ptr->pop(salt);
        // compute the keccak gas cost
        CuEVM::gas_cost::keccak_cost(cached_state.gas_used, uint256_get_uint32_t(&length));
    }
    // #ifdef __CUDA_ARCH__
    //     printf("Before has_gas %d error code %d\n", threadIdx.x, error_code);
    //     print_bnt(arith, cached_state.gas_limit);
    //     print_bnt(arith, cached_state.gas_used);
    // #endif

    error_code |= CuEVM::gas_cost::has_gas(cached_state.gas_limit, cached_state.gas_used);

    if (error_code == ERROR_SUCCESS) {
        // increase the memory cost
        current_state.memory_ptr->increase_memory_cost(memory_expansion_cost);
        // #ifdef __CUDA_ARCH__
        //         printf("loading initialisation_code %d:\n", threadIdx.x);
        //         print_bnt(arith, memory_offset);
        //         print_bnt(arith, length);
        //         current_state.memory_ptr->print();
        // #endif
        // get the initialisation code
        CuEVM::byte_array_t initialisation_code;
        current_state.memory_ptr->get(memory_offset, length, initialisation_code);
        // #ifdef __CUDA_ARCH__
        //         printf("initialisation_code %d:\n", threadIdx.x);
        //         initialisation_code.print();
        // #endif
        evm_word_t sender_address;
        current_state.message_ptr->get_recipient(sender_address);
        const evm_word_t *sender_address_ptr = &current_state.message_ptr->recipient;
        evm_word_t contract_address;

        // // warm up the contract address
        // error_code |=
        //     current_state.touch_state.set_warm_account(arith,
        //     contract_address);
        // printf("generic_CREATE senderaddress ptr %p\n", sender_address_ptr);
        // sender_address_ptr->print();

        uint32_t sender_nonce_uint = current_state.state_db_ptr->get_nonce(sender_address_ptr);
        evm_word_t sender_nonce(sender_nonce_uint);
        // Do not get_account after this to reuse sender_account
        if (opcode == OP_CREATE2) {
            CuEVM::utils::get_contract_address_create2(&contract_address, &sender_address, &salt, initialisation_code);
        } else {
            CuEVM::utils::get_contract_address_create(&contract_address, &sender_address, &sender_nonce);
        }

        if (!current_state.state_db_ptr->is_empty_account(&contract_address)) {
            // corner collision case: must set warm for the contract address
            current_state.state_db_ptr->set_warm_account(&contract_address);
            error_code |= ERROR_MESSAGE_CALL_CREATE_CONTRACT_EXISTS;
        }

        // gas capped limit
        gas_t gas_capped;
        CuEVM::gas_cost::max_gas_call(gas_capped, cached_state.gas_limit, cached_state.gas_used);
        // add the gas sent to the gas used
        cached_state.gas_used += gas_capped;
        // the return data offset and size
        evm_word_t ret_offset, ret_size;
        ret_offset.set_zero();
        ret_size.set_zero();
        CuEVM::byte_array_t call_data;

        evm_message_call_t_shadow *message_call_ptr = new CuEVM::evm_message_call_t_shadow(
            &current_state.message_ptr->recipient, &contract_address, &contract_address, gas_capped, &value,
            current_state.message_ptr->get_depth() + 1, opcode, &contract_address, call_data, initialisation_code,
            ret_offset, ret_size, current_state.message_ptr->get_static_env());

        current_state.message_ptr->copy_from(message_call_ptr);
        // create the new evm call state
        new_state_ptr = new CuEVM::evm_call_state_t(&current_state, current_state.message_ptr, message_call_ptr);

        error_code |= (current_state.message_ptr->get_static_env() ? ERROR_STATIC_CALL_CONTEXT_CREATE :
#ifdef EIP_3860
                                                                   (uint256_get_uint32_t(&length) > max_initcode_size
                                                                        ? ERROR_CREATE_INIT_CODE_SIZE_EXCEEDED
                                                                        : ERROR_SUCCESS)
#else
                                                                   ERROR_SUCCESS
#endif
        );
        // printf("generic_CREATE error_code: %d\n", error_code);
        if (current_state.state_db_ptr->is_contract(&sender_address)) {
            current_state.state_db_ptr->update_nonce(&sender_address,
                                                     current_state.state_db_ptr->get_nonce(&sender_address) + 1);
        }
    }

    // printf("generic_CREATE error_code: %d\n", error_code);
    return error_code;
}

/**
 * The STOP operation.
 * @param[out] return_data The return data.
 * @return return error code.
 */
__device__ int32_t STOP(CuEVM::evm_return_data_t &return_data) {
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
__device__ int32_t CREATE(CuEVM::evm_call_state_t &current_state, CuEVM::evm_call_state_t *&new_state_ptr,
                          CuEVM::cached_evm_call_state &cached_state) {
    return generic_CREATE(current_state, new_state_ptr, OP_CREATE, cached_state);
}

/**
 * The CALL operation. gives the new evm call state
 * @param[in] arith The arithmetical environment.
 * @param[in] current_state The current state.
 * @param[out] new_state_ptr The new state pointer.
 * @return 0 if the operation is successful, otherwise the error code.
 */
__device__ int32_t CALL(CuEVM::evm_call_state_t &current_state, CuEVM::evm_call_state_t *&new_state_ptr,
                        CuEVM::cached_evm_call_state &cached_state) {
    evm_word_t gas_word, address, value, args_offset, args_size, ret_offset, ret_size;
    // #ifdef __CUDA_ARCH__
    //     printf("opcode CALL %d\n", threadIdx.x);
    // #endif
    int32_t error_code = cached_state.stack_ptr->pop(gas_word);
    error_code |= cached_state.stack_ptr->pop(address);
    error_code |= cached_state.stack_ptr->pop(value);
    error_code |= cached_state.stack_ptr->pop(args_offset);
    error_code |= cached_state.stack_ptr->pop(args_size);
    error_code |= cached_state.stack_ptr->pop(ret_offset);
    error_code |= cached_state.stack_ptr->pop(ret_size);
    gas_t gas = uint256_get_uint32_t(&gas_word);
    // printf("opcode CALL params after pop\n");
    // print_bnt(arith, gas);
    // print_bnt(arith, address);
    // print_bnt(arith, value);
    // print_bnt(arith, args_offset);
    // print_bnt(arith, args_size);
    // print_bnt(arith, ret_offset);
    // print_bnt(arith, ret_size);
    // #ifdef __CUDA_ARCH__
    //     printf("opcode CALL before error_code == ERROR_SUCCESS %d\n", threadIdx.x);
    // #endif
    if (error_code == ERROR_SUCCESS) {
        // clean the address
        CuEVM::utils::evm_address_conversion(address);

        CuEVM::byte_array_t call_data;
        CuEVM::byte_array_t code;

        evm_message_call_t_shadow *message_call_ptr = new CuEVM::evm_message_call_t_shadow(
            &current_state.message_ptr->recipient, &address, &address, gas, &value,
            current_state.message_ptr->get_depth() + 1, OP_CALL, &address, call_data, code, ret_offset, ret_size,
            current_state.message_ptr->get_static_env());

        current_state.message_ptr->copy_from(message_call_ptr);

        new_state_ptr = new CuEVM::evm_call_state_t(&current_state, current_state.message_ptr, message_call_ptr);

        // #ifdef __CUDA_ARCH__
        //         printf("opcode CALL after constructing message call t  %d\n", threadIdx.x);
        // #endif
        // #ifdef __CUDA_ARCH__
        // printf("opcode CALL after constructing message call t  %d\n", threadIdx.x);
        // #endif
    }
    if (error_code == ERROR_SUCCESS)  // break down scope to avoid stack problems
        error_code |= generic_CALL(args_offset, args_size, current_state, new_state_ptr, cached_state);

    // printf("opcode CALL error_code %d thread %d\n", error_code, THREADIDX);
    return error_code;
}

/**
 * The CALLCODE operation. gives the new evm call state
 * @param[in] arith The arithmetical environment.
 * @param[in] current_state The current state.
 * @param[out] new_state_ptr The new state pointer.
 * @return 0 if the operation is successful, otherwise the error code.
 */
__device__ int32_t CALLCODE(CuEVM::evm_call_state_t &current_state, CuEVM::evm_call_state_t *&new_state_ptr,
                            CuEVM::cached_evm_call_state &cached_state) {
    evm_word_t gas_word, address, value, args_offset, args_size, ret_offset, ret_size;
    int32_t error_code = cached_state.stack_ptr->pop(gas_word);
    error_code |= cached_state.stack_ptr->pop(address);
    error_code |= cached_state.stack_ptr->pop(value);
    error_code |= cached_state.stack_ptr->pop(args_offset);
    error_code |= cached_state.stack_ptr->pop(args_size);
    error_code |= cached_state.stack_ptr->pop(ret_offset);
    error_code |= cached_state.stack_ptr->pop(ret_size);
    gas_t gas = uint256_get_uint32_t(&gas_word);
    if (error_code == ERROR_SUCCESS) {
        // clean the address
        CuEVM::utils::evm_address_conversion(address);

        CuEVM::byte_array_t call_data;
        CuEVM::byte_array_t code;

        evm_message_call_t_shadow *message_call_ptr = new CuEVM::evm_message_call_t_shadow(
            &current_state.message_ptr->recipient, &current_state.message_ptr->recipient, &address, gas, &value,
            current_state.message_ptr->get_depth() + 1, OP_CALLCODE, &current_state.message_ptr->recipient, call_data,
            code, ret_offset, ret_size, current_state.message_ptr->get_static_env());

        current_state.message_ptr->copy_from(message_call_ptr);
        new_state_ptr = new CuEVM::evm_call_state_t(&current_state, current_state.message_ptr, message_call_ptr);

        error_code |= generic_CALL(args_offset, args_size, current_state, new_state_ptr, cached_state);
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
__device__ int32_t RETURN(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack,
                          CuEVM::evm_memory_t &memory, CuEVM::evm_return_data_t &return_data) {
    evm_word_t memory_offset, length;
    int32_t error_code = stack.pop(memory_offset);
    error_code |= stack.pop(length);

    CuEVM::gas_t memory_expansion_cost;
    error_code |= CuEVM::gas_cost::memory_grow_cost(memory, memory_offset, length, memory_expansion_cost, gas_used);

    error_code |= CuEVM::gas_cost::has_gas(gas_limit, gas_used);

    if (error_code == ERROR_SUCCESS) {
        memory.increase_memory_cost(memory_expansion_cost);

        error_code |= memory.get(memory_offset, length, return_data) | ERROR_RETURN;
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
__device__ int32_t DELEGATECALL(CuEVM::evm_call_state_t &current_state, CuEVM::evm_call_state_t *&new_state_ptr,
                                CuEVM::cached_evm_call_state &cached_state) {
    evm_word_t gas_word, address, value, args_offset, args_size, ret_offset, ret_size;
    int32_t error_code = cached_state.stack_ptr->pop(gas_word);
    error_code |= cached_state.stack_ptr->pop(address);
    current_state.message_ptr->get_value(value);
    error_code |= cached_state.stack_ptr->pop(args_offset);
    error_code |= cached_state.stack_ptr->pop(args_size);
    error_code |= cached_state.stack_ptr->pop(ret_offset);
    error_code |= cached_state.stack_ptr->pop(ret_size);
    gas_t gas = uint256_get_uint32_t(&gas_word);

    if (error_code == ERROR_SUCCESS) {
        // clean the address
        CuEVM::utils::evm_address_conversion(address);

        CuEVM::byte_array_t call_data;
        CuEVM::byte_array_t code;

        evm_message_call_t_shadow *message_call_ptr = new CuEVM::evm_message_call_t_shadow(
            &current_state.message_ptr->sender, &current_state.message_ptr->recipient, &address, gas, &value,
            current_state.message_ptr->get_depth() + 1, OP_DELEGATECALL, &current_state.message_ptr->recipient,
            call_data, code, ret_offset, ret_size, current_state.message_ptr->get_static_env());
        current_state.message_ptr->copy_from(message_call_ptr);

        new_state_ptr = new CuEVM::evm_call_state_t(&current_state, current_state.message_ptr, message_call_ptr);

        error_code |= generic_CALL(args_offset, args_size, current_state, new_state_ptr, cached_state);
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
__device__ int32_t CREATE2(CuEVM::evm_call_state_t &current_state, CuEVM::evm_call_state_t *&new_state_ptr,
                           CuEVM::cached_evm_call_state &cached_state) {
    return generic_CREATE(current_state, new_state_ptr, OP_CREATE2, cached_state);
}

/**
 * The STATICCALL operation. gives the new evm call state
 * @param[in] arith The arithmetical environment.
 * @param[in] current_state The current state.
 * @param[out] new_state_ptr The new state pointer.
 * @return 0 if the operation is successful, otherwise the error code.
 */
__device__ int32_t STATICCALL(CuEVM::evm_call_state_t &current_state, CuEVM::evm_call_state_t *&new_state_ptr,
                              CuEVM::cached_evm_call_state &cached_state) {
    evm_word_t gas_word, address, value, args_offset, args_size, ret_offset, ret_size;
    int32_t error_code = cached_state.stack_ptr->pop(gas_word);
    error_code |= cached_state.stack_ptr->pop(address);
    value.set_zero();
    error_code |= cached_state.stack_ptr->pop(args_offset);
    error_code |= cached_state.stack_ptr->pop(args_size);
    error_code |= cached_state.stack_ptr->pop(ret_offset);
    error_code |= cached_state.stack_ptr->pop(ret_size);
    gas_t gas = uint256_get_uint32_t(&gas_word);

    if (error_code == ERROR_SUCCESS) {
        // clean the address
        CuEVM::utils::evm_address_conversion(address);

        CuEVM::byte_array_t call_data;
        CuEVM::byte_array_t code;

        evm_message_call_t_shadow *message_call_ptr = new CuEVM::evm_message_call_t_shadow(
            &current_state.message_ptr->sender, &current_state.message_ptr->recipient, &address, gas, &value,
            current_state.message_ptr->get_depth() + 1, OP_STATICCALL, &current_state.message_ptr->recipient, call_data,
            code, ret_offset, ret_size, current_state.message_ptr->get_static_env());
        current_state.message_ptr->copy_from(message_call_ptr);

        new_state_ptr = new CuEVM::evm_call_state_t(&current_state, current_state.message_ptr, message_call_ptr);

        error_code |= generic_CALL(args_offset, args_size, current_state, new_state_ptr, cached_state);
    }

    return error_code;
}

/**
 * The REVERT operation.
 * @param[in] gas_limit The gas limit.
 * @param[inout] gas_used The gas used.
 * @param[in] stack The stack.
 * @param[in] memory The memory.
 * @param[out] return_data The return data.
 */
__device__ int32_t REVERT(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack,
                          CuEVM::evm_memory_t &memory, CuEVM::evm_return_data_t &return_data) {
    evm_word_t memory_offset, length;
    int32_t error_code = stack.pop(memory_offset);
    error_code |= stack.pop(length);

    CuEVM::gas_t memory_expansion_cost;

    error_code |= CuEVM::gas_cost::memory_grow_cost(memory, memory_offset, length, memory_expansion_cost, gas_used);

    error_code |= CuEVM::gas_cost::has_gas(gas_limit, gas_used);

    if (error_code == ERROR_SUCCESS) {
        memory.increase_memory_cost(memory_expansion_cost);

        error_code |= memory.get(memory_offset, length, return_data) | ERROR_REVERT;
    }
    return error_code;
}

/**
 * The INVALID operation.
 * @return The error code.
 */
__device__ int32_t INVALID() { return ERROR_NOT_IMPLEMENTED; }

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
__device__ int32_t SELFDESTRUCT(const CuEVM::gas_t &gas_limit, CuEVM::gas_t &gas_used, CuEVM::evm_stack_t &stack,
                                CuEVM::evm_message_call_t &message, CuEVM::StateDb *state_db_ptr,
                                CuEVM::evm_return_data_t &return_data) {
    int32_t error_code = ERROR_SUCCESS;
    if (message.get_static_env()) {
        error_code = ERROR_STATIC_CALL_CONTEXT_SELFDESTRUCT;
    } else {
        evm_word_t recipient;
        error_code |= stack.pop(recipient);

        // custom logic, cannot use access_account_cost (no warm cost)
        if (!state_db_ptr->is_warm_account(&recipient)) gas_used += GAS_COLD_ACCOUNT_ACCESS;

        evm_word_t *sender_balance = state_db_ptr->get_balance(&message.contract_address);

        if (uint256_is_zero(sender_balance)) {
            if (state_db_ptr->is_empty_account(&recipient)) {
                gas_used += GAS_NEW_ACCOUNT;
            }
        }
        error_code |= CuEVM::gas_cost::has_gas(gas_limit, gas_used);
        if (error_code == ERROR_SUCCESS) {
            evm_word_t *recipient_balance = state_db_ptr->get_balance(&recipient);
            uint256_add(recipient_balance, recipient_balance, sender_balance);
            sender_balance->set_zero();
            state_db_ptr->update_balance(&recipient, recipient_balance);
            state_db_ptr->update_balance(&message.contract_address, sender_balance);
            // receiver = self => 0 balance
            return_data = CuEVM::evm_return_data_t();
            error_code |= ERROR_RETURN;
        }
    }
    return error_code;
}
}  // namespace CuEVM::operations
