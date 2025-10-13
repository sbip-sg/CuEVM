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
__device__ int32_t generic_CALL(const evm_word_t *args_offset, const evm_word_t *args_size,
                                CuEVM::evm_memory_t *parent_memory_ptr, CuEVM::evm_call_context_t *&new_context_ptr,
                                CuEVM::cached_evm_call_context &cached_state) {
    bool non_zero_value = uint256_cmp_word(&new_context_ptr->value, 0) != 0;
    int32_t error_code = ((new_context_ptr->static_env && non_zero_value && (new_context_ptr->call_type == OP_CALL))
                              ? ERROR_STATIC_CALL_CONTEXT_CALL_VALUE
                              : ERROR_SUCCESS);

    // replace gas_used, throw away after the call
    // because we did not increase_memory_cost between expansions
    gas_t temp_memory_gas_used = 0;
    // memory call data
    gas_t memory_expansion_cost_args;

    uint32_t args_offset_ui32 = uint256_get_uint32_t(args_offset);
    uint32_t args_size_ui32 = uint256_get_uint32_t(args_size);
    if (uint256_cmp_word(args_offset, args_offset_ui32) != 0 || uint256_cmp_word(args_size, args_size_ui32) != 0) {
        return ERR_MEMORY_INVALID_OFFSET;
    }
    error_code |= CuEVM::gas_cost::memory_grow_cost(parent_memory_ptr, args_offset_ui32, args_size_ui32,
                                                    memory_expansion_cost_args, temp_memory_gas_used);

    // memory return data
    evm_word_t ret_offset = new_context_ptr->fixed_ret_offset;
    evm_word_t ret_size = new_context_ptr->fixed_ret_size;
    gas_t memory_expansion_cost_ret;
    uint32_t ret_offset_ui32 = uint256_get_uint32_t(&ret_offset);
    uint32_t ret_size_ui32 = uint256_get_uint32_t(&ret_size);
    if (uint256_cmp_word(&ret_offset, ret_offset_ui32) != 0 || uint256_cmp_word(&ret_size, ret_size_ui32) != 0) {
        return ERR_MEMORY_INVALID_OFFSET;
    }
    error_code |= CuEVM::gas_cost::memory_grow_cost(parent_memory_ptr, ret_offset_ui32, ret_size_ui32,
                                                    memory_expansion_cost_ret, temp_memory_gas_used);

    gas_t memory_expansion_cost;
    if (memory_expansion_cost_args > memory_expansion_cost_ret) {
        memory_expansion_cost = memory_expansion_cost_args;
    } else {
        memory_expansion_cost = memory_expansion_cost_ret;
    }

    cached_state.gas_used += memory_expansion_cost;

    // adress warm call
    evm_word_t *contract_address_ptr = &new_context_ptr->to;
    // move access account cost to before this function

    gas_t gas_stippend = 0;
    if (new_context_ptr->call_type != OP_DELEGATECALL && non_zero_value) {
        cached_state.gas_used += GAS_CALL_VALUE;
        gas_stippend = GAS_CALL_STIPEND;
        // If the empty account is called
        // only for call opcode
        if ((CuEVM::global_state_db_ptr->is_empty_account(contract_address_ptr)) &&
            (new_context_ptr->call_type == OP_CALL)) {
            cached_state.gas_used += GAS_NEW_ACCOUNT;
        };
    }

    // max gas call, gas_sent_with_call
    gas_t gas_capped = CuEVM::gas_cost::max_gas_call(cached_state.gas_limit, cached_state.gas_used);
    // printf("gas capped %lu\n", gas_capped);
    // limit the gas to the gas capped
    if (new_context_ptr->gas_limit > gas_capped) {
        new_context_ptr->gas_limit = gas_capped;
    }
    // printf("new gas limit %u\n", new_context_ptr->gas_limit);
    // add the the gas sent to the gas used
    cached_state.gas_used += new_context_ptr->gas_limit;

    // Gas stipen 2300 is added to the total gas limit but not gas used
    // add the gas stippend to gas limit of the child call
    new_context_ptr->gas_limit += gas_stippend;

    error_code |= CuEVM::gas_cost::has_gas(cached_state.gas_limit, cached_state.gas_used);

    if (error_code == ERROR_SUCCESS) {
        // increase the memory cost
        parent_memory_ptr->increase_memory_cost(memory_expansion_cost);

        if (new_context_ptr->call_type != OP_CALLCODE &&
            new_context_ptr->call_type != OP_DELEGATECALL) {  // special case: the code is set outside
            new_context_ptr->byte_code =
                CuEVM::global_state_db_ptr->get_code(new_context_ptr->byte_code_size, contract_address_ptr);
            new_context_ptr->bytecode_offset = find_global_bytecode_offset(contract_address_ptr);
        }
#ifdef BUILD_LIBRARY
        if (new_context_ptr->from.words[0] == REENTRANCY_ATTACKER_ADDRESS) {
            // bypass args logic for reentrancy attacker
            CuEVM::evm_call_context_t *parent_context = new_context_ptr->parent;
            while (parent_context->parent != nullptr) {
                parent_context = parent_context->parent;
            }
            new_context_ptr->call_data = parent_context->call_data;
            new_context_ptr->call_data_size = parent_context->call_data_size;
            new_context_ptr->value = parent_context->value;
            // calldata belong to the memory structure, safe to use without needing to free later.
            return ERROR_SUCCESS;
        }
#endif
        uint8_t *call_data = nullptr;
        if (uint256_cmp_word(args_size, 0) > 0)
            error_code |= parent_memory_ptr->get(args_offset_ui32, args_size_ui32, call_data);
        new_context_ptr->call_data = call_data;
        new_context_ptr->call_data_size = args_size_ui32;
    }

    // printf("generic_CALL error_code: %d idx %d\n", error_code, THREADIDX);
    return error_code;
}

/**
 * Make a generic create call.
 * @param[in] arith The arithmetical environment.
 * @param[in] current_state The current state.
 * @param[out] new_state_ptr The new state pointer.
 * @return 0 if the operation is successful, otherwise the error code.
 */
__device__ int32_t generic_CREATE(CuEVM::evm_call_context_t *current_context,
                                  CuEVM::evm_call_context_t *&new_context_ptr, const uint32_t call_type,
                                  CuEVM::cached_evm_call_context &cached_state) {
    evm_word_t *value, *memory_offset, *length;
    if (cached_state.stack_ptr->size() < 3) return ERROR_STACK_UNDERFLOW;
    value = cached_state.stack_ptr->get_address_at_index(1);
    memory_offset = cached_state.stack_ptr->get_address_at_index(2);
    length = cached_state.stack_ptr->get_address_at_index(3);
    cached_state.stack_ptr->reduce_size(3);
    // create cost
    cached_state.gas_used += GAS_CREATE;

    // compute the memory cost
    gas_t memory_expansion_cost;
    uint32_t memory_offset_ui32 = uint256_get_uint32_t(memory_offset);
    uint32_t length_ui32 = uint256_get_uint32_t(length);
    if ((uint256_cmp_word(memory_offset, memory_offset_ui32) != 0 && !uint256_is_zero(length)) ||
        uint256_cmp_word(length, length_ui32) != 0) {
        return ERR_MEMORY_INVALID_OFFSET;
    }
    int32_t error_code = CuEVM::gas_cost::memory_grow_cost(current_context->memory_ptr, memory_offset_ui32, length_ui32,
                                                           memory_expansion_cost, cached_state.gas_used);

    // compute the initcode gas cost
    CuEVM::gas_cost::initcode_cost(cached_state.gas_used, uint256_get_uint32_t(length));

    evm_word_t salt;
    if (call_type == OP_CREATE2) {
        error_code |= cached_state.stack_ptr->pop(salt);
        // compute the keccak gas cost
        CuEVM::gas_cost::keccak_cost(cached_state.gas_used, uint256_get_uint32_t(length));
    }

    error_code |= CuEVM::gas_cost::has_gas(cached_state.gas_limit, cached_state.gas_used);

    if (error_code == ERROR_SUCCESS) {
        // increase the memory cost
        current_context->memory_ptr->increase_memory_cost(memory_expansion_cost);

        // get the initialisation code
        uint8_t *initialisation_code = nullptr;
        current_context->memory_ptr->get(memory_offset_ui32, length_ui32, initialisation_code);

        evm_word_t contract_address;

        uint32_t sender_nonce_uint = CuEVM::global_state_db_ptr->get_nonce(&current_context->to);
        evm_word_t sender_nonce(sender_nonce_uint);
        // Do not get_account after this to reuse sender_account
        if (call_type == OP_CREATE2) {
            CuEVM::utils::get_contract_address_create2(&contract_address, &current_context->to, &salt,
                                                       initialisation_code, length_ui32);

        } else {
            CuEVM::utils::get_contract_address_create(&contract_address, &current_context->to, &sender_nonce);
        }

        if (!CuEVM::global_state_db_ptr->is_empty_create(&contract_address)) {
            // corner collision case: must set warm for the contract address
            CuEVM::global_state_db_ptr->set_warm_account(&contract_address);
            error_code |= ERROR_MESSAGE_CALL_CREATE_CONTRACT_EXISTS;
        }

        // gas capped limit
        gas_t gas_capped = CuEVM::gas_cost::max_gas_call(cached_state.gas_limit, cached_state.gas_used);

        // // add the gas sent to the gas used
        cached_state.gas_used += gas_capped;
        // the return data offset and size
        evm_word_t ret_offset, ret_size;
        ret_offset.set_zero();
        ret_size.set_zero();

        // new_context_ptr = new CuEVM::evm_call_context_t();
        new_context_ptr = memory_pool::get_call_context(current_context->depth);

        int32_t bytecode_offset = -1;

        new_context_ptr->initiate_values(current_context, gas_capped, current_context->to, contract_address,
                                         contract_address, *value, call_type, nullptr, 0, initialisation_code,
                                         length_ui32, bytecode_offset, 0, current_context->static_env);

        error_code |= (current_context->static_env ? ERROR_STATIC_CALL_CONTEXT_CREATE :
#ifdef EIP_3860
                                                   (uint256_get_uint32_t(length) > max_initcode_size
                                                        ? ERROR_CREATE_INIT_CODE_SIZE_EXCEEDED
                                                        : ERROR_SUCCESS)
#else
                                                   ERROR_SUCCESS
#endif
        );

        // todo update nonce
        if (CuEVM::global_state_db_ptr->is_contract(&current_context->to)) {
            CuEVM::global_state_db_ptr->update_nonce(&current_context->to,
                                                     CuEVM::global_state_db_ptr->get_nonce(&current_context->to) + 1);
        }

        CuEVM::global_state_db_ptr->update_nonce(&contract_address, 1);
    }

    return error_code;
}

/**
 * The STOP operation.
 * @param[out] return_data The return data.
 * @return return error code.
 */
__device__ int32_t STOP(CuEVM::evm_call_context_t *call_state_ptr) {
    // TODO: fix this
    call_state_ptr->dynamic_ret_size = 0;
    if (call_state_ptr->parent != nullptr) {
        call_state_ptr->parent->dynamic_ret_size = 0;
    }
    return ERROR_RETURN;
}

/**
 * The CREATE operation. gives the new evm call state
 * @param[in] arith The arithmetical environment.
 * @param[in] current_state The current state.
 * @param[out] new_state_ptr The new state pointer.
 * @return 0 if the operation is successful, otherwise the error code.
 */
__device__ int32_t CREATE(CuEVM::evm_call_context_t *current_context, CuEVM::evm_call_context_t *&new_context_ptr,
                          CuEVM::cached_evm_call_context &cached_state) {
    return generic_CREATE(current_context, new_context_ptr, OP_CREATE, cached_state);
}

/**
 * The CALL operation. gives the new evm call state
 * @param[in] arith The arithmetical environment.
 * @param[in] current_state The current state.
 * @param[out] new_state_ptr The new state pointer.
 * @return 0 if the operation is successful, otherwise the error code.
 */
__device__ int32_t CALL(CuEVM::evm_call_context_t *current_context, CuEVM::evm_call_context_t *&new_context_ptr,
                        CuEVM::cached_evm_call_context &cached_state) {
    evm_word_t *gas_word, *original_address, *value, *args_offset, *args_size, *ret_offset, *ret_size;

    if (cached_state.stack_ptr->size() < 7) return ERROR_STACK_UNDERFLOW;
    gas_word = cached_state.stack_ptr->get_address_at_index(1);
    original_address = cached_state.stack_ptr->get_address_at_index(2);
    value = cached_state.stack_ptr->get_address_at_index(3);
    args_offset = cached_state.stack_ptr->get_address_at_index(4);
    args_size = cached_state.stack_ptr->get_address_at_index(5);
    ret_offset = cached_state.stack_ptr->get_address_at_index(6);
    ret_size = cached_state.stack_ptr->get_address_at_index(7);
    cached_state.stack_ptr->reduce_size(7);

    gas_t gas = uint256_get_uint64_t(gas_word);

    evm_word_t address = *original_address;
    // clean the address
    CuEVM::utils::evm_address_conversion(address);

    // new_context_ptr = new CuEVM::evm_call_context_t();
    new_context_ptr = memory_pool::get_call_context(current_context->depth);

    int32_t bytecode_offset = -1;

    new_context_ptr->initiate_values(current_context, gas, current_context->to, address, address, *value, OP_CALL,
                                     nullptr, 0, nullptr, 0, bytecode_offset, uint256_get_uint32_t(ret_offset),
                                     uint256_get_uint32_t(ret_size), current_context->static_env);

    CuEVM::gas_cost::access_account_cost(cached_state.gas_used, CuEVM::global_state_db_ptr, &address,
                                         current_context->snapshot_state);

    return generic_CALL(args_offset, args_size, current_context->memory_ptr, new_context_ptr, cached_state);
}

/**
 * The CALLCODE operation. gives the new evm call state
 * @param[in] arith The arithmetical environment.
 * @param[in] current_state The current state.
 * @param[out] new_state_ptr The new state pointer.
 * @return 0 if the operation is successful, otherwise the error code.
 */
__device__ int32_t CALLCODE(CuEVM::evm_call_context_t *current_context, CuEVM::evm_call_context_t *&new_context_ptr,
                            CuEVM::cached_evm_call_context &cached_state) {
    evm_word_t *gas_word, *original_address, *value, *args_offset, *args_size, *ret_offset, *ret_size;
    if (cached_state.stack_ptr->size() < 7) return ERROR_STACK_UNDERFLOW;
    gas_word = cached_state.stack_ptr->get_address_at_index(1);
    original_address = cached_state.stack_ptr->get_address_at_index(2);
    value = cached_state.stack_ptr->get_address_at_index(3);
    args_offset = cached_state.stack_ptr->get_address_at_index(4);
    args_size = cached_state.stack_ptr->get_address_at_index(5);
    ret_offset = cached_state.stack_ptr->get_address_at_index(6);
    ret_size = cached_state.stack_ptr->get_address_at_index(7);
    cached_state.stack_ptr->reduce_size(7);

    gas_t gas = uint256_get_uint64_t(gas_word);

    // clean the address
    evm_word_t address = *original_address;
    CuEVM::utils::evm_address_conversion(address);

    // get the account warm result first, get_code will warm up the account
    CuEVM::gas_cost::access_account_cost(cached_state.gas_used, CuEVM::global_state_db_ptr, &address,
                                         current_context->snapshot_state);

    uint32_t byte_code_size = 0;
    uint8_t *byte_code =
        CuEVM::global_state_db_ptr->get_code(byte_code_size, &address, current_context->snapshot_state);

    // new_context_ptr = new CuEVM::evm_call_context_t();
    new_context_ptr = memory_pool::get_call_context(current_context->depth);
    int32_t bytecode_offset = find_global_bytecode_offset(&address);
    if (uint256_cmp_word(&address, CuEVM::no_precompile_contracts) == -1)
        new_context_ptr->initiate_values(current_context, gas, current_context->to, address, current_context->to,
                                         *value, OP_CALLCODE, nullptr, 0, byte_code, byte_code_size, bytecode_offset,
                                         uint256_get_uint32_t(ret_offset), uint256_get_uint32_t(ret_size),
                                         current_context->static_env);
    else
        new_context_ptr->initiate_values(current_context, gas, current_context->to, current_context->to,
                                         current_context->to, *value, OP_CALLCODE, nullptr, 0, byte_code,
                                         byte_code_size, bytecode_offset, uint256_get_uint32_t(ret_offset),
                                         uint256_get_uint32_t(ret_size), current_context->static_env);

    return generic_CALL(args_offset, args_size, current_context->memory_ptr, new_context_ptr, cached_state);
}

/**
 * The DELEGATECALL operation. gives the new evm call state
 * @param[in] arith The arithmetical environment.
 * @param[in] current_state The current state.
 * @param[out] new_state_ptr The new state pointer.
 * @return 0 if the operation is successful, otherwise the error code.
 */
__device__ int32_t DELEGATECALL(CuEVM::evm_call_context_t *current_context, CuEVM::evm_call_context_t *&new_context_ptr,
                                CuEVM::cached_evm_call_context &cached_state) {
    evm_word_t *gas_word, *original_address, *args_offset, *args_size, *ret_offset, *ret_size;
    evm_word_t value = current_context->value;

    if (cached_state.stack_ptr->size() < 6) return ERROR_STACK_UNDERFLOW;
    gas_word = cached_state.stack_ptr->get_address_at_index(1);
    original_address = cached_state.stack_ptr->get_address_at_index(2);
    args_offset = cached_state.stack_ptr->get_address_at_index(3);
    args_size = cached_state.stack_ptr->get_address_at_index(4);
    ret_offset = cached_state.stack_ptr->get_address_at_index(5);
    ret_size = cached_state.stack_ptr->get_address_at_index(6);
    cached_state.stack_ptr->reduce_size(6);

    gas_t gas = uint256_get_uint64_t(gas_word);

    // clean the address
    evm_word_t address = *original_address;
    CuEVM::utils::evm_address_conversion(address);

    CuEVM::gas_cost::access_account_cost(cached_state.gas_used, CuEVM::global_state_db_ptr, &address,
                                         current_context->snapshot_state);

    uint32_t byte_code_size = 0;
    uint8_t *byte_code =
        CuEVM::global_state_db_ptr->get_code(byte_code_size, &address, current_context->snapshot_state);

    new_context_ptr = memory_pool::get_call_context(current_context->depth);
    int32_t bytecode_offset = find_global_bytecode_offset(&address);

    if (uint256_cmp_word(&address, CuEVM::no_precompile_contracts) == -1)
        new_context_ptr->initiate_values(current_context, gas, current_context->from, address, current_context->to,
                                         value, OP_DELEGATECALL, nullptr, 0, byte_code, byte_code_size, bytecode_offset,
                                         uint256_get_uint32_t(ret_offset), uint256_get_uint32_t(ret_size),
                                         current_context->static_env);
    else
        new_context_ptr->initiate_values(current_context, gas, current_context->from, current_context->to,
                                         current_context->to, value, OP_DELEGATECALL, nullptr, 0, byte_code,
                                         byte_code_size, bytecode_offset, uint256_get_uint32_t(ret_offset),
                                         uint256_get_uint32_t(ret_size), current_context->static_env);
    // printf("new context ptr\n");
    // new_context_ptr->print();

    return generic_CALL(args_offset, args_size, current_context->memory_ptr, new_context_ptr, cached_state);
}

/**
 * The STATICCALL operation. gives the new evm call state
 * @param[in] arith The arithmetical environment.
 * @param[in] current_state The current state.
 * @param[out] new_state_ptr The new state pointer.
 * @return 0 if the operation is successful, otherwise the error code.
 */
__device__ int32_t STATICCALL(CuEVM::evm_call_context_t *current_context, CuEVM::evm_call_context_t *&new_context_ptr,
                              CuEVM::cached_evm_call_context &cached_state) {
    evm_word_t *gas_word, *original_address, *args_offset, *args_size, *ret_offset, *ret_size;
    evm_word_t value = 0;
    if (cached_state.stack_ptr->size() < 6) return ERROR_STACK_UNDERFLOW;
    gas_word = cached_state.stack_ptr->get_address_at_index(1);
    original_address = cached_state.stack_ptr->get_address_at_index(2);
    args_offset = cached_state.stack_ptr->get_address_at_index(3);
    args_size = cached_state.stack_ptr->get_address_at_index(4);
    ret_offset = cached_state.stack_ptr->get_address_at_index(5);
    ret_size = cached_state.stack_ptr->get_address_at_index(6);
    cached_state.stack_ptr->reduce_size(6);

    gas_t gas = uint256_get_uint64_t(gas_word);

    // clean the address
    evm_word_t address = *original_address;
    CuEVM::utils::evm_address_conversion(address);

    new_context_ptr = memory_pool::get_call_context(current_context->depth);
    new_context_ptr->initiate_values(current_context, gas, current_context->to, address, address, value, OP_STATICCALL,
                                     nullptr, 0, nullptr, 0, -1, uint256_get_uint32_t(ret_offset),
                                     uint256_get_uint32_t(ret_size), true);
    // printf("new context ptr\n");
    // new_context_ptr->print();
    CuEVM::gas_cost::access_account_cost(cached_state.gas_used, CuEVM::global_state_db_ptr, &address,
                                         current_context->snapshot_state);

    return generic_CALL(args_offset, args_size, current_context->memory_ptr, new_context_ptr, cached_state);
}
/**
 * The CREATE2 operation. gives the new evm call state
 * @param[in] arith The arithmetical environment.
 * @param[in] current_state The current state.
 * @param[out] new_state_ptr The new state pointer.
 * @return 0 if the operation is successful, otherwise the error code.
 */
__device__ int32_t CREATE2(CuEVM::evm_call_context_t *current_context, CuEVM::evm_call_context_t *&new_context_ptr,
                           CuEVM::cached_evm_call_context &cached_state) {
    return generic_CREATE(current_context, new_context_ptr, OP_CREATE2, cached_state);
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
                          CuEVM::evm_call_context_t *call_state_ptr) {
    evm_word_t memory_offset, length;
    int32_t error_code = stack.pop(memory_offset);
    error_code |= stack.pop(length);

    CuEVM::gas_t memory_expansion_cost;
    uint32_t memory_offset_ui32 = uint256_get_uint32_t(&memory_offset);
    uint32_t length_ui32 = uint256_get_uint32_t(&length);
    if (uint256_cmp_word(&memory_offset, memory_offset_ui32) != 0 || uint256_cmp_word(&length, length_ui32) != 0) {
        return ERR_MEMORY_INVALID_OFFSET;
    }
    error_code |= CuEVM::gas_cost::memory_grow_cost(call_state_ptr->memory_ptr, memory_offset_ui32, length_ui32,
                                                    memory_expansion_cost, gas_used);

    error_code |= CuEVM::gas_cost::has_gas(gas_limit, gas_used);

    if (error_code == ERROR_SUCCESS) {
        call_state_ptr->set_parent_return_data(memory_offset_ui32, length_ui32);
        error_code = ERROR_RETURN;
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
                          CuEVM::evm_call_context_t *call_state_ptr) {
    evm_word_t memory_offset, length;
    int32_t error_code = stack.pop(memory_offset);
    error_code |= stack.pop(length);

    CuEVM::gas_t memory_expansion_cost;
    uint32_t memory_offset_ui32 = uint256_get_uint32_t(&memory_offset);
    uint32_t length_ui32 = uint256_get_uint32_t(&length);
    // if memory offset is overflow and length is not 0, then return error
    if (uint256_cmp_word(&memory_offset, memory_offset_ui32) != 0 && uint256_cmp_word(&length, 0) != 0) {
        return ERR_MEMORY_INVALID_OFFSET;
    }
    error_code |= CuEVM::gas_cost::memory_grow_cost(call_state_ptr->memory_ptr, memory_offset_ui32, length_ui32,
                                                    memory_expansion_cost, gas_used);

    error_code |= CuEVM::gas_cost::has_gas(gas_limit, gas_used);

    if (error_code == ERROR_SUCCESS) {
        call_state_ptr->set_parent_return_data(memory_offset_ui32, length_ui32);
        error_code = ERROR_REVERT;
    }
    return error_code;
}

/**
 * The INVALID operation.
 * @return The error code.
 */
__device__ int32_t INVALID() { return ERROR_INVALID_OPCODE; }

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
                                CuEVM::evm_call_context_t *call_context
#ifdef BUILD_GO_LIBRARY
                                ,
                                CuEVM::simplified_trace_data *trace_data
#endif
) {
    int32_t error_code = ERROR_SUCCESS;
    if (call_context->static_env) {
        error_code = ERROR_STATIC_CALL_CONTEXT_SELFDESTRUCT;
    } else {
        evm_word_t recipient;
        error_code |= stack.pop(recipient);
        gas_used += GAS_SELFDESTRUCT;
        // custom logic, cannot use access_account_cost (no warm cost)
        if (!global_state_db_ptr->is_warm_account(&recipient, call_context->snapshot_state))
            gas_used += GAS_COLD_ACCOUNT_ACCESS;

        evm_word_t *sender_balance = global_state_db_ptr->get_balance(&call_context->to);

        if (!uint256_is_zero(sender_balance)) {
            if (global_state_db_ptr->is_empty_account(&recipient)) {
                gas_used += GAS_NEW_ACCOUNT;
            }
        }

        error_code |= CuEVM::gas_cost::has_gas(gas_limit, gas_used);
        if (error_code == ERROR_SUCCESS) {
            global_state_db_ptr->increase_balance(&recipient, sender_balance, call_context->snapshot_state);
            // sender_balance->set_zero();
            global_state_db_ptr->deduct_balance(&call_context->to, sender_balance, call_context->snapshot_state);
            // receiver = self => 0 balance
            if (call_context->depth > 1) call_context->parent->dynamic_ret_size = 0;
            error_code |= ERROR_RETURN;
        }
#ifdef BUILD_GO_LIBRARY
        trace_data->selfdestruct_oracle(call_context->pc);
#endif
    }
    return error_code;
}
}  // namespace CuEVM::operations
