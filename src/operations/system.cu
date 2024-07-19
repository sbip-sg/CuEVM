#include "../include/operations/system.cuh"

#include "../include/utils/error_codes.cuh"
#include "../include/utils/opcodes.cuh"
#include "../include/gas_cost.cuh"

namespace cuEVM::operations
{
    /**
     * Make a generic call.
     * @param[in] arith The arithmetical environment.
     * @param[in] current_state The current state.
     * @param[out] new_state_ptr The new state pointer.
     * @return 0 if the operation is successful, otherwise the error code.
     */
    __host__ __device__ int32_t generic_CALL(
        ArithEnv &arith,
        const bn_t args_offset,
        const bn_t args_size,
        cuEVM::state::AccessState &access_state,
        cuEVM::evm_call_state_t &current_state,
        cuEVM::evm_call_state_t* &new_state_ptr)
    {
        // try to send value in call
        bn_t value;
        new_state_ptr->message_ptr->get_value(arith, value);
        int32_t error_code = (
            (new_state_ptr->message_ptr->get_static_env() &&
            (cgbn_compare_ui32(arith.env, value, 0) != 0) &&
            (new_state_ptr->message_ptr->get_call_type() == OP_CALL)) ?
            ERROR_STATIC_CALL_CONTEXT_CALL_VALUE :
            ERROR_SUCCESS
        );
        
        // charge the gas for the call

        // memory call data
        bn_t memory_expansion_cost_args;
        error_code |= cuEVM::gas_cost::memory_grow_cost(
            arith,
            *current_state.memory_ptr,
            args_offset,
            args_size,
            memory_expansion_cost_args,
            current_state.gas_used);

        // memory return data
        bn_t ret_offset, ret_size;
        new_state_ptr->message_ptr->get_return_data_offset(arith, ret_offset);
        new_state_ptr->message_ptr->get_return_data_size(arith, ret_size);
        bn_t memory_expansion_cost_ret;
        error_code |= cuEVM::gas_cost::memory_grow_cost(
            arith,
            *current_state.memory_ptr,
            ret_offset,
            ret_size,
            memory_expansion_cost_ret,
            current_state.gas_used);

        // adress warm call
        bn_t contract_address;
        new_state_ptr->message_ptr->get_contract_address(arith, contract_address);
        cuEVM::gas_cost::access_account_cost(
            arith,
            current_state.gas_used,
            access_state,
            contract_address);
        // positive value call cost (except delegate call)
        // empty account call cost
        bn_t gas_stippend;
        cgbn_set_ui32(arith.env, gas_stippend, 0);
        if (new_state_ptr->message_ptr->get_call_type() != OP_DELEGATECALL)
        {
            if (cgbn_compare_ui32(arith.env, value, 0) > 0)
            {
                cgbn_add_ui32(arith.env, current_state.gas_used, current_state.gas_used, GAS_CALL_VALUE);
                cgbn_set_ui32(arith.env, gas_stippend, GAS_CALL_STIPEND);
                // If the empty account is called
                // only for call opcode
                if (
                    new_state_ptr->touch_state.is_empty_account(arith, contract_address) &&
                    (new_state_ptr->message_ptr->get_call_type() == OP_CALL)
                )
                {
                    cgbn_add_ui32(arith.env, current_state.gas_used, current_state.gas_used, GAS_NEW_ACCOUNT);
                };
            }
        }

        if (arith.has_gas(gas_limit, gas_used, error_code))
        {
            bn_t gas;
            new_message.get_gas_limit(gas);

            // gas capped = (63/64) * gas_left
            bn_t gas_capped;
            arith.max_gas_call(gas_capped, gas_limit, gas_used);

            if (cgbn_compare(arith._env, gas, gas_capped) > 0)
            {
                cgbn_set(arith._env, gas, gas_capped);
            }

            // add to gas used the sent gas
            cgbn_add(arith._env, gas_used, gas_used, gas);

            // add the call stippend
            cgbn_add(arith._env, gas, gas, gas_stippend);

            // set the new gas limit
            new_message.set_gas_limit(gas);

            // set the byte code
            account_t *contract;
            contract = touch_state.get_account(contract_address, READ_CODE);

            new_message.set_byte_code(
                contract->bytecode,
                contract->code_size);

            uint8_t *call_data;
            size_t call_data_size;
            call_data = memory.get(
                args_offset,
                args_size,
                error_code);
            arith.size_t_from_cgbn(call_data_size, args_size);

            new_message.set_data(call_data, call_data_size);

            if (valid_CALL(arith, new_message, touch_state))
            {
                // new message done
                // call the child
                evm.child_CALL(
                    error_code,
                    new_message);
            }
            else
            {
                bn_t child_success;
                cgbn_set_ui32(arith._env, child_success, 0);
                stack.push(child_success, error_code);
                return_data.set(
                    NULL,
                    0);
                // TODO: verify better if contains the GAS STIPPEND
                cgbn_sub(arith._env, gas_used, gas_used, gas);
                delete &new_message;
            }
        }
        else
        {
            delete &new_message;
        }
    }

    /**
     * Make a generic create call.
     * @param[in] arith The arithmetical environment.
     * @param[in] current_state The current state.
     * @param[out] new_state_ptr The new state pointer.
     * @return 0 if the operation is successful, otherwise the error code.
     */
    __host__ __device__ int32_t generic_CREATE(
        ArithEnv &arith,
        cuEVM::evm_call_state_t &current_state,
        cuEVM::evm_call_state_t* &new_state_ptr)
    {

        #ifdef ONLY_CPU
        printf("CREATE: gas_used: %d\n", cgbn_get_ui32(arith._env, gas_used));
        #endif
        if (message.get_static_env())
        {
            error_code = ERROR_STATIC_CALL_CONTEXT_CREATE;
            delete &new_message;
        }
        else if (cgbn_compare_ui32(arith._env, args_size, MAX_INIT_CODE_SIZE) >= 0)
        {
            // EIP-3860
            error_code = ERROR_CREATE_INIT_CODE_SIZE_EXCEEDED;
            delete &new_message;
        }
        else
        {
            #ifdef ONLY_CPU
            printf("CREATE: gas_used: %d\n", cgbn_get_ui32(arith._env, gas_used));
            #endif
            // set the init code
            SHARED_MEMORY cuEVM::byte_array_t initialisation_code;
            arith.size_t_from_cgbn(initialisation_code.size, args_size);
            initialisation_code.data = memory.get(
                args_offset,
                args_size,
                error_code);
            new_message.set_byte_code(
                initialisation_code.data,
                initialisation_code.size);

            // // set the gas limit
            bn_t gas_capped;
            arith.max_gas_call(gas_capped, gas_limit, gas_used);
            new_message.set_gas_limit(gas_capped);

            // // add to gas used
            cgbn_add(arith._env, gas_used, gas_used, gas_capped);

            #ifdef ONLY_CPU
            bn_t pr_gas;
            cgbn_sub(arith._env, pr_gas, gas_limit, gas_used);
            printf("GENERIC_CREATE parent unused gas: %d\n", cgbn_get_ui32(arith._env, pr_gas));
            #endif

            // warm up the contract address
            bn_t contract_address;
            new_message.get_recipient(contract_address);
            account_t *account = touch_state.get_account(contract_address, READ_NONE);

            // setup return offset to null
            bn_t ret_offset, ret_size;
            cgbn_set_ui32(arith._env, ret_offset, 0);
            cgbn_set_ui32(arith._env, ret_size, 0);
            new_message.set_return_data_offset(ret_offset);
            new_message.set_return_data_size(ret_size);

            if (valid_CREATE(arith, new_message, touch_state))
            {
                // increase the nonce if the sender is a contract
                // TODO: seems like an akward think to do
                // why in the parent and not in the child the nonce
                // if the contract deployment fails the nonce is still
                // increased?
                #ifdef ONLY_CPU
                printf("CREATE: gas_used: %d\n", cgbn_get_ui32(arith._env, gas_used));
                #endif
                bn_t sender;
                new_message.get_sender(sender);
                if (touch_state.is_contract(sender))
                {
                    bn_t sender_nonce;
                    touch_state.get_account_nonce(sender, sender_nonce);
                    cgbn_add_ui32(arith._env, sender_nonce, sender_nonce, 1);
                    touch_state.set_account_nonce(sender, sender_nonce);
                }
                // new message done
                // call the child
                evm.child_CALL(
                    error_code,
                    new_message);
            }
            else
            {
                bn_t child_success;
                cgbn_set_ui32(arith._env, child_success, 0);
                stack.push(child_success, error_code);
                // cgbn_sub(arith._env, gas_used, gas_used, gas_capped);
                return_data.set(
                    NULL,
                    0);
                delete &new_message;
            }
        }
    }

    /**
     * The STOP operation.
     * @param[out] return_data The return data.
     * @return return error code.
     */
    __host__ __device__ int32_t STOP(
        cuEVM::evm_return_data_t &return_data)
    {
        return_data = cuEVM::evm_return_data_t();
        return ERROR_RETURN;
    }

    /**
     * The CREATE operation. gives the new evm call state
     * @param[in] arith The arithmetical environment.
     * @param[in] current_state The current state.
     * @param[out] new_state_ptr The new state pointer.
     * @return 0 if the operation is successful, otherwise the error code.
     */
    __host__ __device__ int32_t CREATE(
        ArithEnv &arith,
        cuEVM::evm_call_state_t &current_state,
        cuEVM::evm_call_state_t* &new_state_ptr)
    {
        bn_t value, memory_offset, length;
        stack.pop(value, error_code);
        stack.pop(memory_offset, error_code);
        stack.pop(length, error_code);

        // // create cost
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_CREATE);

        // compute the memory cost
        memory.grow_cost(
            memory_offset,
            length,
            gas_used,
            error_code);

            #ifdef ONLY_CPU
            printf("CREATE: gas_used: %d\n", cgbn_get_ui32(arith._env, gas_used));
            #endif
        // compute the initcode gas cost
        arith.initcode_cost(
            gas_used,
            length);

            #ifdef ONLY_CPU
            printf("CREATE: gas_used: %d\n", cgbn_get_ui32(arith._env, gas_used));
            #endif
        if (arith.has_gas(gas_limit, gas_used, error_code))
        {
            #ifdef ONLY_CPU
            printf("CREATE: gas_used: %d\n", cgbn_get_ui32(arith._env, gas_used));
            #endif
            bn_t sender_address;
            message.get_recipient(sender_address); // I_{a}
            bn_t sender_nonce;
            touch_state.get_account_nonce(sender_address, sender_nonce);

            // compute the address
            bn_t address;
            message_t::get_create_contract_address(
                arith,
                address,
                sender_address,
                sender_nonce);

            message_t *new_message = new message_t(
                arith,
                sender_address,
                address,
                address,
                gas_limit,
                value,
                message.get_depth() + 1,
                opcode,
                address,
                NULL,
                0,
                NULL,
                0,
                memory_offset,
                length,
                message.get_static_env());

            generic_CREATE(
                arith,
                gas_limit,
                gas_used,
                error_code,
                stack,
                message,
                memory,
                touch_state,
                evm,
                *new_message,
                memory_offset,
                length,
                return_data);

            pc = pc + 1;
        }
    }

    /**
     * The CALL operation. gives the new evm call state
     * @param[in] arith The arithmetical environment.
     * @param[in] current_state The current state.
     * @param[out] new_state_ptr The new state pointer.
     * @return 0 if the operation is successful, otherwise the error code.
     */
    __host__ __device__ int32_t CALL(
        ArithEnv &arith,
        cuEVM::evm_call_state_t &current_state,
        cuEVM::evm_call_state_t* &new_state_ptr)
    {
        bn_t gas, address, value, args_offset, args_size, ret_offset, ret_size;
        stack.pop(gas, error_code);
        stack.pop(address, error_code);
        stack.pop(value, error_code);
        stack.pop(args_offset, error_code);
        stack.pop(args_size, error_code);
        stack.pop(ret_offset, error_code);
        stack.pop(ret_size, error_code);

        if (error_code == ERR_NONE)
        {
            // clean the address
            arith.address_conversion(address);
            bn_t sender;
            message.get_recipient(sender); // I_{a}
            bn_t recipient;
            cgbn_set(arith._env, recipient, address); // t
            bn_t contract_address;
            cgbn_set(arith._env, contract_address, address); // t
            bn_t storage_address;
            cgbn_set(arith._env, storage_address, address); // t

            message_t *new_message = new message_t(
                arith,
                sender,
                recipient,
                contract_address,
                gas,
                value,
                message.get_depth() + 1,
                opcode,
                storage_address,
                NULL,
                0,
                NULL,
                0,
                ret_offset,
                ret_size,
                message.get_static_env());

            generic_CALL(
                arith,
                gas_limit,
                gas_used,
                error_code,
                stack,
                message,
                memory,
                touch_state,
                evm,
                *new_message,
                args_offset,
                args_size,
                return_data);

            pc = pc + 1;
        }
    }

    /**
     * The CALLCODE operation. gives the new evm call state
     * @param[in] arith The arithmetical environment.
     * @param[in] current_state The current state.
     * @param[out] new_state_ptr The new state pointer.
     * @return 0 if the operation is successful, otherwise the error code.
    */
    __host__ __device__ int32_t CALLCODE(
        ArithEnv &arith,
        cuEVM::evm_call_state_t &current_state,
        cuEVM::evm_call_state_t* &new_state_ptr)
    {
        bn_t gas, address, value, args_offset, args_size, ret_offset, ret_size;
        stack.pop(gas, error_code);
        stack.pop(address, error_code);
        stack.pop(value, error_code);
        stack.pop(args_offset, error_code);
        stack.pop(args_size, error_code);
        stack.pop(ret_offset, error_code);
        stack.pop(ret_size, error_code);

        if (error_code == ERR_NONE)
        {
            // clean the address
            arith.address_conversion(address);
            bn_t sender;
            message.get_recipient(sender); // I_{a}
            bn_t recipient;
            cgbn_set(arith._env, recipient, sender); // I_{a}
            bn_t contract_address;
            cgbn_set(arith._env, contract_address, address); // t
            bn_t storage_address;
            cgbn_set(arith._env, storage_address, sender); // I_{a}

            message_t *new_message = new message_t(
                arith,
                sender,
                recipient,
                contract_address,
                gas,
                value,
                message.get_depth() + 1,
                opcode,
                storage_address,
                NULL,
                0,
                NULL,
                0,
                ret_offset,
                ret_size,
                message.get_static_env());

            generic_CALL(
                arith,
                gas_limit,
                gas_used,
                error_code,
                stack,
                message,
                memory,
                touch_state,
                evm,
                *new_message,
                args_offset,
                args_size,
                return_data);

            pc = pc + 1;
        }
    }

    /**
     * The RETURN operation.
     * @param[in] arith The arithmetical environment.
     * @param[in] gas_limit The gas limit.
     * @param[inout] gas_used The gas used.
     * @param[in] stack The stack.
     * @param[in] memory The memory.
     * @param[out] return_data The return data.
     * @return 0 if the operation is successful, otherwise the error code.
    */
    __host__ __device__ int32_t RETURN(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        cuEVM::evm_stack_t &stack,
        cuEVM::evm_memory_t &memory,
        cuEVM::evm_return_data_t &return_data)
    {
        bn_t memory_offset, length;
        stack.pop(memory_offset, error_code);
        stack.pop(length, error_code);

        // TODO addback dynamic cost from sub execution

        if (error_code == ERR_NONE)
        {
            memory.grow_cost(
                memory_offset,
                length,
                gas_used,
                error_code);

            if (arith.has_gas(gas_limit, gas_used, error_code))
            {
                uint8_t *data;
                size_t data_size;
                data = memory.get(
                    memory_offset,
                    length,
                    error_code);
                arith.size_t_from_cgbn(data_size, length);

                if (error_code == ERR_NONE)
                {
                    return_data.set(
                        data,
                        data_size);
                    error_code = ERR_RETURN;
                }
            }
        }
    }

    /**
     * The DELEGATECALL operation. gives the new evm call state
     * @param[in] arith The arithmetical environment.
     * @param[in] current_state The current state.
     * @param[out] new_state_ptr The new state pointer.
     * @return 0 if the operation is successful, otherwise the error code.
    */
    __host__ __device__ int32_t DELEGATECALL(
        ArithEnv &arith,
        cuEVM::evm_call_state_t &current_state,
        cuEVM::evm_call_state_t* &new_state_ptr)
    {
        bn_t gas, address, value, args_offset, args_size, ret_offset, ret_size;
        stack.pop(gas, error_code);
        stack.pop(address, error_code);
        message.get_value(value);
        stack.pop(args_offset, error_code);
        stack.pop(args_size, error_code);
        stack.pop(ret_offset, error_code);
        stack.pop(ret_size, error_code);

        if (error_code == ERR_NONE)
        {
            // clean the address
            arith.address_conversion(address);
            bn_t sender;
            message.get_sender(sender); // keep the message call sender I_{s}
            bn_t recipient;
            message.get_recipient(recipient); // I_{a}
            bn_t contract_address;
            cgbn_set(arith._env, contract_address, address); // t
            bn_t storage_address;
            message.get_recipient(storage_address); // I_{a}

            message_t *new_message = new message_t(
                arith,
                sender,
                recipient,
                contract_address,
                gas,
                value,
                message.get_depth() + 1,
                opcode,
                storage_address,
                NULL,
                0,
                NULL,
                0,
                ret_offset,
                ret_size,
                message.get_static_env());

            generic_CALL(
                arith,
                gas_limit,
                gas_used,
                error_code,
                stack,
                message,
                memory,
                touch_state,
                evm,
                *new_message,
                args_offset,
                args_size,
                return_data);

            pc = pc + 1;
        }
    }

    /**
     * The CREATE2 operation. gives the new evm call state
     * @param[in] arith The arithmetical environment.
     * @param[in] current_state The current state.
     * @param[out] new_state_ptr The new state pointer.
     * @return 0 if the operation is successful, otherwise the error code.
     */
    __host__ __device__ int32_t CREATE2(
        ArithEnv &arith,
        cuEVM::evm_call_state_t &current_state,
        cuEVM::evm_call_state_t* &new_state_ptr)
    {
        bn_t value, memory_offset, length, salt;
        stack.pop(value, error_code);
        stack.pop(memory_offset, error_code);
        stack.pop(length, error_code);
        stack.pop(salt, error_code);

        // create cost
        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_CREATE);

        // compute the keccak gas cost
        arith.keccak_cost(
            gas_used,
            length);

        // compute the memory cost
        memory.grow_cost(
            memory_offset,
            length,
            gas_used,
            error_code);

        // compute the initcode gas cost
        arith.initcode_cost(
            gas_used,
            length);

        if (arith.has_gas(gas_limit, gas_used, error_code))
        {
            SHARED_MEMORY cuEVM::byte_array_t initialisation_code;

            arith.size_t_from_cgbn(initialisation_code.size, length);
            initialisation_code.data = memory.get(
                memory_offset,
                length,
                error_code);

            bn_t sender_address;
            message.get_recipient(sender_address); // I_{a}

            // compute the address
            bn_t address;
            message_t::get_create2_contract_address(
                arith,
                address,
                sender_address,
                salt,
                initialisation_code);

            // create the message
            message_t *new_message = new message_t(
                arith,
                sender_address,
                address,
                address,
                gas_limit,
                value,
                message.get_depth() + 1,
                opcode,
                address,
                NULL,
                0,
                NULL,
                0,
                memory_offset,
                length,
                message.get_static_env());

            generic_CREATE(
                arith,
                gas_limit,
                gas_used,
                error_code,
                stack,
                message,
                memory,
                touch_state,
                evm,
                *new_message,
                memory_offset,
                length,
                return_data);

            pc = pc + 1;
        }
    }

    /**
     * The STATICCALL operation. gives the new evm call state
     * @param[in] arith The arithmetical environment.
     * @param[in] current_state The current state.
     * @param[out] new_state_ptr The new state pointer.
     * @return 0 if the operation is successful, otherwise the error code.
    */
    __host__ __device__ int32_t STATICCALL(
        ArithEnv &arith,
        cuEVM::evm_call_state_t &current_state,
        cuEVM::evm_call_state_t* &new_state_ptr)
    {
        bn_t gas, address, value, args_offset, args_size, ret_offset, ret_size;
        stack.pop(gas, error_code);
        stack.pop(address, error_code);
        cgbn_set_ui32(arith._env, value, 0);
        stack.pop(args_offset, error_code);
        stack.pop(args_size, error_code);
        stack.pop(ret_offset, error_code);
        stack.pop(ret_size, error_code);

        if (error_code == ERR_NONE)
        {
            // clean the address
            arith.address_conversion(address);
            bn_t sender;
            message.get_recipient(sender); // I_{a}
            bn_t recipient;
            cgbn_set(arith._env, recipient, address); // t
            bn_t contract_address;
            cgbn_set(arith._env, contract_address, address); // t
            bn_t storage_address;
            cgbn_set(arith._env, storage_address, address); // t

            message_t *new_message = new message_t(
                arith,
                sender,
                recipient,
                contract_address,
                gas,
                value,
                message.get_depth() + 1,
                opcode,
                storage_address,
                NULL,
                0,
                NULL,
                0,
                ret_offset,
                ret_size,
                1);

            generic_CALL(
                arith,
                gas_limit,
                gas_used,
                error_code,
                stack,
                message,
                memory,
                touch_state,
                evm,
                *new_message,
                args_offset,
                args_size,
                return_data);

            pc = pc + 1;
        }
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
    __host__ __device__ int32_t REVERT(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        cuEVM::evm_stack_t &stack,
        cuEVM::evm_memory_t &memory,
        cuEVM::evm_return_data_t &return_data)
    {
        bn_t memory_offset, length;
        stack.pop(memory_offset, error_code);
        stack.pop(length, error_code);

        if (error_code == ERR_NONE)
        {
            memory.grow_cost(
                memory_offset,
                length,
                gas_used,
                error_code);

            if (arith.has_gas(gas_limit, gas_used, error_code))
            {
                uint8_t *data;
                size_t data_size;
                data = memory.get(
                    memory_offset,
                    length,
                    error_code);
                arith.size_t_from_cgbn(data_size, length);

                if (error_code == ERR_NONE)
                {
                    return_data.set(
                        data,
                        data_size);

                    error_code = ERR_REVERT;
                }
            }
        }
    }

    /**
     * The INVALID operation.
     * @return The error code.
    */
    __host__ __device__ int32_t INVALID()
    {
        return ERROR_NOT_IMPLEMENTED;
    }

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
    __host__ __device__ int32_t SELFDESTRUCT(
        ArithEnv &arith,
        const bn_t &gas_limit,
        bn_t &gas_used,
        cuEVM::evm_stack_t &stack,
        cuEVM::evm_message_call_t &message,
        cuEVM::state::TouchState &touch_state,
        cuEVM::evm_return_data_t &return_data)
    {
        if (message.get_static_env())
        {
            error_code = ERROR_STATIC_CALL_CONTEXT_SELFDESTRUCT;
        }
        else
        {
            bn_t recipient;
            stack.pop(recipient, error_code);

            cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_SELFDESTRUCT);

            bn_t dummy_gas;
            cgbn_set_ui32(arith._env, dummy_gas, 0);
            touch_state.charge_gas_access_account(
                recipient,
                dummy_gas);
            if (cgbn_compare_ui32(arith._env, dummy_gas, GAS_WARM_ACCESS) != 0)
            {
                cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_COLD_ACCOUNT_ACCESS);
            }

            bn_t sender;
            message.get_recipient(sender); // I_{a}
            bn_t sender_balance;
            touch_state.get_account_balance(sender, sender_balance);

            if (cgbn_compare_ui32(arith._env, sender_balance, 0) > 0)
            {
                if (touch_state.is_empty_account(recipient))
                {
                    cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_NEW_ACCOUNT);
                }
            }

            if (arith.has_gas(gas_limit, gas_used, error_code))
            {
                bn_t recipient_balance;
                touch_state.get_account_balance(recipient, recipient_balance);

                if (cgbn_compare(arith._env, recipient, sender) != 0)
                {
                    cgbn_add(arith._env, recipient_balance, recipient_balance, sender_balance);
                    touch_state.set_account_balance(recipient, recipient_balance);
                }
                cgbn_set_ui32(arith._env, sender_balance, 0);
                touch_state.set_account_balance(sender, sender_balance);
                // TODO: delete or not the storage/code?
                touch_state.delete_account(sender);

                return_data.set(
                    NULL,
                    0);
                error_code = ERR_RETURN;
            }
        }
    }
} // namespace cuEVM::operation