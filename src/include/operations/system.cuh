
    /**
     * The system operations class.
     * It contains the implementation of the system operations.
     * 00:
     * - STOP
     * f0s: System operations:
     * - CREATE
     * - CALL
     * - CALLCODE
     * - RETURN
     * - DELEGATECALL
     * - CREATE2
     * - STATICCALL
     * - REVERT
     * - INVALID
     * - SELFDESTRUCT
     */
    namespace cuEVM::operations
    {
        /**
         * Get if a message call is valid.
         * @param[in] arith The arithmetical environment.
         * @param[in] message The message call.
         * @param[in] touch_state The touch state.
         * @return 1 if the message call is valid, 0 otherwise.
         */
        __host__ __device__ int32_t valid_CALL(
            ArithEnv &arith,
            message_t &message,
            touch_state_t &touch_state)
        {
            bn_t sender, value;
            bn_t sender_balance;
            // bn_t receiver
            // bn_t receiver_balance;
            uint8_t call_type;
            uint32_t depth;
            message.get_sender(sender);
            // message.get_recipient(receiver);
            message.get_value(value);
            call_type = message.get_call_type();
            depth = message.get_depth();

            // verify depth
            if (depth >= MAX_DEPTH)
            {
                // error_code = ERROR_MESSAGE_CALL_DEPTH_EXCEEDED;
                return 0;
            }

            // verify if the value can be transfered
            // if the sender has enough balance
            // value>0
            //(cgbn_compare(arith._env, sender, receiver) != 0) &&   // sender != receiver matter only on transfer
            if ((cgbn_compare_ui32(arith._env, value, 0) > 0) &&
                (call_type != OP_DELEGATECALL) // no delegatecall
            )
            {
                touch_state.get_account_balance(sender, sender_balance);
                // touch_state.get_account_balance(receiver, receiver_balance);
                //  verify the balance before transfer
                if (cgbn_compare(arith._env, sender_balance, value) < 0)
                {
                    // error_code = ERROR_MESSAGE_CALL_SENDER_BALANCE;
                    return 0;
                }
            }

            return 1;
        }

        /**
         * Get if a message create call is valid.
         * @param[in] arith The arithmetical environment.
         * @param[in] message The message call.
         * @param[in] touch_state The touch state.
         * @return 1 if the message create call is valid, 0 otherwise.
         */
        __host__ __device__ int32_t valid_CREATE(
            ArithEnv &arith,
            message_t &message,
            touch_state_t &touch_state)
        {
            bn_t sender;
            message.get_sender(sender);
            if (touch_state.is_contract(sender))
            {
                bn_t sender_nonce;
                touch_state.get_account_nonce(sender, sender_nonce);
                cgbn_add_ui32(arith._env, sender_nonce, sender_nonce, 1);
                size_t nonce;
                if (arith.uint64_t_from_cgbn(nonce, sender_nonce))
                {
                    // error_code = ERROR_MESSAGE_CALL_CREATE_NONCE_EXCEEDED;
                    return 0;
                }
            }

            return valid_CALL(arith, message, touch_state);
        }

        /**
         * Make a generic call.
         * @param[in] arith The arithmetical environment.
         * @param[in] gas_limit The gas limit.
         * @param[inout] gas_used The gas used.
         * @param[out] error_code The error code.
         * @param[inout] stack The stack.
         * @param[in] message The current context message call.
         * @param[in] memory The memory.
         * @param[in] touch_state The touch state.
         * @param[out] evm The evm.
         * @param[in] new_message The new message call.
         * @param[in] args_offset The message offset for call data.
         * @param[in] args_size The message size for call data.
         * @param[out] return_data The return data.
         */
        __host__ __device__ int32_t generic_CALL(
            ArithEnv &arith,
            bn_t &gas_limit,
            bn_t &gas_used,
            uint32_t &error_code,
            cuEVM::stack::EVMStack &stack,
            message_t &message,
            cuEVM::memory::EVMMemory &memory,
            touch_state_t &touch_state,
            evm_t &evm,
            message_t &new_message,
            bn_t &args_offset,
            bn_t &args_size,
            cuEVM::EVMReturnData &return_data)
        {
            // try to send value in call
            bn_t value;
            new_message.get_value(value);
            if (message.get_static_env())
            {
                if (
                    (cgbn_compare_ui32(arith._env, value, 0) != 0) &&
                    (new_message.get_call_type() == OP_CALL) // TODO: akward that is just CALL
                )
                {
                    error_code = ERROR_STATIC_CALL_CONTEXT_CALL_VALUE;
                }
            }

            // charge the gas for the call

            // memory call data
            memory.grow_cost(
                args_offset,
                args_size,
                gas_used,
                error_code);

            // memory return data
            bn_t ret_offset, ret_size;
            new_message.get_return_data_offset(ret_offset);
            new_message.get_return_data_size(ret_size);
            memory.grow_cost(
                ret_offset,
                ret_size,
                gas_used,
                error_code);

            // adress warm call
            bn_t contract_address;
            new_message.get_contract_address(contract_address);
            touch_state.charge_gas_access_account(
                contract_address,
                gas_used);

            // positive value call cost (except delegate call)
            // empty account call cost
            bn_t gas_stippend;
            cgbn_set_ui32(arith._env, gas_stippend, 0);
            if (new_message.get_call_type() != OP_DELEGATECALL)
            {
                if (cgbn_compare_ui32(arith._env, value, 0) > 0)
                {
                    cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_CALL_VALUE);
                    cgbn_set_ui32(arith._env, gas_stippend, GAS_CALL_STIPEND);
                    // If the empty account is called
                    // only for call opcode
                    if (
                        touch_state.is_empty_account(contract_address) &&
                        (new_message.get_call_type() == OP_CALL)
                    )
                    {
                        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_NEW_ACCOUNT);
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
         * @param[in] gas_limit The gas limit.
         * @param[inout] gas_used The gas used.
         * @param[out] error_code The error code.
         * @param[inout] stack The stack.
         * @param[in] message The current context message call.
         * @param[in] memory The memory.
         * @param[in] touch_state The touch state.
         * @param[out] evm The evm.
         * @param[in] new_message The new message call.
         * @param[in] args_offset The message offset for init code.
         * @param[in] args_size The message size for init code.
         * @param[out] return_data The return data.
         */
        __host__ __device__ int32_t generic_CREATE(
            ArithEnv &arith,
            bn_t &gas_limit,
            bn_t &gas_used,
            uint32_t &error_code,
            cuEVM::stack::EVMStack &stack,
            message_t &message,
            cuEVM::memory::EVMMemory &memory,
            touch_state_t &touch_state,
            evm_t &evm,
            message_t &new_message,
            bn_t &args_offset,
            bn_t &args_size,
            cuEVM::EVMReturnData &return_data)
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
         * @param[out] error_code The error code.
         */
        __host__ __device__ int32_t operation_STOP(
            cuEVM::EVMReturnData &return_data,
            uint32_t &error_code)
        {
            return_data.set(
                NULL,
                0);
            error_code = ERR_RETURN;
        }

        /**
         * The CREATE operation.
         * @param[in] arith The arithmetical environment.
         * @param[in] gas_limit The gas limit.
         * @param[inout] gas_used The gas used.
         * @param[out] error_code The error code.
         * @param[inout] pc The program counter.
         * @param[inout] stack The stack.
         * @param[in] message The current context message call.
         * @param[in] memory The memory.
         * @param[in] touch_state The touch state.
         * @param[in] opcode The operation opcode
         * @param[out] evm The evm.
         * @param[out] return_data The return data.
         */
        __host__ __device__ int32_t operation_CREATE(
            ArithEnv &arith,
            bn_t &gas_limit,
            bn_t &gas_used,
            uint32_t &error_code,
            uint32_t &pc,
            cuEVM::stack::EVMStack &stack,
            message_t &message,
            cuEVM::memory::EVMMemory &memory,
            touch_state_t &touch_state,
            uint8_t &opcode,
            evm_t &evm,
            cuEVM::EVMReturnData &return_data)
        {
            bn_t value, memory_offset, length;
            stack.pop(value, error_code);
            stack.pop(memory_offset, error_code);
            stack.pop(length, error_code);

            // // create cost
            cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_CREATE);

            #ifdef ONLY_CPU
            printf("CREATE: gas_used: %d\n", cgbn_get_ui32(arith._env, gas_used));
            printf("CREATE: memory_offset: %d\n", cgbn_get_ui32(arith._env, memory_offset));
            printf("CREATE: length: %d\n", cgbn_get_ui32(arith._env, length));
            #endif

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
         * The CALL operation.
         * @param[in] arith The arithmetical environment.
         * @param[in] gas_limit The gas limit.
         * @param[inout] gas_used The gas used.
         * @param[out] error_code The error code.
         * @param[inout] pc The program counter.
         * @param[inout] stack The stack.
         * @param[in] message The current context message call.
         * @param[in] memory The memory.
         * @param[inout] touch_state The touch state.
         * @param[in] opcode The operation opcode
         * @param[out] evm The evm.
         * @param[out] return_data The return data.
         */
        __host__ __device__ int32_t operation_CALL(
            ArithEnv &arith,
            bn_t &gas_limit,
            bn_t &gas_used,
            uint32_t &error_code,
            uint32_t &pc,
            cuEVM::stack::EVMStack &stack,
            message_t &message,
            cuEVM::memory::EVMMemory &memory,
            touch_state_t &touch_state,
            uint8_t &opcode,
            evm_t &evm,
            cuEVM::EVMReturnData &return_data)
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
         * The CALLCODE operation.
         * @param[in] arith The arithmetical environment.
         * @param[in] gas_limit The gas limit.
         * @param[inout] gas_used The gas used.
         * @param[out] error_code The error code.
         * @param[inout] pc The program counter.
         * @param[inout] stack The stack.
         * @param[in] message The current context message call.
         * @param[in] memory The memory.
         * @param[inout] touch_state The touch state.
         * @param[in] opcode The operation opcode
         * @param[out] evm The evm.
         * @param[out] return_data The return data.
        */
        __host__ __device__ int32_t operation_CALLCODE(
            ArithEnv &arith,
            bn_t &gas_limit,
            bn_t &gas_used,
            uint32_t &error_code,
            uint32_t &pc,
            cuEVM::stack::EVMStack &stack,
            message_t &message,
            cuEVM::memory::EVMMemory &memory,
            touch_state_t &touch_state,
            uint8_t &opcode,
            evm_t &evm,
            cuEVM::EVMReturnData &return_data)
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
         * @param[out] error_code The error code.
         * @param[in] stack The stack.
         * @param[in] memory The memory.
         * @param[out] return_data The return data.
        */
        __host__ __device__ int32_t operation_RETURN(
            ArithEnv &arith,
            bn_t &gas_limit,
            bn_t &gas_used,
            uint32_t &error_code,
            cuEVM::stack::EVMStack &stack,
            cuEVM::memory::EVMMemory &memory,
            cuEVM::EVMReturnData &return_data)
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
         * The DELEGATECALL operation.
         * @param[in] arith The arithmetical environment.
         * @param[in] gas_limit The gas limit.
         * @param[inout] gas_used The gas used.
         * @param[out] error_code The error code.
         * @param[inout] pc The program counter.
         * @param[inout] stack The stack.
         * @param[in] message The current context message call.
         * @param[in] memory The memory.
         * @param[inout] touch_state The touch state.
         * @param[in] opcode The operation opcode
         * @param[out] evm The evm.
         * @param[out] return_data The return data.
        */
        __host__ __device__ int32_t operation_DELEGATECALL(
            ArithEnv &arith,
            bn_t &gas_limit,
            bn_t &gas_used,
            uint32_t &error_code,
            uint32_t &pc,
            cuEVM::stack::EVMStack &stack,
            message_t &message,
            cuEVM::memory::EVMMemory &memory,
            touch_state_t &touch_state,
            uint8_t &opcode,
            evm_t &evm,
            cuEVM::EVMReturnData &return_data)
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
         * The CREATE2 operation.
         * @param[in] arith The arithmetical environment.
         * @param[in] gas_limit The gas limit.
         * @param[inout] gas_used The gas used.
         * @param[out] error_code The error code.
         * @param[inout] pc The program counter.
         * @param[inout] stack The stack.
         * @param[in] message The current context message call.
         * @param[in] memory The memory.
         * @param[inout] touch_state The touch state.
         * @param[in] opcode The operation opcode
         * @param[out] evm The evm.
         * @param[out] return_data The return data.
         */
        __host__ __device__ int32_t operation_CREATE2(
            ArithEnv &arith,
            bn_t &gas_limit,
            bn_t &gas_used,
            uint32_t &error_code,
            uint32_t &pc,
            cuEVM::stack::EVMStack &stack,
            message_t &message,
            cuEVM::memory::EVMMemory &memory,
            touch_state_t &touch_state,
            uint8_t &opcode,
            evm_t &evm,
            cuEVM::EVMReturnData &return_data)
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
         * The STATICCALL operation.
         * @param[in] arith The arithmetical environment.
         * @param[in] gas_limit The gas limit.
         * @param[inout] gas_used The gas used.
         * @param[out] error_code The error code.
         * @param[inout] pc The program counter.
         * @param[inout] stack The stack.
         * @param[in] message The current context message call.
         * @param[in] memory The memory.
         * @param[inout] touch_state The touch state.
         * @param[in] opcode The operation opcode
         * @param[out] evm The evm.
         * @param[out] return_data The return data.
        */
        __host__ __device__ int32_t operation_STATICCALL(
            ArithEnv &arith,
            bn_t &gas_limit,
            bn_t &gas_used,
            uint32_t &error_code,
            uint32_t &pc,
            cuEVM::stack::EVMStack &stack,
            message_t &message,
            cuEVM::memory::EVMMemory &memory,
            touch_state_t &touch_state,
            uint8_t &opcode,
            evm_t &evm,
            cuEVM::EVMReturnData &return_data)
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
         * @param[out] error_code The error code.
         * @param[in] stack The stack.
         * @param[in] memory The memory.
         * @param[out] return_data The return data.
        */
        __host__ __device__ int32_t operation_REVERT(
            ArithEnv &arith,
            bn_t &gas_limit,
            bn_t &gas_used,
            uint32_t &error_code,
            cuEVM::stack::EVMStack &stack,
            cuEVM::memory::EVMMemory &memory,
            cuEVM::EVMReturnData &return_data)
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
         * @param[out] error_code The error code.
        */
        __host__ __device__ int32_t operation_INVALID(
            uint32_t &error_code)
        {
            error_code = ERR_NOT_IMPLEMENTED;
        }

        /**
         * The SELFDESTRUCT operation.
         * @param[in] arith The arithmetical environment.
         * @param[in] gas_limit The gas limit.
         * @param[inout] gas_used The gas used.
         * @param[out] error_code The error code.
         * @param[inout] pc The program counter.
         * @param[inout] stack The stack.
         * @param[in] message The current context message call.
         * @param[inout] touch_state The touch state.
         * @param[out] return_data The return data.
         * @param[out] evm The evm.
        */
        __host__ __device__ int32_t operation_SELFDESTRUCT(
            ArithEnv &arith,
            bn_t &gas_limit,
            bn_t &gas_used,
            uint32_t &error_code,
            uint32_t &pc,
            cuEVM::stack::EVMStack &stack,
            message_t &message,
            touch_state_t &touch_state,
            cuEVM::EVMReturnData &return_data,
            evm_t &evm)
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