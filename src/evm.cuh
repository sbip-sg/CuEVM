#ifndef _EVM_H_
#define _EVM_H_

#include "utils.h"
#include "stack.cuh"
#include "message.cuh"
#include "memory.cuh"
#include "returndata.cuh"
#include "block.cuh"
#include "tracer.cuh"
#include "state.cuh"
#include "keccak.cuh"
#include "jump_destinations.cuh"
#include "alu_operations.cuh"
#include "env_operations.cuh"
#include "internal_operations.cuh"

template <class params>
class evm_t
{
public:
    /**
     * The arithmetical environment used by the arbitrary length
     * integer library.
     */
    typedef arith_env_t<params> arith_t;
    /**
     * The arbitrary length integer type.
     */
    typedef typename arith_t::bn_t bn_t;
    /**
     * The block information class.
     */
    typedef block_t<params> block_t;
    /**
     * The block information data type.
     */
    typedef block_t::block_data_t block_data_t;
    /**
     * World state information class.
     */
    typedef world_state_t<params> world_state_t;
    /**
     * World state information data type.
     */
    typedef world_state_t::state_data_t state_data_t;
    /**
     * The account information data.
     */
    typedef world_state_t::account_t account_t;
    /**
     * The access state class.
     */
    typedef accessed_state_t<params> accessed_state_t;
    /**
     * The access state data type.
     */
    typedef accessed_state_t::accessed_state_data_t accessed_state_data_t;
    /**
     * The touch state class.
     */
    typedef touch_state_t<params> touch_state_t;
    /**
     * The touch state data type.
     */
    typedef touch_state_t::touch_state_data_t touch_state_data_t;
    /**
     * The stackk class.
     */
    typedef stack_t<params> stack_t;
    /**
     * The memory class.
     */
    typedef memory_t<params> memory_t;
    /**
     * The transaction class.
     */
    typedef transaction_t<params> transaction_t;
    /**
     * The transaction content data structure.
     */
    typedef typename transaction_t::transaction_data_t transaction_data_t;
    /**
     * The message class.
     */
    typedef message_t<params> message_t;
    /**
     * The message content data structure.
     */
    typedef typename message_t::message_data_t message_data_t;
    /**
     * The tracer class.
     */
    typedef tracer_t<params> tracer_t;
    /**
     * The tracer data structure.
     */
    typedef typename tracer_t::tracer_data_t tracer_data_t;
    /**
     * The keccak class.
     */
    typedef keccak::keccak_t keccak_t;
    /**
     * The keccak parameters.
     */
    typedef typename keccak_t::sha3_parameters_t sha3_parameters_t;
    /**
     * The arithmetic operations class.
     */
    typedef arithmetic_operations<params> arithmetic_operations;
    /**
     * The comparison operations class.
     */
    typedef comparison_operations<params> comparison_operations;
    /**
     * The bitwise operations class.
     */
    typedef bitwise_operations<params> bitwise_operations;
    /**
     * The stack operations class.
     */
    typedef stack_operations<params> stack_operations;
    /**
     * The block operations class.
     */
    typedef block_operations<params> block_operations;
    /**
     * The environmental operations class.
     */
    typedef environmental_operations<params> environmental_operations;
    /**
     * The internal operations class.
     */
    typedef internal_operations<params> internal_operations;

    // constants
    static const uint32_t MAX_DEPTH = 1024;
    static const uint32_t MAX_EXECUTION_STEPS = 30000;
    static const uint32_t DEPTH_PAGE_SIZE = 32;

    typedef struct
    {
        state_data_t *world_state_data;
        block_data_t *block_data;
        sha3_parameters_t *sha3_parameters;
        transaction_data_t *transactions_data;
        accessed_state_data_t *accessed_states_data;
        touch_state_data_t *touch_states_data;
#ifdef TRACER
        tracer_data_t *tracers_data;
#endif
        uint32_t *errors;
        size_t count;
    } evm_instances_t;

    arith_t _arith;
    world_state_t *_world_state;
    block_t *_block;
    keccak_t *_keccak;
    transaction_t *_transaction;
    accessed_state_t *_accessed_state;
    touch_state_t *_transaction_touch_state;
    uint32_t _instance;
#ifdef TRACER
    tracer_t *_tracer;
    uint32_t _trace_pc;
    uint8_t _trace_opcode;
    bn_t _trace_address;
#endif
    touch_state_t **_touch_state_ptrs;
    return_data_t **_last_return_data_ptrs;
    return_data_t *_final_return_data;
    message_t **_message_ptrs;
    memory_t **_memory_ptrs;
    stack_t **_stack_ptrs;
    bn_t *_gas_useds;
    bn_t *_gas_refunds;
    uint32_t *_pcs;
    accessed_state_data_t *_final_accessed_state_data;
    touch_state_data_t *_final_touch_state_data;
    uint32_t _depth;
    uint32_t _allocated_depth;
    bn_t _gas_limit;        /** YP: \f$T_{g}\f$*/
    bn_t _gas_price;        /**< YP: \f$p\f$ or \f$T_{p}\f$*/
    bn_t _gas_priority_fee; /**< YP: \f$f\f$*/
    uint8_t *_bytecode; /**< YP: \f$I_{b}\f$*/
    uint32_t _code_size;
    uint8_t _opcode;
    jump_destinations_t *_jump_destinations;
    uint32_t _error_code;
    uint32_t *_final_error;
    /*
     * Internal execution environment
     * I_{a} = message.get_recipient
     * I_{o} = _last_return_data_ptrs[_depth-1]
     * I_{p} = _gas_price
     * I_{d} = message.get_data
     * I_{s} = _stack_ptrs[_depth]
     * I_{v} = message.get_value
     * I_{b} = _bytecode
     * I_{e} = _depth
     * I_{w} = message.get_static_env
    */

    // constructor
    __host__ __device__ __forceinline__ evm_t(
        arith_t arith,
        state_data_t *world_state_data,
        block_data_t *block_data,
        sha3_parameters_t *sha3_parameters,
        transaction_data_t *transaction_data,
        accessed_state_data_t *accessed_state_data,
        touch_state_data_t *touch_state_data,
#ifdef TRACER
        tracer_data_t *tracer_data,
#endif
        uint32_t instance,
        uint32_t *error) : _arith(arith), _instance(instance), _final_error(error)
    {
        _world_state = new world_state_t(arith, world_state_data);
        _block = new block_t(arith, block_data);
        _keccak = new keccak_t(sha3_parameters);
        _transaction = new transaction_t(arith, transaction_data);
        _accessed_state = new accessed_state_t(_world_state);
        _transaction_touch_state = new touch_state_t(_accessed_state, NULL);
        _final_accessed_state_data = accessed_state_data;
        _final_touch_state_data = touch_state_data;
        _depth = 0;
        _allocated_depth = DEPTH_PAGE_SIZE;
        _touch_state_ptrs = new touch_state_t *[_allocated_depth];
        _last_return_data_ptrs = new return_data_t *[_allocated_depth];
        _final_return_data = new return_data_t();
        _message_ptrs = new message_t *[_allocated_depth];
        _memory_ptrs = new memory_t *[_allocated_depth];
        _stack_ptrs = new stack_t *[_allocated_depth];
        _gas_useds = new bn_t[_allocated_depth];
        _gas_refunds = new bn_t[_allocated_depth];
        _pcs = new uint32_t[_allocated_depth];
#ifdef TRACER
        _tracer = new tracer_t(arith, tracer_data);
#endif
        _jump_destinations = NULL;
        _error_code = ERR_NONE;
    }

    __host__ __device__ __forceinline__ ~evm_t()
    {
        _accessed_state->to_accessed_state_data_t(*_final_accessed_state_data);
        _transaction_touch_state->to_touch_state_data_t(*_final_touch_state_data);
        delete _world_state;
        delete _block;
        delete _keccak;
        delete _transaction;
        delete _accessed_state;
        delete _transaction_touch_state;
#ifdef TRACER
        delete _tracer;
#endif
        delete[] _touch_state_ptrs;
        delete[] _last_return_data_ptrs;
        delete _final_return_data;
        delete[] _message_ptrs;
        delete[] _memory_ptrs;
        delete[] _stack_ptrs;
        delete[] _gas_useds;
        delete[] _gas_refunds;
        delete[] _pcs;
        _allocated_depth = 0;
        _depth = 0;
    }

    __host__ __device__ __forceinline__ void grow()
    {
        uint32_t new_allocated_depth = _allocated_depth + DEPTH_PAGE_SIZE;
        touch_state_t **new_touch_state_ptrs = new touch_state_t *[new_allocated_depth];
        return_data_t **new_return_data_ptrs = new return_data_t *[new_allocated_depth];
        message_t **new_message_ptrs = new message_t *[new_allocated_depth];
        memory_t **new_memory_ptrs = new memory_t *[new_allocated_depth];
        stack_t **new_stack_ptrs = new stack_t *[new_allocated_depth];
        bn_t *new_gas_useds = new bn_t[new_allocated_depth];
        bn_t *new_gas_refunds = new bn_t[new_allocated_depth];
        uint32_t *new_pcs = new uint32_t[new_allocated_depth];

        memcpy(
            new_touch_state_ptrs,
            _touch_state_ptrs,
            _allocated_depth * sizeof(touch_state_t *));
        memcpy(
            new_return_data_ptrs,
            _last_return_data_ptrs,
            _allocated_depth * sizeof(return_data_t *));
        memcpy(
            new_message_ptrs,
            _message_ptrs,
            _allocated_depth * sizeof(message_t *));
        memcpy(
            new_memory_ptrs,
            _memory_ptrs,
            _allocated_depth * sizeof(memory_t *));
        memcpy(
            new_stack_ptrs,
            _stack_ptrs,
            _allocated_depth * sizeof(stack_t *));
        memcpy(
            new_gas_useds,
            _gas_useds,
            _allocated_depth * sizeof(bn_t));
        memcpy(
            new_gas_refunds,
            _gas_refunds,
            _allocated_depth * sizeof(bn_t));
        memcpy(
            new_pcs,
            _pcs,
            _allocated_depth * sizeof(uint32_t));

        delete[] _touch_state_ptrs;
        delete[] _last_return_data_ptrs;
        delete[] _message_ptrs;
        delete[] _memory_ptrs;
        delete[] _stack_ptrs;
        delete[] _gas_useds;
        delete[] _gas_refunds;
        delete[] _pcs;
        _touch_state_ptrs = new_touch_state_ptrs;
        _last_return_data_ptrs = new_return_data_ptrs;
        _message_ptrs = new_message_ptrs;
        _memory_ptrs = new_memory_ptrs;
        _stack_ptrs = new_stack_ptrs;
        _gas_useds = new_gas_useds;
        _gas_refunds = new_gas_refunds;
        _pcs = new_pcs;
        _allocated_depth = new_allocated_depth;
    }

    __host__ __device__ void process_transaction(
        bn_t &gas_used, /**< YP: \f$g_{0}\f$*/
        uint32_t &error_code)
    {
        bn_t block_base_fee; // YP: \f$H_{f}\f$
        _block->get_base_fee(block_base_fee);
        bn_t block_gas_limit;
        _block->get_gas_limit(block_gas_limit);
        _transaction->validate_transaction(
            *_transaction_touch_state,
            gas_used,
            _gas_price,
            _gas_priority_fee,
            error_code,
            block_base_fee,
            block_gas_limit);
    }

    __host__ __device__ void update_env()
    {
        _message_ptrs[_depth]->get_gas_limit(_gas_limit);
        _bytecode = _message_ptrs[_depth]->get_byte_code();
        _code_size = _message_ptrs[_depth]->get_code_size();
        #ifdef TRACER
        _message_ptrs[_depth]->get_contract_address(_trace_address);
        #endif
        if (_jump_destinations != NULL)
        {
            delete _jump_destinations;
            _jump_destinations = NULL;
        }
        _jump_destinations = new jump_destinations_t(_bytecode, _code_size);
    }

    __host__ __device__ void init_message_call(
        uint32_t &error_code)
    {
        update_env();
        _last_return_data_ptrs[_depth] = new return_data_t();
        _stack_ptrs[_depth] = new stack_t(_arith);
        _memory_ptrs[_depth] = new memory_t(_arith);
        if (_depth > 0)
        {
            _touch_state_ptrs[_depth] = new touch_state_t(
                _accessed_state,
                _touch_state_ptrs[_depth - 1]);
        }
        else
        {
            _touch_state_ptrs[_depth] = new touch_state_t(
                _accessed_state,
                _transaction_touch_state);
        }
        _pcs[_depth] = 0;
        cgbn_set_ui32(_arith._env, _gas_useds[_depth], 0);
        cgbn_set_ui32(_arith._env, _gas_refunds[_depth], 0);

        // transfer the value from the sender to receiver
        bn_t sender, receiver, value;
        bn_t sender_balance, receiver_balance;
        _message_ptrs[_depth]->get_sender(sender);
        _message_ptrs[_depth]->get_recipient(receiver);
        _message_ptrs[_depth]->get_value(value);
        if ((cgbn_compare_ui32(_arith._env, value, 0) > 0) &&       // value>0
            (cgbn_compare(_arith._env, sender, receiver) != 0) &&   // sender != receiver
            (_message_ptrs[_depth]->get_call_type() != OP_DELEGATECALL) // no delegatecall
        )
        {
            _touch_state_ptrs[_depth]->get_account_balance(sender, sender_balance);
            _touch_state_ptrs[_depth]->get_account_balance(receiver, receiver_balance);
            // verify the balance before transfer
            if (cgbn_compare(_arith._env, sender_balance, value) < 0)
            {
                error_code = ERROR_MESSAGE_CALL_SENDER_BALANCE;
                return;
            }
            cgbn_sub(_arith._env, sender_balance, sender_balance, value);
            cgbn_add(_arith._env, receiver_balance, receiver_balance, value);
            _touch_state_ptrs[_depth]->set_account_balance(sender, sender_balance);
            _touch_state_ptrs[_depth]->set_account_balance(receiver, receiver_balance);
        }
        // warm up the accounts
        account_t *account;
        account = _touch_state_ptrs[_depth]->get_account(sender, READ_NONE);
        account = _touch_state_ptrs[_depth]->get_account(receiver, READ_NONE);
        account = NULL;
    }

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
    class system_operations
    {
    public:
        __host__ __device__ __forceinline__ static void generic_call(
            arith_t &arith,
            bn_t &gas_limit,
            bn_t &gas_used,
            uint32_t &error_code,
            uint32_t &pc,
            stack_t &stack,
            message_t &message,
            memory_t &memory,
            touch_state_t &touch_state,
            uint8_t &opcode,
            evm_t &evm,
            message_t &new_message,
            bn_t &args_offset,
            bn_t &args_size)
        {
            // try to send value in static call
            bn_t value;
            new_message.get_value(value);
            if (message.get_static_env())
            {
                if (cgbn_compare_ui32(arith._env, value, 0) != 0)
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
                    if (touch_state.is_empty_account(contract_address))
                    {
                        cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_NEW_ACCOUNT);
                    };
                }
            }

            if (arith.has_gas(gas_limit, gas_used, error_code))
            {
                bn_t gas;
                new_message.get_gas_limit(gas);

                bn_t gas_left;
                cgbn_sub(arith._env, gas_left, gas_limit, gas_used);

                // gas capped = (63/64) * gas_left
                bn_t gas_left_capped;
                cgbn_set(arith._env, gas_left_capped, gas_left);
                cgbn_div_ui32(arith._env, gas_left_capped, gas_left_capped, 64);
                cgbn_sub(arith._env, gas_left_capped, gas_left, gas_left_capped);

                if (cgbn_compare(arith._env, gas, gas_left_capped) > 0)
                {
                    cgbn_set(arith._env, gas, gas_left_capped);
                }

                // add to gas used the sent gas
                cgbn_add(arith._env, gas_used, gas_used, gas);

                // add the call stippend
                cgbn_add(arith._env, gas, gas, gas_stippend);

                // set the new gas limit
                new_message.set_gas_limit(gas);

                uint8_t *call_data;
                size_t call_data_size;
                call_data = memory.get(
                    args_offset,
                    args_size,
                    error_code);
                arith.size_t_from_cgbn(call_data_size, args_size);

                new_message.set_data(call_data, call_data_size);

                // new message done
                // call the child
                evm.child_call(
                    error_code,
                    new_message);
            }
        }

        __host__ __device__ __forceinline__ static void generic_create(
            uint32_t &error_code)
        {
            error_code = ERR_NOT_IMPLEMENTED;
        }

        __host__ __device__ __forceinline__ static void operation_STOP(
            return_data_t &return_data,
            uint32_t &error_code)
        {
            return_data.set(
                NULL,
                0);
            error_code = ERR_RETURN;
        }

        __host__ __device__ __forceinline__ static void operation_CREATE(
            arith_t &arith,
            bn_t &gas_limit,
            bn_t &gas_used,
            uint32_t &error_code,
            uint32_t &pc,
            stack_t &stack,
            message_t &message,
            memory_t &memory,
            touch_state_t &touch_state,
            uint8_t &opcode,
            evm_t &evm)
        {
            error_code = ERR_NOT_IMPLEMENTED;
        }

        __host__ __device__ __forceinline__ static void operation_CALL(
            arith_t &arith,
            bn_t &gas_limit,
            bn_t &gas_used,
            uint32_t &error_code,
            uint32_t &pc,
            stack_t &stack,
            message_t &message,
            memory_t &memory,
            touch_state_t &touch_state,
            uint8_t &opcode,
            evm_t &evm)
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
                bn_t sender;
                message.get_recipient(sender); // I_{a}
                bn_t recipient;
                cgbn_set(arith._env, recipient, address); // t
                bn_t contract_address;
                cgbn_set(arith._env, contract_address, address); // t
                bn_t storage_address;
                cgbn_set(arith._env, storage_address, address); // t

                account_t *contract;
                contract = touch_state.get_account(contract_address, READ_CODE);
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
                    contract->bytecode,
                    contract->code_size,
                    ret_offset,
                    ret_size,
                    message.get_static_env());

                generic_call(
                    arith,
                    gas_limit,
                    gas_used,
                    error_code,
                    pc,
                    stack,
                    message,
                    memory,
                    touch_state,
                    opcode,
                    evm,
                    *new_message,
                    args_offset,
                    args_size);

                pc = pc + 1;
            }
        }

        __host__ __device__ __forceinline__ static void operation_CALLCODE(
            arith_t &arith,
            bn_t &gas_limit,
            bn_t &gas_used,
            uint32_t &error_code,
            uint32_t &pc,
            stack_t &stack,
            message_t &message,
            memory_t &memory,
            touch_state_t &touch_state,
            uint8_t &opcode,
            evm_t &evm)
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
                bn_t sender;
                message.get_recipient(sender); // I_{a}
                bn_t recipient;
                cgbn_set(arith._env, recipient, sender); // I_{a}
                bn_t contract_address;
                cgbn_set(arith._env, contract_address, address); // t
                bn_t storage_address;
                cgbn_set(arith._env, storage_address, sender); // I_{a}
                account_t *contract;
                contract = touch_state.get_account(contract_address, READ_CODE);

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
                    contract->bytecode,
                    contract->code_size,
                    ret_offset,
                    ret_size,
                    message.get_static_env());

                generic_call(
                    arith,
                    gas_limit,
                    gas_used,
                    error_code,
                    pc,
                    stack,
                    message,
                    memory,
                    touch_state,
                    opcode,
                    evm,
                    *new_message,
                    args_offset,
                    args_size);

                pc = pc + 1;
            }
        }

        __host__ __device__ __forceinline__ static void operation_RETURN(
            arith_t &arith,
            bn_t &gas_limit,
            bn_t &gas_used,
            uint32_t &error_code,
            stack_t &stack,
            memory_t &memory,
            return_data_t &return_data)
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

                    return_data.set(
                        data,
                        data_size);

                    error_code = ERR_RETURN;
                }
            }
        }

        __host__ __device__ __forceinline__ static void operation_DELEGATECALL(
            arith_t &arith,
            bn_t &gas_limit,
            bn_t &gas_used,
            uint32_t &error_code,
            uint32_t &pc,
            stack_t &stack,
            message_t &message,
            memory_t &memory,
            touch_state_t &touch_state,
            uint8_t &opcode,
            evm_t &evm)
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
                bn_t sender;
                message.get_sender(sender); // keep the message call sender I_{s}
                bn_t recipient;
                message.get_recipient(recipient); // I_{a}
                bn_t contract_address;
                cgbn_set(arith._env, contract_address, address); // t
                bn_t storage_address;
                message.get_recipient(storage_address); // I_{a}
                account_t *contract;
                contract = touch_state.get_account(contract_address, READ_CODE);

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
                    contract->bytecode,
                    contract->code_size,
                    ret_offset,
                    ret_size,
                    message.get_static_env());

                generic_call(
                    arith,
                    gas_limit,
                    gas_used,
                    error_code,
                    pc,
                    stack,
                    message,
                    memory,
                    touch_state,
                    opcode,
                    evm,
                    *new_message,
                    args_offset,
                    args_size);

                pc = pc + 1;
            }
        }

        __host__ __device__ __forceinline__ static void operation_CREATE2(
            arith_t &arith,
            bn_t &gas_limit,
            bn_t &gas_used,
            uint32_t &error_code,
            uint32_t &pc,
            stack_t &stack,
            message_t &message,
            memory_t &memory,
            touch_state_t &touch_state,
            uint8_t &opcode,
            evm_t &evm)
        {
            error_code = ERR_NOT_IMPLEMENTED;
        }

        __host__ __device__ __forceinline__ static void operation_STATICCALL(
            arith_t &arith,
            bn_t &gas_limit,
            bn_t &gas_used,
            uint32_t &error_code,
            uint32_t &pc,
            stack_t &stack,
            message_t &message,
            memory_t &memory,
            touch_state_t &touch_state,
            uint8_t &opcode,
            evm_t &evm)
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
                bn_t sender;
                message.get_recipient(sender); // I_{a}
                bn_t recipient;
                cgbn_set(arith._env, recipient, address); // t
                bn_t contract_address;
                cgbn_set(arith._env, contract_address, address); // t
                bn_t storage_address;
                cgbn_set(arith._env, storage_address, address); // t
                account_t *contract;
                contract = touch_state.get_account(contract_address, READ_CODE);

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
                    contract->bytecode,
                    contract->code_size,
                    ret_offset,
                    ret_size,
                    1);

                generic_call(
                    arith,
                    gas_limit,
                    gas_used,
                    error_code,
                    pc,
                    stack,
                    message,
                    memory,
                    touch_state,
                    opcode,
                    evm,
                    *new_message,
                    args_offset,
                    args_size);

                pc = pc + 1;
            }
        }

        __host__ __device__ __forceinline__ static void operation_REVERT(
            arith_t &arith,
            bn_t &gas_limit,
            bn_t &gas_used,
            uint32_t &error_code,
            stack_t &stack,
            memory_t &memory,
            return_data_t &return_data)
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

                    return_data.set(
                        data,
                        data_size);

                    error_code = ERR_REVERT;
                }
            }
        }

        __host__ __device__ __forceinline__ static void operation_INVALID(
            uint32_t &error_code)
        {
            error_code = ERR_NOT_IMPLEMENTED;
        }

        __host__ __device__ __forceinline__ static void operation_SELFDESTRUCT(
            arith_t &arith,
            bn_t &gas_limit,
            bn_t &gas_used,
            uint32_t &error_code,
            uint32_t &pc,
            stack_t &stack,
            message_t &message,
            memory_t &memory,
            touch_state_t &touch_state,
            uint8_t &opcode,
            evm_t &evm)
        {
            error_code = ERR_NOT_IMPLEMENTED;
        }
    };

    __host__ __device__ void run(
        uint32_t error_code)
    {
        // get the first message call from transaction
        _message_ptrs[_depth] = _transaction->get_message_call(*_accessed_state);
        // process the transaction
        bn_t intrsinc_gas_used;
        process_transaction(intrsinc_gas_used, error_code);
        if (error_code != ERR_NONE)
        {
            // TODO: do stuff for fail transaction
            return;
        }
        // init the first message call
        init_message_call(error_code);
        // add the transaction cost
        cgbn_add(_arith._env, _gas_useds[_depth], _gas_useds[_depth], intrsinc_gas_used);
        // run the message call
        uint32_t execution_step = 0;
        while (
            (execution_step < MAX_EXECUTION_STEPS))
        {

            if (_pcs[_depth] >= _code_size)
            {
                _opcode = OP_STOP;
            }
            else
            {
                _opcode = _bytecode[_pcs[_depth]];
            }
            ONE_THREAD_PER_INSTANCE(
                printf("pc: %d opcode: %d\n", _pcs[_depth], _opcode);)
#ifdef TRACER
            _trace_pc = _pcs[_depth];
            _trace_opcode = _opcode;
#endif
            // PUSHX
            if (((_opcode & 0xF0) == 0x60) || ((_opcode & 0xF0) == 0x70))
            {
                stack_operations::operation_PUSHX(
                    _arith,
                    _gas_limit,
                    _gas_useds[_depth],
                    error_code,
                    _pcs[_depth],
                    *_stack_ptrs[_depth],
                    _bytecode,
                    _code_size,
                    _opcode);
            }
            else if ((_opcode & 0xF0) == 0x80) // DUPX
            {
                stack_operations::operation_DUPX(
                    _arith,
                    _gas_limit,
                    _gas_useds[_depth],
                    error_code,
                    _pcs[_depth],
                    *_stack_ptrs[_depth],
                    _opcode);
            }
            else if ((_opcode & 0xF0) == 0x90) // SWAPX
            {
                stack_operations::operation_SWAPX(
                    _arith,
                    _gas_limit,
                    _gas_useds[_depth],
                    error_code,
                    _pcs[_depth],
                    *_stack_ptrs[_depth],
                    _opcode);
            }
            else if ((_opcode & 0xF0) == 0xA0) // LOGX
            {
                internal_operations::operation_LOGX(
                    _arith,
                    _gas_limit,
                    _gas_useds[_depth],
                    error_code,
                    _pcs[_depth],
                    *_stack_ptrs[_depth],
                    _opcode);
            }
            else
            {
                switch (_opcode)
                {
                case OP_STOP: // STOP
                {
                    if (_depth == 0)
                    {
                        system_operations::operation_STOP(
                            *_final_return_data,
                            error_code);
                    }
                    else
                    {
                        system_operations::operation_STOP(
                            *_last_return_data_ptrs[_depth - 1],
                            error_code);
                    }
                }
                break;
                case OP_ADD: // ADD
                {
                    arithmetic_operations::operation_ADD(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth]);
                }
                break;
                case OP_MUL: // MUL
                {
                    arithmetic_operations::operation_MUL(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth]);
                }
                break;
                case OP_SUB: // SUB
                {
                    arithmetic_operations::operation_SUB(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth]);
                }
                break;
                case OP_DIV: // DIV
                {
                    arithmetic_operations::operation_DIV(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth]);
                }
                break;
                case OP_SDIV: // SDIV
                {
                    arithmetic_operations::operation_SDIV(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth]);
                }
                break;
                case OP_MOD: // MOD
                {
                    arithmetic_operations::operation_MOD(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth]);
                }
                break;
                case OP_SMOD: // SMOD
                {
                    arithmetic_operations::operation_SMOD(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth]);
                }
                break;
                case OP_ADDMOD: // ADDMOD
                {
                    arithmetic_operations::operation_ADDMOD(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth]);
                }
                break;
                case OP_MULMOD: // MULMOD
                {
                    arithmetic_operations::operation_MULMOD(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth]);
                }
                break;
                case OP_EXP: // EXP
                {
                    arithmetic_operations::operation_EXP(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth]);
                }
                break;
                case OP_SIGNEXTEND: // SIGNEXTEND
                {
                    arithmetic_operations::operation_SIGNEXTEND(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth]);
                }
                break;
                case OP_LT: // LT
                {
                    comparison_operations::operation_LT(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth]);
                }
                break;
                case OP_GT: // GT
                {
                    comparison_operations::operation_GT(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth]);
                }
                break;
                case OP_SLT: // SLT
                {
                    comparison_operations::operation_SLT(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth]);
                }
                break;
                case OP_SGT: // SGT
                {
                    comparison_operations::operation_SGT(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth]);
                }
                break;
                case OP_EQ: // EQ
                {
                    comparison_operations::operation_EQ(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth]);
                }
                break;
                case OP_ISZERO: // ISZERO
                {
                    comparison_operations::operation_ISZERO(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth]);
                }
                break;
                case OP_AND: // AND
                {
                    bitwise_operations::operation_AND(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth]);
                }
                break;
                case OP_OR: // OR
                {
                    bitwise_operations::operation_OR(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth]);
                }
                break;
                case OP_XOR: // XOR
                {
                    bitwise_operations::operation_XOR(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth]);
                }
                break;
                case OP_NOT: // NOT
                {
                    bitwise_operations::operation_NOT(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth]);
                }
                break;
                case OP_BYTE: // BYTE
                {
                    bitwise_operations::operation_BYTE(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth]);
                }
                break;
                case OP_SHL: // SHL
                {
                    bitwise_operations::operation_SHL(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth]);
                }
                break;
                case OP_SHR: // SHR
                {
                    bitwise_operations::operation_SHR(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth]);
                }
                break;
                case OP_SAR: // SAR
                {
                    bitwise_operations::operation_SAR(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth]);
                }
                break;
                case OP_SHA3: // SHA3
                {
                    environmental_operations::operation_SHA3(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth],
                        *_keccak,
                        *_memory_ptrs[_depth]);
                }
                break;
                case OP_ADDRESS: // ADDRESS
                {
                    environmental_operations::operation_ADDRESS(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth],
                        *_message_ptrs[_depth]);
                }
                break;
                case OP_BALANCE: // BALANCE
                {
                    environmental_operations::operation_BALANCE(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth],
                        *_touch_state_ptrs[_depth]);
                }
                break;
                case OP_ORIGIN: // ORIGIN
                {
                    environmental_operations::operation_ORIGIN(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth],
                        *_transaction);
                }
                break;
                case OP_CALLER: // CALLER
                {
                    environmental_operations::operation_CALLER(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth],
                        *_message_ptrs[_depth]);
                }
                break;
                case OP_CALLVALUE: // CALLVALUE
                {
                    environmental_operations::operation_CALLVALUE(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth],
                        *_message_ptrs[_depth]);
                }
                break;
                case OP_CALLDATALOAD: // CALLDATALOAD
                {
                    environmental_operations::operation_CALLDATALOAD(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth],
                        *_message_ptrs[_depth]);
                }
                break;
                case OP_CALLDATASIZE: // CALLDATASIZE
                {
                    environmental_operations::operation_CALLDATASIZE(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth],
                        *_message_ptrs[_depth]);
                }
                break;
                case OP_CALLDATACOPY: // CALLDATACOPY
                {
                    environmental_operations::operation_CALLDATACOPY(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth],
                        *_message_ptrs[_depth],
                        *_memory_ptrs[_depth]);
                }
                break;
                case OP_CODESIZE: // CODESIZE
                {
                    environmental_operations::operation_CODESIZE(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth],
                        *_message_ptrs[_depth]);
                }
                break;
                case OP_CODECOPY: // CODECOPY
                {
                    environmental_operations::operation_CODECOPY(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth],
                        *_message_ptrs[_depth],
                        *_memory_ptrs[_depth]);
                }
                break;
                case OP_GASPRICE: // GASPRICE
                {
                    environmental_operations::operation_GASPRICE(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth],
                        *_block,
                        *_transaction);
                }
                break;
                case OP_EXTCODESIZE: // EXTCODESIZE
                {
                    environmental_operations::operation_EXTCODESIZE(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth],
                        *_touch_state_ptrs[_depth]);
                }
                break;
                case OP_EXTCODECOPY: // EXTCODECOPY
                {
                    environmental_operations::operation_EXTCODECOPY(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth],
                        *_touch_state_ptrs[_depth],
                        *_memory_ptrs[_depth]);
                }
                break;
                case OP_RETURNDATASIZE: // RETURNDATASIZE
                {
                    environmental_operations::operation_RETURNDATASIZE(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth],
                        *_last_return_data_ptrs[_depth]);
                }
                break;
                case OP_RETURNDATACOPY: // RETURNDATACOPY
                {
                    environmental_operations::operation_RETURNDATACOPY(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth],
                        *_memory_ptrs[_depth],
                        *_last_return_data_ptrs[_depth]);
                }
                break;
                case OP_EXTCODEHASH: // EXTCODEHASH
                {
                    environmental_operations::operation_EXTCODEHASH(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth],
                        *_touch_state_ptrs[_depth],
                        *_keccak);
                }
                case OP_BLOCKHASH: // BLOCKHASH
                {
                    block_operations::operation_BLOCKHASH(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth],
                        *_block);
                }
                break;
                case OP_COINBASE: // COINBASE
                {
                    block_operations::operation_COINBASE(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth],
                        *_block);
                }
                break;
                case OP_TIMESTAMP: // TIMESTAMP
                {
                    block_operations::operation_TIMESTAMP(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth],
                        *_block);
                }
                break;
                case OP_NUMBER: // NUMBER
                {
                    block_operations::operation_NUMBER(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth],
                        *_block);
                }
                break;
                case OP_DIFFICULTY: // DIFFICULTY
                {
                    block_operations::operation_PREVRANDAO(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth],
                        *_block);
                }
                break;
                case OP_GASLIMIT: // GASLIMIT
                {
                    block_operations::operation_GASLIMIT(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth],
                        *_block);
                }
                break;
                case OP_CHAINID: // CHAINID
                {
                    block_operations::operation_CHAINID(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth],
                        *_block);
                }
                break;
                case OP_SELFBALANCE: // SELFBALANCE
                {
                    environmental_operations::operation_SELFBALANCE(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth],
                        *_touch_state_ptrs[_depth],
                        *_message_ptrs[_depth]);
                }
                break;
                case OP_BASEFEE: // BASEFEE
                {
                    block_operations::operation_BASEFEE(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth],
                        *_block);
                }
                break;
                case OP_POP: // POP
                {
                    stack_operations::operation_POP(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth]);
                }
                break;
                case OP_MLOAD: // MLOAD
                {
                    internal_operations::operation_MLOAD(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth],
                        *_memory_ptrs[_depth]);
                }
                break;
                case OP_MSTORE: // MSTORE
                {
                    internal_operations::operation_MSTORE(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth],
                        *_memory_ptrs[_depth]);
                }
                break;
                case OP_MSTORE8: // MSTORE8
                {
                    internal_operations::operation_MSTORE8(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth],
                        *_memory_ptrs[_depth]);
                }
                break;
                case OP_SLOAD: // SLOAD
                {
                    internal_operations::operation_SLOAD(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth],
                        *_touch_state_ptrs[_depth],
                        *_message_ptrs[_depth]);
                }
                break;
                case OP_SSTORE: // SSTORE
                {
                    internal_operations::operation_SSTORE(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        _gas_refunds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth],
                        *_touch_state_ptrs[_depth],
                        *_message_ptrs[_depth]);
                }
                break;
                case OP_JUMP: // JUMP
                {
                    internal_operations::operation_JUMP(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth],
                        *_jump_destinations);
                }
                break;
                case OP_JUMPI: // JUMPI
                {
                    internal_operations::operation_JUMPI(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth],
                        *_jump_destinations);
                }
                break;
                case OP_PC: // PC
                {
                    internal_operations::operation_PC(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth]);
                }
                break;
                case OP_MSIZE: // MSIZE
                {
                    internal_operations::operation_MSIZE(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth],
                        *_memory_ptrs[_depth]);
                }
                break;
                case OP_GAS: // GAS
                {
                    internal_operations::operation_GAS(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth]);
                }
                break;
                case OP_JUMPDEST: // JUMPDEST
                {
                    internal_operations::operation_JUMPDEST(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth]);
                }
                break;
                case OP_PUSH0: // PUSH0
                {
                    stack_operations::operation_PUSH0(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth]);
                }
                break;
                case OP_CREATE: // CREATE
                {
                    system_operations::operation_CREATE(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth],
                        *_message_ptrs[_depth],
                        *_memory_ptrs[_depth],
                        *_touch_state_ptrs[_depth],
                        _opcode,
                        *this);
                }
                break;
                case OP_CALL: // CALL
                {
                    system_operations::operation_CALL(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth],
                        *_message_ptrs[_depth],
                        *_memory_ptrs[_depth],
                        *_touch_state_ptrs[_depth],
                        _opcode,
                        *this);
                }
                break;
                case OP_CALLCODE: // CALLCODE
                {
                    system_operations::operation_CALLCODE(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth],
                        *_message_ptrs[_depth],
                        *_memory_ptrs[_depth],
                        *_touch_state_ptrs[_depth],
                        _opcode,
                        *this);
                }
                break;
                case OP_RETURN: // RETURN
                {
                    if (_depth == 0)
                    {
                        system_operations::operation_RETURN(
                            _arith,
                            _gas_limit,
                            _gas_useds[_depth],
                            error_code,
                            *_stack_ptrs[_depth],
                            *_memory_ptrs[_depth],
                            *_final_return_data);
                    }
                    else
                    {
                        system_operations::operation_RETURN(
                            _arith,
                            _gas_limit,
                            _gas_useds[_depth],
                            error_code,
                            *_stack_ptrs[_depth],
                            *_memory_ptrs[_depth],
                            *_last_return_data_ptrs[_depth]);
                    }
                }
                break;
                case OP_DELEGATECALL: // DELEGATECALL
                {
                    system_operations::operation_DELEGATECALL(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth],
                        *_message_ptrs[_depth],
                        *_memory_ptrs[_depth],
                        *_touch_state_ptrs[_depth],
                        _opcode,
                        *this);
                }
                break;
                case OP_CREATE2: // CREATE2
                {
                    system_operations::operation_CREATE2(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth],
                        *_message_ptrs[_depth],
                        *_memory_ptrs[_depth],
                        *_touch_state_ptrs[_depth],
                        _opcode,
                        *this);
                }
                break;
                case OP_STATICCALL: // STATICCALL
                {
                    system_operations::operation_STATICCALL(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth],
                        *_message_ptrs[_depth],
                        *_memory_ptrs[_depth],
                        *_touch_state_ptrs[_depth],
                        _opcode,
                        *this);
                }
                break;
                case OP_REVERT: // REVERT
                {
                    if (_depth == 0)
                    {
                        system_operations::operation_REVERT(
                            _arith,
                            _gas_limit,
                            _gas_useds[_depth],
                            error_code,
                            *_stack_ptrs[_depth],
                            *_memory_ptrs[_depth],
                            *_final_return_data);
                    }
                    else
                    {
                        system_operations::operation_REVERT(
                            _arith,
                            _gas_limit,
                            _gas_useds[_depth],
                            error_code,
                            *_stack_ptrs[_depth],
                            *_memory_ptrs[_depth],
                            *_last_return_data_ptrs[_depth]);
                    }
                }
                break;
                case OP_INVALID: // INVALID
                {
                    system_operations::operation_INVALID(
                        error_code);
                }
                break;
                case OP_SELFDESTRUCT: // SELFDESTRUCT
                {
                    system_operations::operation_SELFDESTRUCT(
                        _arith,
                        _gas_limit,
                        _gas_useds[_depth],
                        error_code,
                        _pcs[_depth],
                        *_stack_ptrs[_depth],
                        *_message_ptrs[_depth],
                        *_memory_ptrs[_depth],
                        *_touch_state_ptrs[_depth],
                        _opcode,
                        *this);
                }
                break;
                default:
                {
                    system_operations::operation_INVALID(
                        error_code);
                }
                break;
                }
            }

            if (error_code != ERR_NONE)
            {
                if (_depth == 0)
                {
                    #ifdef TRACER
                    _tracer->push(
                        _trace_address,
                        _trace_pc,
                        _trace_opcode,
                        *_stack_ptrs[_depth],
                        *_memory_ptrs[_depth],
                        *_touch_state_ptrs[_depth],
                        _gas_useds[_depth],
                        _gas_limit,
                        _gas_refunds[_depth],
                        error_code);
                    #endif
                }
                // delete the env
                delete _stack_ptrs[_depth];
                _stack_ptrs[_depth] = NULL;
                delete _memory_ptrs[_depth];
                _memory_ptrs[_depth] = NULL;
                delete _last_return_data_ptrs[_depth];
                _last_return_data_ptrs[_depth] = NULL;
                if (error_code == ERR_RETURN)
                {
                    // tocuh state update father
                    if (_depth == 0)
                    {
                        _transaction_touch_state->update_with_child_state(
                            *_touch_state_ptrs[_depth]);
                    }
                    else
                    {
                        _touch_state_ptrs[_depth - 1]->update_with_child_state(
                            *_touch_state_ptrs[_depth]);
                    }
                }
                // delete the touch state
                delete _touch_state_ptrs[_depth];
                _touch_state_ptrs[_depth] = NULL;
                // gas refund
                if (_depth > 0)
                {
                    if (error_code == ERR_RETURN)
                    {
                        cgbn_add(_arith._env, _gas_refunds[_depth - 1], _gas_refunds[_depth - 1], _gas_refunds[_depth]);
                    }
                    if ((error_code == ERR_RETURN) || (error_code == ERR_REVERT))
                    {
                        bn_t gas_left;
                        cgbn_sub(_arith._env, gas_left, _gas_limit, _gas_useds[_depth]);
                        cgbn_sub(_arith._env, _gas_useds[_depth - 1], _gas_useds[_depth - 1], gas_left);
                    }
                    // reset the gas
                    cgbn_set_ui32(_arith._env, _gas_useds[_depth], 0);
                    cgbn_set_ui32(_arith._env, _gas_refunds[_depth], 0);
                }
                else
                {
                    bn_t gas_value;
                    bn_t beneficiary;
                    _block->get_coin_base(beneficiary);
                    if (error_code == ERR_RETURN)
                    {
                        bn_t gas_left;
                        // \f$T_{g} - g\f$
                        cgbn_sub(_arith._env, gas_left, _gas_limit, _gas_useds[_depth]);
                        bn_t capped_refund_gas;
                        // \f$g/5\f$
                        cgbn_div_ui32(_arith._env, capped_refund_gas, gas_left, 5);
                        // min ( \f$g/5\f$, \f$R_{g}\f$)
                        if (cgbn_compare(_arith._env, capped_refund_gas, _gas_refunds[_depth]) > 0)
                        {
                            cgbn_set(_arith._env, capped_refund_gas, _gas_refunds[_depth]);
                        }
                        // g^{*} = \f$T_{g} - g + min ( \f$g/5\f$, \f$R_{g}\f$)\f$
                        cgbn_add(_arith._env, gas_value, gas_left, capped_refund_gas);
                        // add to sender balance g^{*}
                        bn_t sender_balance;
                        bn_t sender_address;
                        _transaction->get_sender(sender_address);
                        _transaction_touch_state->get_account_balance(sender_address, sender_balance);
                        cgbn_add(_arith._env, sender_balance, sender_balance, gas_value);
                        _transaction_touch_state->set_account_balance(sender_address, sender_balance);

                        // the gas value for the beneficiary is \f$T_{g} - g^{*}\f$
                        cgbn_sub(_arith._env, gas_value, _gas_limit, gas_value);
                    }
                    else
                    {
                        cgbn_mul(_arith._env, gas_value, _gas_limit, _gas_priority_fee);
                    }
                    bn_t beneficiary_balance;
                    _transaction_touch_state->get_account_balance(beneficiary, beneficiary_balance);
                    cgbn_add(_arith._env, beneficiary_balance, beneficiary_balance, gas_value);
                    _transaction_touch_state->set_account_balance(beneficiary, beneficiary_balance);
                }

                bn_t ret_offset, ret_size;
                _message_ptrs[_depth]->get_return_data_offset(ret_offset);
                _message_ptrs[_depth]->get_return_data_size(ret_size);
                // delete the message
                delete _message_ptrs[_depth];
                _message_ptrs[_depth] = NULL;

                // add the answer in the parent
                if (_depth == 0)
                {
                    // sent the gas value to the block, and sender
                    _transaction_touch_state->to_touch_state_data_t(
                        *_final_touch_state_data);
                    _accessed_state->to_accessed_state_data_t(
                        *_final_accessed_state_data);
                    if (error_code == ERR_RETURN)
                    {
                        _error_code = ERR_NONE;
                    }
                    else
                    {
                        _error_code = error_code;
                    }
                    *_final_error = _error_code;
                    delete _jump_destinations;
                    _jump_destinations = NULL;
                    break;
                }
                else
                {
                    bn_t child_success;
                    if (error_code == ERR_RETURN)
                    {
                        cgbn_set_ui32(_arith._env, child_success, 1);
                    }
                    else
                    {
                        cgbn_set_ui32(_arith._env, child_success, 0);
                    }
                    // go back in the parent call
                    _depth = _depth - 1;
                    update_env();
                    error_code = ERR_NONE;
                    _stack_ptrs[_depth]->push(child_success, error_code);
                    bn_t return_data_index;
                    cgbn_set_ui32(_arith._env, return_data_index, 0);
                    uint8_t *data;
                    size_t data_size;
                    data = _arith.get_data(
                        *(_last_return_data_ptrs[_depth]->get_data()),
                        return_data_index,
                        ret_size,
                        data_size);
                    _memory_ptrs[_depth]->set(
                        data,
                        ret_offset,
                        ret_size,
                        data_size,
                        error_code);
                    #ifdef TRACER
                    _tracer->push(
                        _trace_address,
                        _trace_pc,
                        _trace_opcode,
                        *_stack_ptrs[_depth],
                        *_memory_ptrs[_depth],
                        *_touch_state_ptrs[_depth],
                        _gas_useds[_depth],
                        _gas_limit,
                        _gas_refunds[_depth],
                        error_code);
                    #endif
                }
            }
            else
            {
                #ifdef TRACER
                _tracer->push(
                    _trace_address,
                    _trace_pc,
                    _trace_opcode,
                    *_stack_ptrs[_depth],
                    *_memory_ptrs[_depth],
                    *_touch_state_ptrs[_depth],
                    _gas_useds[_depth],
                    _gas_limit,
                    _gas_refunds[_depth],
                    error_code);
                #endif
            }
        }
    }

    __host__ __device__ __forceinline__ void child_call(
        uint32_t &error_code,
        message_t &new_message)
    {
        if ((_depth + 1) < MAX_DEPTH)
        {
            if (_depth == _allocated_depth)
            {
                grow();
            }
            _pcs[_depth + 1] = 0;
            _depth = _depth + 1;
            _message_ptrs[_depth] = &new_message;
            init_message_call(error_code);
            update_env();
        }
        else
        {
            error_code = ERROR_MAX_DEPTH_EXCEEDED;
        }
    }

    __host__ static void get_cpu_instances(
        evm_instances_t &instances,
        const cJSON *test)
    {
        arith_t arith(cgbn_report_monitor, 0);

        world_state_t *cpu_world_state;
        cpu_world_state = new world_state_t(arith, test);
        instances.world_state_data = cpu_world_state->_content;
        delete cpu_world_state;
        cpu_world_state = NULL;

        block_t *cpu_block = NULL;
        cpu_block = new block_t(arith, test);
        instances.block_data = cpu_block->_content;
        delete cpu_block;
        cpu_block = NULL;
        
        keccak_t *keccak;
        keccak = new keccak_t();
        instances.sha3_parameters = keccak->_parameters;
        delete keccak;
        keccak = NULL;

        instances.transactions_data = transaction_t::get_transactions(test, instances.count);

        instances.accessed_states_data = accessed_state_t::get_cpu_instances(instances.count);

        instances.touch_states_data = touch_state_t::get_cpu_instances(instances.count);

        #ifdef TRACER
        instances.tracers_data = tracer_t::get_cpu_instances(instances.count);
        #endif

    
        #ifndef ONLY_CPU
        CUDA_CHECK(cudaMallocManaged(
            (void **)&(instances.errors),
            sizeof(uint32_t) * instances.count
        ));
        #else
        instances.errors = new uint32_t[instances.count];
        #endif
        memset(instances.errors, ERR_NONE, sizeof(uint32_t) * instances.count);
    }

    __host__ static void get_gpu_instances(
        evm_instances_t &gpu_instances,
        evm_instances_t &cpu_instances)
    {
        gpu_instances.count = cpu_instances.count;

        gpu_instances.world_state_data = cpu_instances.world_state_data;

        gpu_instances.block_data = cpu_instances.block_data;

        gpu_instances.sha3_parameters = cpu_instances.sha3_parameters;

        gpu_instances.transactions_data = cpu_instances.transactions_data;

        gpu_instances.accessed_states_data = accessed_state_t::get_gpu_instances_from_cpu_instances(cpu_instances.accessed_states_data, cpu_instances.count);

        gpu_instances.touch_states_data = touch_state_t::get_gpu_instances_from_cpu_instances(cpu_instances.touch_states_data, cpu_instances.count);

        #ifdef TRACER
        gpu_instances.tracers_data = tracer_t::get_gpu_instances_from_cpu_instances(cpu_instances.tracers_data, cpu_instances.count);
        #endif

        gpu_instances.errors = cpu_instances.errors;
    }

    __host__ static void get_cpu_from_gpu_instances(
        evm_instances_t &cpu_instances,
        evm_instances_t &gpu_instances)
    {
        cpu_instances.count = gpu_instances.count;

        cpu_instances.world_state_data = gpu_instances.world_state_data;
        cpu_instances.block_data = gpu_instances.block_data;
        cpu_instances.sha3_parameters = gpu_instances.sha3_parameters;
        cpu_instances.transactions_data = gpu_instances.transactions_data;
        accessed_state_t::free_cpu_instances(cpu_instances.accessed_states_data, cpu_instances.count);
        cpu_instances.accessed_states_data = accessed_state_t::get_cpu_instances_from_gpu(gpu_instances.accessed_states_data, gpu_instances.count);
        touch_state_t::free_cpu_instances(cpu_instances.touch_states_data, cpu_instances.count);
        cpu_instances.touch_states_data = touch_state_t::get_cpu_instances_from_gpu(gpu_instances.touch_states_data, gpu_instances.count);
        #ifdef TRACER
        tracer_t::free_cpu_instances(cpu_instances.tracers_data, cpu_instances.count);
        cpu_instances.tracers_data = tracer_t::get_cpu_instances_from_gpu(gpu_instances.tracers_data, gpu_instances.count);
        #endif
        cpu_instances.errors = gpu_instances.errors;
    }

    __host__ static void free_instances(
        evm_instances_t &cpu_instances)
    {
        arith_t arith(cgbn_report_monitor, 0);

        world_state_t *cpu_world_state;
        cpu_world_state = new world_state_t(arith, cpu_instances.world_state_data);
        cpu_world_state->free_content();
        delete cpu_world_state;
        cpu_world_state = NULL;

        block_t *cpu_block = NULL;
        cpu_block = new block_t(arith, cpu_instances.block_data);
        cpu_block->free_content();
        delete cpu_block;
        cpu_block = NULL;

        keccak_t *keccak;
        keccak = new keccak_t(cpu_instances.sha3_parameters);
        keccak->free_parameters();
        delete keccak;
        keccak = NULL;

        transaction_t::free_instances(cpu_instances.transactions_data, cpu_instances.count);
        cpu_instances.transactions_data = NULL;

        accessed_state_t::free_cpu_instances(cpu_instances.accessed_states_data, cpu_instances.count);
        cpu_instances.accessed_states_data = NULL;

        touch_state_t::free_cpu_instances(cpu_instances.touch_states_data, cpu_instances.count);
        cpu_instances.touch_states_data = NULL;

        #ifdef TRACER
        tracer_t::free_cpu_instances(cpu_instances.tracers_data, cpu_instances.count);
        cpu_instances.tracers_data = NULL;
        #endif

        #ifndef ONLY_CPU
        CUDA_CHECK(cudaFree(cpu_instances.errors));
        #else
        delete[] cpu_instances.errors;
        #endif
        cpu_instances.errors = NULL;
    }

    __host__ static void print_evm_instances_t(
        arith_t &arith,
        evm_instances_t instances)
    {
        world_state_t *cpu_world_state;
        cpu_world_state = new world_state_t(arith, instances.world_state_data);
        printf("World state:\n");
        cpu_world_state->print();
        delete cpu_world_state;
        cpu_world_state = NULL;

        block_t *cpu_block = NULL;
        cpu_block = new block_t(arith, instances.block_data);
        printf("Block:\n");
        cpu_block->print();
        delete cpu_block;
        cpu_block = NULL;

        printf("Instances:\n");
        transaction_t *transaction;
        for (size_t idx = 0; idx < instances.count; idx++)
        {
            printf("Instance %lu\n", idx);
            transaction = new transaction_t(arith, &(instances.transactions_data[idx]));
            transaction->print();
            delete transaction;
            transaction = NULL;

            accessed_state_t::print_accessed_state_data_t(arith, instances.accessed_states_data[idx]);

            touch_state_t::print_touch_state_data_t(arith, instances.touch_states_data[idx]);

            #ifdef TRACER
            tracer_t::print_tracer_data_t(arith, instances.tracers_data[idx]);
            #endif

            printf("Error: %u\n", instances.errors[idx]);
            
        }
    }

    __host__ static cJSON *json_from_evm_instances_t(
        arith_t &arith,
        evm_instances_t instances)
    {
        cJSON *root = cJSON_CreateObject();

        world_state_t *cpu_world_state;
        cpu_world_state = new world_state_t(arith, instances.world_state_data);
        cJSON *world_state_json = cpu_world_state->json();
        cJSON_AddItemToObject(root, "pre", world_state_json);
        delete cpu_world_state;
        cpu_world_state = NULL;

        block_t *cpu_block = NULL;
        cpu_block = new block_t(arith, instances.block_data);
        cJSON *block_json = cpu_block->json();
        cJSON_AddItemToObject(root, "env", block_json);
        delete cpu_block;
        cpu_block = NULL;

        cJSON *instances_json = cJSON_CreateArray();
        cJSON_AddItemToObject(root, "post", instances_json);
        transaction_t *transaction;

        for (uint32_t idx=0; idx < instances.count; idx++)
        {
            cJSON *instance_json = cJSON_CreateObject();
            cJSON_AddItemToArray(instances_json, instance_json);
            transaction = new transaction_t(arith, &(instances.transactions_data[idx]));
            cJSON *transaction_json = transaction->json();
            cJSON_AddItemToObject(instance_json, "msg", transaction_json);
            delete transaction;
            transaction = NULL;

            cJSON *accessed_state_json = accessed_state_t::json_from_accessed_state_data_t(arith, instances.accessed_states_data[idx]);
            cJSON_AddItemToObject(instance_json, "access_state", accessed_state_json);

            cJSON *touch_state_json = touch_state_t::json_from_touch_state_data_t(arith, instances.touch_states_data[idx]);
            cJSON_AddItemToObject(instance_json, "touch_state", touch_state_json);

            #ifdef TRACER
            cJSON *tracer_json = tracer_t::json_from_tracer_data_t(arith, instances.tracers_data[idx]);
            cJSON_AddItemToObject(instance_json, "traces", tracer_json);
            #endif

            cJSON_AddItemToObject(instance_json, "error", cJSON_CreateNumber(instances.errors[idx]));
            cJSON_AddItemToObject(instance_json, "success", cJSON_CreateBool((instances.errors[idx] == ERR_NONE) || (instances.errors[idx] == ERR_RETURN) || (instances.errors[idx] == ERR_SUCCESS)));
        }
        return root;
    }
};

template <class params>
__global__ void kernel_evm(
    cgbn_error_report_t *report,
    typename evm_t<params>::evm_instances_t *instances)
{
    uint32_t instance = (blockIdx.x * blockDim.x + threadIdx.x) / params::TPI;

    if (instance >= instances->count)
        return;

    typedef arith_env_t<params> arith_t;
    typedef typename arith_t::bn_t bn_t;
    typedef evm_t<params> evm_t;

    // setup arith
    arith_t arith(
        cgbn_report_monitor,
        report,
        instance);

    // setup evm
    evm_t *evm = NULL;
    evm = new evm_t(
        arith,
        instances->world_state_data,
        instances->block_data,
        instances->sha3_parameters,
        &(instances->transactions_data[instance]),
        &(instances->accessed_states_data[instance]),
        &(instances->touch_states_data[instance]),
        #ifdef TRACER
        &(instances->tracers_data[instance]),
        #endif
        instance,
        &(instances->errors[instance]));

    uint32_t tmp_error_code;
    // run the evm
    evm->run(tmp_error_code);

    // free the evm
    delete evm;
    evm = NULL;
}

#endif