#ifndef _EVM_H_
#define _EVM_H_
#include <Python.h>
#include "include/utils.h"
#include "include/python_utils.cuh"
#include "stack.cuh"
#include "message.cuh"
#include "memory.cuh"
#include "returndata.cuh"
#include "block.cuh"
#include "tracer.cuh"
#include "state.cuh"
#include "keccak.cuh"
#include "jump_destinations.cuh"
#include "logs.cuh"
#include "alu_operations.cuh"
#include "env_operations.cuh"
#include "internal_operations.cuh"

class evm_t
{
public:

    /**
     * The block information data type.
     */
    typedef block_t::block_data_t block_data_t;
    /**
     * World state information class.
     */
    // typedef world_state_t<params> world_state_t;
    /**
     * World state information data type.
     */
    typedef world_state_t::state_data_t state_data_t;
    /**
     * The account information data.
     */
    typedef world_state_t::account_t account_t;

    /**
     * The access state data type.
     */
    typedef accessed_state_t::accessed_state_data_t accessed_state_data_t;

    /**
     * The touch state data type.
     */
    typedef touch_state_t::touch_state_data_t touch_state_data_t;

    /**
     * The transaction content data structure.
     */
    typedef typename transaction_t::transaction_data_t transaction_data_t;

    /**
     * The message content data structure.
     */
    typedef typename message_t::message_data_t message_data_t;

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
     * The logs state data type.
     */
    typedef log_state_t::log_state_data_t log_state_data_t;


    // constants
    static const uint32_t MAX_DEPTH = 1024; /**< The maximum call depth*/
    static const uint32_t MAX_EXECUTION_STEPS = 30000; /**< maximum number of execution steps TODO: DELETE*/
    static const uint32_t DEPTH_PAGE_SIZE = 32; /**< allocation size for depth variable like stack, memory, states, return data*/
    static const uint32_t MAX_CODE_SIZE = 24576; /**< EIP-170 Maximum contract size*/
    static const uint32_t MAX_INIT_CODE_SIZE = 2 * MAX_CODE_SIZE; /**< EIP-3860 Maximum initicode size*/
    /**
     * The numver of bytes in a hash.
     */
    static const uint32_t HASH_BYTES = 32;

    /**
     * The evm instances data structure.
    */
    typedef struct
    {
        state_data_t *world_state_data; /**< The world state content*/
        block_data_t *block_data; /**< The current block infomation*/
        sha3_parameters_t *sha3_parameters; /**< The constants for the KECCAK*/
        transaction_data_t *transactions_data; /**< The transactions information*/
        accessed_state_data_t *accessed_states_data; /**< The data cotaining the states access by the transactions execution*/
        touch_state_data_t *touch_states_data; /**< The data containing the states modified by the transactions execution*/
        log_state_data_t *logs_data; /**< The logs done by the transactions*/
#ifdef TRACER
        tracer_data_t *tracers_data; /**< Tracer datas for debug*/
#endif
        uint32_t *errors; /**< The the result of every transaction*/
        size_t count; /**< The number of instances/transactions*/
    } evm_instances_t;

    arith_t _arith; /**< The arithmetical environment*/
    world_state_t *_world_state; /**< The world state*/
    block_t *_block; /**< The current block*/
    keccak_t *_keccak; /**< The keccak object*/
    transaction_t *_transaction; /**< The current transaction*/
    accessed_state_t *_accessed_state; /**< The accessed state*/
    touch_state_t *_transaction_touch_state; /**< The final touch state of the transaction*/
    log_state_t *_transaction_log_state; /**< The final log state of the transaction*/
    uint32_t _instance; /**< The current instance/transaction number*/
#ifdef TRACER
    tracer_t *_tracer; /**< The tracer*/
    uint32_t _trace_pc; /**< The current program counter*/
    uint8_t _trace_opcode; /**< The current opcode*/
    bn_t _trace_address; /**< The current address of the executing context code*/
#endif
    touch_state_t **_touch_state_ptrs; /**< The touch states for every depth call*/
    log_state_t **_log_state_ptrs; /**< The log states for every depth call*/
    return_data_t **_last_return_data_ptrs; /**< The last return data for every depth call*/
    return_data_t *_final_return_data; /**< The final return data*/
    message_t **_message_ptrs; /**< The message call for every depth call*/
    memory_t **_memory_ptrs; /**< The memory for every depth call*/
    stack_t **_stack_ptrs; /**< The stack for every depth call*/
    bn_t *_gas_useds; /**< The current gas used for every depth call*/
    bn_t *_gas_refunds; /**< The current gas refunds for every depth call*/
    uint32_t *_pcs; /**< The current program counter for every depth call*/
    accessed_state_data_t *_final_accessed_state_data; /**< The final accessed state data*/
    touch_state_data_t *_final_touch_state_data; /**< The final touch state data*/
    log_state_data_t *_final_log_state_data; /**< The final log state data*/
    uint32_t _depth; /**< The current depth*/
    uint32_t _allocated_depth; /**< The allocated depth*/
    bn_t _gas_limit;        /**< The current gas limit YP: \f$T_{g}\f$*/
    bn_t _gas_price;        /**< The gas price YP: \f$p\f$ or \f$T_{p}\f$*/
    bn_t _gas_priority_fee; /**< The priority gas fee YP: \f$f\f$*/
    uint8_t *_bytecode;     /**< The current executing code YP: \f$I_{b}\f$*/
    uint32_t _code_size;   /**< The current executing code size*/
    uint8_t _opcode;       /**< The current opcode*/
    jump_destinations_t *_jump_destinations; /**< The jump destinations for the current execution context*/
    uint32_t _error_code; /**< The error code*/
    uint32_t *_final_error; /**< The final error code*/
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

    /**
     * The cosntructor for an evm instance.
     * @param[in] arith The arithmetical environment.
     * @param[in] world_state_data The world state data.
     * @param[in] block_data The block data.
     * @param[in] sha3_parameters The sha3 parameters.
     * @param[in] transaction_data The transaction data.
     * @param[out] accessed_state_data The accessed state data.
     * @param[out] touch_state_data The touch state data.
     * @param[out] log_state_data The log state data.
     * @param[out] tracer_data The tracer data.
     * @param[in] instance The instance number.
     * @param[out] error The error code.
     * @return The evm instance.
    */
    __host__ __device__ __forceinline__ evm_t(
        arith_t arith,
        state_data_t *world_state_data,
        block_data_t *block_data,
        sha3_parameters_t *sha3_parameters,
        transaction_data_t *transaction_data,
        accessed_state_data_t *accessed_state_data,
        touch_state_data_t *touch_state_data,
        log_state_data_t *log_state_data,
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
        _transaction_log_state = new log_state_t(arith);
        _final_accessed_state_data = accessed_state_data;
        _final_touch_state_data = touch_state_data;
        _final_log_state_data = log_state_data;
        _depth = 0;
        _allocated_depth = DEPTH_PAGE_SIZE;
        _touch_state_ptrs = new touch_state_t *[_allocated_depth];
        _log_state_ptrs = new log_state_t *[_allocated_depth];
        _last_return_data_ptrs = new return_data_t *[_allocated_depth];
        _final_return_data = new return_data_t();
        _message_ptrs = new message_t *[_allocated_depth];
        _memory_ptrs = new memory_t *[_allocated_depth];
        _stack_ptrs = new stack_t *[_allocated_depth];
        // TODO: infeficient but because of their form
        // we allocate them with maximum depth from the
        // begining
        _gas_useds = new bn_t[MAX_DEPTH];
        _gas_refunds = new bn_t[MAX_DEPTH];
        _pcs = new uint32_t[MAX_DEPTH];
        /*
        _gas_useds = new bn_t[_allocated_depth];
        _gas_refunds = new bn_t[_allocated_depth];
        _pcs = new uint32_t[_allocated_depth];
        */
#ifdef TRACER
        _tracer = new tracer_t(arith, tracer_data);
#endif
        _jump_destinations = NULL;
        _error_code = ERR_NONE;
    }

    /**
     * The destructor for an evm instance.
    */
    __host__ __device__ __forceinline__ ~evm_t()
    {
        // save the final data
        _accessed_state->to_accessed_state_data_t(*_final_accessed_state_data);
        _transaction_touch_state->to_touch_state_data_t(*_final_touch_state_data);
        _transaction_log_state->to_log_state_data_t(*_final_log_state_data);
        delete _world_state;
        delete _block;
        delete _keccak;
        delete _transaction;
        delete _accessed_state;
        delete _transaction_touch_state;
        delete _transaction_log_state;
#ifdef TRACER
        delete _tracer;
#endif
        delete[] _touch_state_ptrs;
        delete[] _log_state_ptrs;
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

    /**
     * Increase the allocation depth of the evm instance.
    */
    __host__ __device__ __forceinline__ void grow()
    {
        uint32_t new_allocated_depth = _allocated_depth + DEPTH_PAGE_SIZE;
        touch_state_t **new_touch_state_ptrs = new touch_state_t *[new_allocated_depth];
        log_state_t **new_log_state_ptrs = new log_state_t *[new_allocated_depth];
        return_data_t **new_return_data_ptrs = new return_data_t *[new_allocated_depth];
        message_t **new_message_ptrs = new message_t *[new_allocated_depth];
        memory_t **new_memory_ptrs = new memory_t *[new_allocated_depth];
        stack_t **new_stack_ptrs = new stack_t *[new_allocated_depth];
        /*
        bn_t *new_gas_useds = new bn_t[new_allocated_depth];
        bn_t *new_gas_refunds = new bn_t[new_allocated_depth];
        uint32_t *new_pcs = new uint32_t[new_allocated_depth];
        */

        memcpy(
            new_touch_state_ptrs,
            _touch_state_ptrs,
            _allocated_depth * sizeof(touch_state_t *));
        memcpy(
            new_log_state_ptrs,
            _log_state_ptrs,
            _allocated_depth * sizeof(log_state_t *));
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
        /*
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
        */

        delete[] _touch_state_ptrs;
        delete[] _log_state_ptrs;
        delete[] _last_return_data_ptrs;
        delete[] _message_ptrs;
        delete[] _memory_ptrs;
        delete[] _stack_ptrs;
        /*
        delete[] _gas_useds;
        delete[] _gas_refunds;
        delete[] _pcs;
        */
        _touch_state_ptrs = new_touch_state_ptrs;
        _log_state_ptrs = new_log_state_ptrs;
        _last_return_data_ptrs = new_return_data_ptrs;
        _message_ptrs = new_message_ptrs;
        _memory_ptrs = new_memory_ptrs;
        _stack_ptrs = new_stack_ptrs;
        /*
        _gas_useds = new_gas_useds;
        _gas_refunds = new_gas_refunds;
        _pcs = new_pcs;
        */
        _allocated_depth = new_allocated_depth;
    }

    /**
     * Init the evm instance for starting the transaction execution.
     * Finds the gas price, priority fee, total gas limit and
     * initialiased the gas used with the transaction initialisation
     * gas cost. Verify if is it a valid transaction.
     * Warm up the coinbase account.
     * @param[out] gas_used The gas used.
     * @param[out] error_code The error code.
     */
    __host__ __device__ void start_TRANSACTION(
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

        // EIP-3651 - Warm um coinbase account
        bn_t coin_base_address;
        _block->get_coin_base(coin_base_address);
        _accessed_state->get_account(coin_base_address, READ_BALANCE);
    }

    /**
     * Update the evm instance for the current depth message call.
     * It updates the gas limit, the bytecode, the code size and
     * the jump destinations.
     */
    __host__ __device__ void update_CALL()
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

    /**
     * Starts a new message call execution.
     * It update the evm instance for the current message call.
     * It allocate the memory, the stack, the touch state, the log state,
     * the return data. It sends the value from the sender to the receiver,
     * and warms up the accounts.
     * @param[out] error_code The error code.
    */
    __host__ __device__ void start_CALL(
        uint32_t &error_code)
    {
        // update the current context
        update_CALL();
        // allocate the memory, the stack, the touch state, the log state, the return data
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
        _log_state_ptrs[_depth] = new log_state_t(_arith);
        // reset the program counter, the gas used and the gas refunds
        _pcs[_depth] = 0;
        cgbn_set_ui32(_arith._env, _gas_useds[_depth], 0);
        cgbn_set_ui32(_arith._env, _gas_refunds[_depth], 0);

        // Gets the information of the sender and the receiver
        bn_t sender, receiver, value;
        bn_t sender_balance, receiver_balance;
        _message_ptrs[_depth]->get_sender(sender);
        _message_ptrs[_depth]->get_recipient(receiver);
        _message_ptrs[_depth]->get_value(value);
        uint32_t call_type;
        call_type = _message_ptrs[_depth]->get_call_type();

        // in create call verify if the the account at the
        // address is not a contract
        if ((call_type == OP_CREATE) ||
            (call_type == OP_CREATE2))
        {
            if (_touch_state_ptrs[_depth]->is_contract(receiver))
            {
                error_code = ERROR_MESSAGE_CALL_CREATE_CONTRACT_EXISTS;
                return;
            }
            // set the account nonce to 1
            bn_t contract_nonce;
            cgbn_set_ui32(_arith._env, contract_nonce, 1);
            _touch_state_ptrs[_depth]->set_account_nonce(receiver, contract_nonce);
        }

        // Transfer the value from sender to receiver
        if ((cgbn_compare_ui32(_arith._env, value, 0) > 0) &&     // value>0
            (cgbn_compare(_arith._env, sender, receiver) != 0) && // sender != receiver
            (call_type != OP_DELEGATECALL)                        // no delegatecall
        )
        {
            _touch_state_ptrs[_depth]->get_account_balance(sender, sender_balance);
            _touch_state_ptrs[_depth]->get_account_balance(receiver, receiver_balance);
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
        // if is a call to a non-contract account
        // if code size is zero. TODO: verify if is consider a last return data
        // only for calls not for create
        if ((_code_size == 0) &&
            (call_type != OP_CREATE) &&
            (call_type != OP_CREATE2))
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
        /**
         * Get if a message call is valid.
         * @param[in] arith The arithmetical environment.
         * @param[in] message The message call.
         * @param[in] touch_state The touch state.
         * @return 1 if the message call is valid, 0 otherwise.
         */
        __host__ __device__ __forceinline__ static int32_t valid_CALL(
            arith_t &arith,
            message_t &message,
            touch_state_t &touch_state)
        {
            bn_t sender, receiver, value;
            bn_t sender_balance;
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
            if ((cgbn_compare_ui32(arith._env, value, 0) > 0) && // value>0
                                                                 //(cgbn_compare(arith._env, sender, receiver) != 0) &&   // sender != receiver matter only on transfer
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
        __host__ __device__ __forceinline__ static int32_t valid_CREATE(
            arith_t &arith,
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
        __host__ __device__ __forceinline__ static void generic_CALL(
            arith_t &arith,
            bn_t &gas_limit,
            bn_t &gas_used,
            uint32_t &error_code,
            stack_t &stack,
            message_t &message,
            memory_t &memory,
            touch_state_t &touch_state,
            evm_t &evm,
            message_t &new_message,
            bn_t &args_offset,
            bn_t &args_size,
            return_data_t &return_data)
        {
            // try to send value in static call
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
        __host__ __device__ __forceinline__ static void generic_CREATE(
            arith_t &arith,
            bn_t &gas_limit,
            bn_t &gas_used,
            uint32_t &error_code,
            stack_t &stack,
            message_t &message,
            memory_t &memory,
            touch_state_t &touch_state,
            evm_t &evm,
            message_t &new_message,
            bn_t &args_offset,
            bn_t &args_size,
            return_data_t &return_data)
        {
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
                // set the init code
                SHARED_MEMORY data_content_t initialisation_code;
                arith.size_t_from_cgbn(initialisation_code.size, args_size);
                initialisation_code.data = memory.get(
                    args_offset,
                    args_size,
                    error_code);
                new_message.set_byte_code(
                    initialisation_code.data,
                    initialisation_code.size);

                // set the gas limit
                bn_t gas_capped;
                arith.max_gas_call(gas_capped, gas_limit, gas_used);
                new_message.set_gas_limit(gas_capped);

                // add to gas used
                cgbn_add(arith._env, gas_used, gas_used, gas_capped);

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
                    cgbn_sub(arith._env, gas_used, gas_used, gas_capped);
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
        __host__ __device__ __forceinline__ static void operation_STOP(
            return_data_t &return_data,
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
         * @param[in] keccak The keccak object.
         * @param[out] evm The evm.
         * @param[out] return_data The return data.
         */
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
            keccak_t &keccak,
            evm_t &evm,
            return_data_t &return_data)
        {
            bn_t value, memory_offset, length;
            stack.pop(value, error_code);
            stack.pop(memory_offset, error_code);
            stack.pop(length, error_code);

            // create cost
            cgbn_add_ui32(arith._env, gas_used, gas_used, GAS_CREATE);

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
                    sender_nonce,
                    keccak);

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
            evm_t &evm,
            return_data_t &return_data)
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
            evm_t &evm,
            return_data_t &return_data)
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
            evm_t &evm,
            return_data_t &return_data)
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
         * @param[in] keccak The keccak object.
         * @param[out] evm The evm.
         * @param[out] return_data The return data.
         */
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
            keccak_t &keccak,
            evm_t &evm,
            return_data_t &return_data)
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
                SHARED_MEMORY data_content_t initialisation_code;

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
                    initialisation_code,
                    keccak);

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
            evm_t &evm,
            return_data_t &return_data)
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
        __host__ __device__ __forceinline__ static void operation_INVALID(
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
        __host__ __device__ __forceinline__ static void operation_SELFDESTRUCT(
            arith_t &arith,
            bn_t &gas_limit,
            bn_t &gas_used,
            uint32_t &error_code,
            uint32_t &pc,
            stack_t &stack,
            message_t &message,
            touch_state_t &touch_state,
            return_data_t &return_data,
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
    };

    /**
     * Run the transaction execution.
     * @param[out] error_code The error code.
    */
    __host__ __device__ void run(
        uint32_t &error_code)
    {
        // get the first message call from transaction
        _message_ptrs[_depth] = _transaction->get_message_call(*_accessed_state, *_keccak);
        // process the transaction
        bn_t intrsinc_gas_used;
        start_TRANSACTION(intrsinc_gas_used, error_code);
        start_CALL(error_code);
        // if it is a invalid transaction or not enough gas to start the call
        if (error_code != ERR_NONE)
        {
            finish_TRANSACTION(error_code);
            free_CALL();
            return;
        }
        // add the transaction cost
        cgbn_add(_arith._env, _gas_useds[_depth], _gas_useds[_depth], intrsinc_gas_used);
        // run the message call
        uint32_t execution_step = 0;
        while (
            (execution_step < MAX_EXECUTION_STEPS))
        {

            // if the program counter is out of bounds
            // it is a STOP operation
            if (_pcs[_depth] >= _code_size)
            {
                _opcode = OP_STOP;
            }
            else
            {
                _opcode = _bytecode[_pcs[_depth]];
            }
            ONE_THREAD_PER_INSTANCE(
                /*printf("pc: %d opcode: %d\n", _pcs[_depth], _opcode);*/)
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
                    *_memory_ptrs[_depth],
                    *_message_ptrs[_depth],
                    *_log_state_ptrs[_depth],
                    _opcode);
            }
            else
            {
                // Depending on the opcode execute the operation
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
                break;
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
                        *_keccak,
                        *this,
                        *_last_return_data_ptrs[_depth]);
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
                        *this,
                        *_last_return_data_ptrs[_depth]);
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
                        *this,
                        *_last_return_data_ptrs[_depth]);
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
                            *_last_return_data_ptrs[_depth - 1]);
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
                        *this,
                        *_last_return_data_ptrs[_depth]);
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
                        *_keccak,
                        *this,
                        *_last_return_data_ptrs[_depth]);
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
                        *this,
                        *_last_return_data_ptrs[_depth]);
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
                            *_last_return_data_ptrs[_depth - 1]);
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
                    if (_depth == 0)
                    {
                        system_operations::operation_SELFDESTRUCT(
                            _arith,
                            _gas_limit,
                            _gas_useds[_depth],
                            error_code,
                            _pcs[_depth],
                            *_stack_ptrs[_depth],
                            *_message_ptrs[_depth],
                            *_touch_state_ptrs[_depth],
                            *_final_return_data,
                            *this);
                    }
                    else
                    {
                        system_operations::operation_SELFDESTRUCT(
                            _arith,
                            _gas_limit,
                            _gas_useds[_depth],
                            error_code,
                            _pcs[_depth],
                            *_stack_ptrs[_depth],
                            *_message_ptrs[_depth],
                            *_touch_state_ptrs[_depth],
                            *_last_return_data_ptrs[_depth - 1],
                            *this);
                    }
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

            // If the operation ended with halting
            // can be normal or exceptional
            if (error_code != ERR_NONE)
            {
                // FIRST verify if is a createX operation and success
                if (
                    (error_code == ERR_RETURN) &&
                    ((_message_ptrs[_depth]->get_call_type() == OP_CREATE) ||
                     (_message_ptrs[_depth]->get_call_type() == OP_CREATE2)))
                {
                    finish_CREATEX(error_code);
                }

                // if it is the root call
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
                    finish_TRANSACTION(error_code);
                    free_CALL();
                    return;
                }
                else
                {
                    finish_CALL(error_code);
                    free_CALL();
                    _depth = _depth - 1;
                    update_CALL();
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

    /**
     * Finish the CREATEX operation
     * @param[out] error_code error code
     */
    __host__ __device__ __forceinline__ void finish_CREATEX(
        uint32_t &error_code)
    {
        // compute the gas to deposit the contract
        bn_t gas_value;
        cgbn_set_ui32(_arith._env, gas_value, GAS_CODE_DEPOSIT);
        bn_t code_size;

        if (_depth > 0)
        {
            _arith.cgbn_from_size_t(code_size, _last_return_data_ptrs[_depth - 1]->size());
        }
        else
        {
            _arith.cgbn_from_size_t(code_size, _final_return_data->size());
        }
        cgbn_mul(_arith._env, gas_value, gas_value, code_size);
        cgbn_add(_arith._env, _gas_useds[_depth], _gas_useds[_depth], gas_value);
        uint32_t tmp_error_code;
        tmp_error_code = ERR_NONE;

        // if enough gas set the bytecode for the contract
        // and the nonce of the new contract
        if (_arith.has_gas(_gas_limit, _gas_useds[_depth], tmp_error_code))
        {
            // compute the address of the contract
            bn_t contract_address;
            _message_ptrs[_depth]->get_recipient(contract_address);
            uint8_t *code;
            size_t code_size;
            if (_depth > 0)
            {
                code = _last_return_data_ptrs[_depth - 1]->get_data()->data;
                code_size = _last_return_data_ptrs[_depth - 1]->size();
            }
            else
            {
                code = _final_return_data->get_data()->data;
                code_size = _final_return_data->size();
            }
            if (code_size <= MAX_CODE_SIZE)
            {
                if ((code_size > 0) && (code[0] == 0xef)) // EIP-3541
                {
                    error_code = ERROR_CREATE_CODE_FIRST_BYTE_INVALID;
                }
                else
                {
                    // set the bytecode
                    _touch_state_ptrs[_depth]->set_account_code(
                        contract_address,
                        code,
                        code_size);
                    // the balance and the nonce is done at the begining of the call
                }
            }
            else
            {
                error_code = ERROR_CREATE_CODE_SIZE_EXCEEDED;
            }
        }
        else
        {
            error_code = tmp_error_code;
        }
    }

    /**
     * Finish the CALL operation or CREATE CALL operation.
     * @param[out] error_code error code
     */
    __host__ __device__ __forceinline__ void finish_CALL(
        uint32_t &error_code)
    {
        bn_t child_success;
        // set the child call to failure
        cgbn_set_ui32(_arith._env, child_success, 0);
        // if the child call return from normal halting
        // no errors
        if ((error_code == ERR_RETURN) || (error_code == ERR_REVERT))
        {
            // give back the gas left from the child computation
            bn_t gas_left;
            cgbn_sub(_arith._env, gas_left, _gas_limit, _gas_useds[_depth]);
            cgbn_sub(_arith._env, _gas_useds[_depth - 1], _gas_useds[_depth - 1], gas_left);

            // if is a succesfull call
            if (error_code == ERR_RETURN)
            {
                // update the parent state with the states of the child
                _touch_state_ptrs[_depth - 1]->update_with_child_state(
                    *_touch_state_ptrs[_depth]);
                _log_state_ptrs[_depth - 1]->update_with_child_state(
                    *_log_state_ptrs[_depth]);
                // sum the refund gas
                cgbn_add(
                    _arith._env,
                    _gas_refunds[_depth - 1],
                    _gas_refunds[_depth - 1],
                    _gas_refunds[_depth]);
                // for CALL operations set the child success to 1
                cgbn_set_ui32(_arith._env, child_success, 1);
                // if CREATEX operation, set the address of the contract
                if (
                    (_message_ptrs[_depth]->get_call_type() == OP_CREATE) ||
                    (_message_ptrs[_depth]->get_call_type() == OP_CREATE2))
                {
                    _message_ptrs[_depth]->get_recipient(child_success);
                }
            }
        }
        // reset the gas used and gas refund in the child
        cgbn_set_ui32(_arith._env, _gas_useds[_depth], 0);
        cgbn_set_ui32(_arith._env, _gas_refunds[_depth], 0);
        // get the memory offset and size of the return data
        // in the parent memory
        bn_t ret_offset, ret_size;
        _message_ptrs[_depth]->get_return_data_offset(ret_offset);
        _message_ptrs[_depth]->get_return_data_size(ret_size);
        // reset the error code for the parent
        error_code = ERR_NONE;

        // push the result in the parent stack
        _stack_ptrs[_depth - 1]->push(child_success, error_code);
        // set the parent memory with the return data
        bn_t return_data_index;
        cgbn_set_ui32(_arith._env, return_data_index, 0);
        uint8_t *data;
        size_t data_size;
        data = _arith.get_data(
            *(_last_return_data_ptrs[_depth - 1]->get_data()),
            return_data_index,
            ret_size,
            data_size);

        // It writes on memory even if the call was reverted
        _memory_ptrs[_depth - 1]->set(
            data,
            ret_offset,
            ret_size,
            data_size,
            error_code);
    }

    /**
     * Free the memory allocated for the CALL/CREATEX operation.
     */
    __host__ __device__ __forceinline__ void free_CALL()
    {
        delete _stack_ptrs[_depth];
        _stack_ptrs[_depth] = NULL;
        delete _memory_ptrs[_depth];
        _memory_ptrs[_depth] = NULL;
        delete _last_return_data_ptrs[_depth];
        _last_return_data_ptrs[_depth] = NULL;
        // delete the touch state
        delete _touch_state_ptrs[_depth];
        _touch_state_ptrs[_depth] = NULL;
        delete _log_state_ptrs[_depth];
        _log_state_ptrs[_depth] = NULL;
        // delete the message
        delete _message_ptrs[_depth];
        _message_ptrs[_depth] = NULL;
    }

    /**
     * Finish the transaction.
     * @param[out] error_code error code
     */
    __host__ __device__ __forceinline__ void finish_TRANSACTION(
        uint32_t &error_code)
    {
        // sent the gas value to the block beneficiary
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
            // send back the gas left and gas refund to the sender
            _transaction->get_sender(sender_address);
            _transaction_touch_state->get_account_balance(sender_address, sender_balance);
            cgbn_add(_arith._env, sender_balance, sender_balance, gas_value);
            _transaction_touch_state->set_account_balance(sender_address, sender_balance);

            // the gas value for the beneficiary is \f$T_{g} - g^{*}\f$
            cgbn_sub(_arith._env, gas_value, _gas_limit, gas_value);

            // update the transaction state
            _transaction_touch_state->update_with_child_state(
                *_touch_state_ptrs[_depth]);
            _transaction_log_state->update_with_child_state(
                *_log_state_ptrs[_depth]);
            // set the eror code for a succesfull transaction
            _error_code = ERR_NONE;
        }
        else
        {
            cgbn_mul(_arith._env, gas_value, _gas_limit, _gas_priority_fee);
            // set z to the given error or 1 TODO: 1 in YP
            _error_code = error_code;
        }
        // send the gas value to the beneficiary
        bn_t beneficiary_balance;
        _transaction_touch_state->get_account_balance(beneficiary, beneficiary_balance);
        cgbn_add(_arith._env, beneficiary_balance, beneficiary_balance, gas_value);
        _transaction_touch_state->set_account_balance(beneficiary, beneficiary_balance);

        // update the final state modification done by the transaction
        _transaction_touch_state->to_touch_state_data_t(
            *_final_touch_state_data);
        _accessed_state->to_accessed_state_data_t(
            *_final_accessed_state_data);
        *_final_error = _error_code;
        delete _jump_destinations;
        _jump_destinations = NULL;
    }

    /**
     * Make a CALL/CREATEX call by increasing the depth.
     * @param[out] error_code error code
     * @param[in] new_message new message call
    */
    __host__ __device__ __forceinline__ void child_CALL(
        uint32_t &error_code,
        message_t &new_message)
    {
            // increase depth and allocate memory if necessary
        _depth = _depth + 1;
        if (_depth == _allocated_depth)
        {
            grow();
        }
        // setup the new message call and start the execution of the call
        _message_ptrs[_depth] = &new_message;
        start_CALL(error_code);
    }

    /**
     * Get the cpu instances from the json test.
     * @param[out] instances evm instances
     * @param[in] test json test
    */
    __host__ static void get_cpu_instances(
        evm_instances_t &instances,
        const cJSON *test)
    {
        //setup the arithmetic environment
        arith_t arith(cgbn_report_monitor, 0);

        // get the world state
        world_state_t *cpu_world_state;
        cpu_world_state = new world_state_t(arith, test);
        instances.world_state_data = cpu_world_state->_content;
        delete cpu_world_state;
        cpu_world_state = NULL;

        // ge the current block
        block_t *cpu_block = NULL;
        cpu_block = new block_t(arith, test);
        instances.block_data = cpu_block->_content;
        delete cpu_block;
        cpu_block = NULL;

        // setup the keccak paramameters
        keccak_t *keccak;
        keccak = new keccak_t();
        instances.sha3_parameters = keccak->_parameters;
        delete keccak;
        keccak = NULL;

        // get the transactions
        transaction_t::get_transactions(instances.transactions_data, test, instances.count);

        // allocated the memory for accessed states
        instances.accessed_states_data = accessed_state_t::get_cpu_instances(instances.count);

        // allocated the memory for touch states
        instances.touch_states_data = touch_state_t::get_cpu_instances(instances.count);

        // allocated the memory for logs
        instances.logs_data = log_state_t::get_cpu_instances(instances.count);

#ifdef TRACER
        // allocated the memory for tracers
        instances.tracers_data = tracer_t::get_cpu_instances(instances.count);
#endif

        // alocate the memory for the result of the transactions
#ifndef ONLY_CPU
        CUDA_CHECK(cudaMallocManaged(
            (void **)&(instances.errors),
            sizeof(uint32_t) * instances.count));
#else
        instances.errors = new uint32_t[instances.count];
#endif
        memset(instances.errors, ERR_NONE, sizeof(uint32_t) * instances.count);
    }
    __host__ static void get_cpu_instances_plain_data(
        evm_instances_t &instances,
        state_data_t* state_data,
        block_data_t* block_data,
        transaction_data_t* transactions_data,
        size_t count
        )
    {
        //setup the arithmetic environment
        arith_t arith(cgbn_report_monitor, 0);
        instances.world_state_data = state_data;
        instances.block_data = block_data;
        // setup the keccak paramameters
        keccak_t *keccak;
        keccak = new keccak_t();
        instances.sha3_parameters = keccak->_parameters;
        delete keccak;
        keccak = NULL;

        // get the transactions
        instances.transactions_data = transactions_data;

        instances.count = count;

        // allocated the memory for accessed states
        instances.accessed_states_data = accessed_state_t::get_cpu_instances(instances.count);

        // allocated the memory for touch states
        instances.touch_states_data = touch_state_t::get_cpu_instances(instances.count);

        // allocated the memory for logs
        instances.logs_data = log_state_t::get_cpu_instances(instances.count);

#ifdef TRACER
        // allocated the memory for tracers
        instances.tracers_data = tracer_t::get_cpu_instances(instances.count);
#endif

        // alocate the memory for the result of the transactions
#ifndef ONLY_CPU
        CUDA_CHECK(cudaMallocManaged(
            (void **)&(instances.errors),
            sizeof(uint32_t) * instances.count));
#else
        instances.errors = new uint32_t[instances.count];
#endif
        memset(instances.errors, ERR_NONE, sizeof(uint32_t) * instances.count);
    }

/*
    __host__ static void get_cpu_instances_pyobject(
            evm_instances_t &instances,
            const PyObject *test)
        {
            //setup the arithmetic environment
            arith_t arith(cgbn_report_monitor, 0);

            state_data_t* state_data = 
            // get the world state
            world_state_t *cpu_world_state;
            cpu_world_state = new world_state_t(arith, test);
            instances.world_state_data = cpu_world_state->_content;
            delete cpu_world_state;
            cpu_world_state = NULL;

            // ge the current block
            block_t *cpu_block = NULL;
            cpu_block = new block_t(arith, test);
            instances.block_data = cpu_block->_content;
            delete cpu_block;
            cpu_block = NULL;

            // setup the keccak paramameters
            keccak_t *keccak;
            keccak = new keccak_t();
            instances.sha3_parameters = keccak->_parameters;
            delete keccak;
            keccak = NULL;

            // get the transactions
            transaction_t::get_transactions(instances.transactions_data, test, instances.count);

            // allocated the memory for accessed states
            instances.accessed_states_data = accessed_state_t::get_cpu_instances(instances.count);

            // allocated the memory for touch states
            instances.touch_states_data = touch_state_t::get_cpu_instances(instances.count);

            // allocated the memory for logs
            instances.logs_data = log_state_t::get_cpu_instances(instances.count);

    #ifdef TRACER
            // allocated the memory for tracers
            instances.tracers_data = tracer_t::get_cpu_instances(instances.count);
    #endif

            // alocate the memory for the result of the transactions
    #ifndef ONLY_CPU
            CUDA_CHECK(cudaMallocManaged(
                (void **)&(instances.errors),
                sizeof(uint32_t) * instances.count));
    #else
            instances.errors = new uint32_t[instances.count];
    #endif
            memset(instances.errors, ERR_NONE, sizeof(uint32_t) * instances.count);
        }

*/
    /**
     * Get the gpu instances from the cpu instances.
     * @param[out] gpu_instances evm instances
     * @param[in] cpu_instances evm instances
    */
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

        gpu_instances.logs_data = log_state_t::get_gpu_instances_from_cpu_instances(cpu_instances.logs_data, cpu_instances.count);

#ifdef TRACER
        gpu_instances.tracers_data = tracer_t::get_gpu_instances_from_cpu_instances(cpu_instances.tracers_data, cpu_instances.count);
#endif

        gpu_instances.errors = cpu_instances.errors;
    }

    /**
     * Get the cpu instances from the gpu instances.
     * @param[out] cpu_instances evm instances
     * @param[in] gpu_instances evm instances
    */
    __host__ static void get_cpu_instances_from_gpu_instances(
        evm_instances_t &cpu_instances,
        evm_instances_t &gpu_instances)
    {
        cpu_instances.count = gpu_instances.count;

        cpu_instances.world_state_data = gpu_instances.world_state_data;
        cpu_instances.block_data = gpu_instances.block_data;
        cpu_instances.sha3_parameters = gpu_instances.sha3_parameters;
        cpu_instances.transactions_data = gpu_instances.transactions_data;
        accessed_state_t::free_cpu_instances(cpu_instances.accessed_states_data, cpu_instances.count);
        cpu_instances.accessed_states_data = accessed_state_t::get_cpu_instances_from_gpu_instances(gpu_instances.accessed_states_data, gpu_instances.count);
        touch_state_t::free_cpu_instances(cpu_instances.touch_states_data, cpu_instances.count);
        cpu_instances.touch_states_data = touch_state_t::get_cpu_instances_from_gpu_instances(gpu_instances.touch_states_data, gpu_instances.count);
        log_state_t::free_cpu_instances(cpu_instances.logs_data, cpu_instances.count);
        cpu_instances.logs_data = log_state_t::get_cpu_instances_from_gpu_instances(gpu_instances.logs_data, gpu_instances.count);
#ifdef TRACER
        tracer_t::free_cpu_instances(cpu_instances.tracers_data, cpu_instances.count);
        cpu_instances.tracers_data = tracer_t::get_cpu_instances_from_gpu_instances(gpu_instances.tracers_data, gpu_instances.count);
#endif
        cpu_instances.errors = gpu_instances.errors;
    }

    /**
     * Free the cpu instances.
     * @param[in] cpu_instances evm instances
    */
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

        log_state_t::free_cpu_instances(cpu_instances.logs_data, cpu_instances.count);
        cpu_instances.logs_data = NULL;

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

    /**
     * Print the evm instances after the transaction execution.
     * @param[in] arith arithmetic environment
     * @param[in] instances evm instances
    */
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
        for (size_t idx = 0; idx < instances.count; idx++)
        {
            printf("Instance %lu\n", idx);
            transaction_t::print_transaction_data_t(arith, instances.transactions_data[idx]);

            accessed_state_t::print_accessed_state_data_t(arith, instances.accessed_states_data[idx]);

            touch_state_t::print_touch_state_data_t(arith, instances.touch_states_data[idx]);

            log_state_t::print_log_state_data_t(arith, instances.logs_data[idx]);

#ifdef TRACER
            tracer_t::print_tracer_data_t(arith, instances.tracers_data[idx]);
#endif

            printf("Error: %u\n", instances.errors[idx]);
        }
    }

    /**
     * Get the pyobject from the evm instances after the transaction execution.
     * @param[in] arith arithmetic environment
     * @param[in] instances evm instances
     * @return pyobject
    */
    __host__ static PyObject* pyobject_from_evm_instances_t(arith_t &arith, evm_instances_t instances) {
        PyObject* root = PyDict_New();

        // world_state_t::pyDict_from_state_data_t(arith, instances.world_state_data);
        world_state_t *cpu_world_state = new world_state_t(arith, instances.world_state_data);
        PyObject* world_state_json = cpu_world_state->toPyObject();  // Assuming toPyObject() is implemented
        PyDict_SetItemString(root, "pre", world_state_json);
        Py_DECREF(world_state_json);
        delete cpu_world_state;


        PyObject* instances_json = PyList_New(0);
        PyDict_SetItemString(root, "post", instances_json);
        Py_DECREF(instances_json);  // Decrement here because PyDict_SetItemString increases the ref count

        for (uint32_t idx = 0; idx < instances.count; idx++) {
            PyObject* instance_json = PyDict_New();
            PyList_Append(instances_json, instance_json);  // Appends and steals the reference, so no need to DECREF

            #ifdef TRACER
            PyObject* tracer_json = tracer_t::pyObject_from_tracer_data_t(arith, instances.tracers_data[idx]);  
            PyDict_SetItemString(instance_json, "traces", tracer_json);
            Py_DECREF(tracer_json);
            #endif

            PyDict_SetItemString(instance_json, "error", PyLong_FromLong(instances.errors[idx]));
            PyDict_SetItemString(instance_json, "success", PyBool_FromLong((instances.errors[idx] == ERR_NONE) || (instances.errors[idx] == ERR_RETURN) || (instances.errors[idx] == ERR_SUCCESS)));
        }

        return root;
    }


    /**
     * Get the json from the evm instances after the transaction execution.
     * @param[in] arith arithmetic environment
     * @param[in] instances evm instances
     * @return json
    */
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

        for (uint32_t idx = 0; idx < instances.count; idx++)
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

            cJSON *log_state_json = log_state_t::json_from_log_state_data_t(arith, instances.logs_data[idx]);
            cJSON_AddItemToObject(instance_json, "log_state", log_state_json);

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

/**
 * The evm kernel running the transactions on the GPU.
 * @param[out] report error report
 * @param[in] instances evm instances
*/
template <class params>
__global__ void kernel_evm(
    cgbn_error_report_t *report,
    typename evm_t::evm_instances_t *instances)
{
    uint32_t instance = (blockIdx.x * blockDim.x + threadIdx.x) / params::TPI;

    if (instance >= instances->count)
        return;

    // setup arith
    arith_t arith(
        cgbn_report_monitor,
        report,
        instance);

    // setup evm
    evm_t evm (
        arith,
        instances->world_state_data,
        instances->block_data,
        instances->sha3_parameters,
        &(instances->transactions_data[instance]),
        &(instances->accessed_states_data[instance]),
        &(instances->touch_states_data[instance]),
        &(instances->logs_data[instance]),
#ifdef TRACER
        &(instances->tracers_data[instance]),
#endif
        instance,
        &(instances->errors[instance]));

    uint32_t tmp_error_code;
    tmp_error_code = ERR_NONE;
    // run the evm
    evm.run(tmp_error_code);
}

#endif
