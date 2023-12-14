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
     * The CGBN wide type with double the given number of bits in environment.
     */
    typedef typename env_t::cgbn_wide_t bn_wide_t;
    /**
     * The arbitrary length integer type used for the storage.
     * It is defined as the EVM word type.
     */
    typedef cgbn_mem_t<params::BITS> evm_word_t;
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
    typedef access_state_t<params> access_state_t;
    /**
     * The access state data type.
    */
    typedef access_state_t::accessed_state_data_t accessed_state_data_t;
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
     * The return data class.
    */
    typedef return_data_t<params> return_data_t;

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
    #endif
    touch_state_t **_touch_state_ptrs;
    return_data_t **_last_return_data_ptrs;
    return_data_t *_final_return_data;
    message_t **_message_ptrs;
    stack_t **_stack_ptrs;
    bn_t *_gas_useds;
    bn_t *_gas_refunds;
    uint32_t *_pcs;
    accessed_state_data_t *_final_accessed_state_data;
    touch_state_data_t *_final_touch_state_data;
    uint32_t _depth;
    uint32_t _allocated_depth;
    bn_t _gas_limit; /** YP: \f$T_{g}\f$*/
    bn_t _gas_price; /**< YP: \f$p\f$ or \f$T_{p}\f$*/
    bn_t _gas_priority_fee; /**< YP: \f$f\f$*/
    bn_t _storage_address;
    bn_t _contract_address;
    uint8_t *_bytecode;
    uint32_t _code_size;
    uint8_t _opcode;
    jump_destinations_t *_jump_destinations;
    uint32_t _error_code;


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
        uint32_t *error
    ) : _arith(arith), _instance(instance), _final_error(error)
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
        delete[] _stack_ptrs;
        delete[] _gas_useds;
        delete[] _gas_refunds;
        delete[] _pcs;
        _touch_state_ptrs = new_touch_state_ptrs;
        _last_return_data_ptrs = new_return_data_ptrs;
        _message_ptrs = new_message_ptrs;
        _stack_ptrs = new_stack_ptrs;
        _gas_useds = new_gas_useds;
        _gas_refunds = new_gas_refunds;
        _pcs = new_pcs;
        _allocated_depth = new_allocated_depth;
    }

    __host__ __device__ void process_transaction(
        bn_t &gas_used, /**< YP: \f$g_{0}\f$*/
        uint32_t &error_code
    )
    {
        bn_t block_base_fee; // YP: \f$H_{f}\f$
        _block->get_base_fee(block_base_fee);
        bn_t block_gas_limit;
        _block->get_gas_limit(block_gas_limit);
        _transaction->validate_transaction(
            _transaction_touch_state,
            gas_used,
            _gas_price,
            _gas_priority_fee,
            error_code,
            block_base_fee,
            block_gas_limit);
    }

    __host__ __device__ void update_env(
        uint32_t &error_code
    )
    {
        _message_ptrs[_depth]->get_gas(_gas_limit);
        _message_ptrs[_depth]->get_storage_address(_storage_address);
        _message_ptrs[_depth]->get_contract_address(_contract_address);
        if (
            (_message_ptrs[_depth]->get_call_type() == OP_CREATE) ||
            (_message_ptrs[_depth]->get_call_type() == OP_CREATE2)
        )
        {
            _bytecode = _message_ptrs[_depth]->_content->data.data;
            _code_size = _message_ptrs[_depth]->_content->data.size;
        } else {
            account_t *contract;
            contract = _touch_state_ptrs[_depth]->get_account(_contract_address, READ_CODE);
            _bytecode = contract->bytecode;
            _code_size = contract->code_size;
        }
        if (_jump_destinations != NULL)
        {
            delete _jump_destinations;
            _jump_destinations = NULL;
        }
        _jump_destinations = new jump_destinations_t(_bytecode, _code_size);
    }

    __host__ __device__ void init_message_call(
        uint32_t &error_code
    )
    {
        update_env(error_code);
        _last_return_data_ptrs[_depth] = new return_data_t();
        _stacks[_depth] = new stack_t(_arith);
        if (_depth > 0) {
            _touch_state_ptrs[_depth] = new touch_state_t(_accessed_state, _touch_state_ptrs[_depth - 1], );
        } else {
            _touch_state_ptrs[_depth] = new touch_state_t(_accessed_state, _transaction_touch_state);
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
        if ( (cgbn_compare_ui32(_arith._env, value, 0) > 0) && // value>0
                (cgbn_compare(_arith._env, sender, receiver) != 0) && // sender != receiver
                (_message_ptrs[_depth]->call_type() != OP_DELEGATECALL ) // no delegatecall
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
        account = _touch_state_ptrs[_depth]->get_account(_contract_address, READ_NONE);
        account = _touch_state_ptrs[_depth]->get_account(_storage_address, READ_NONE);
        account = NULL;
    }

   

    __host__ __device__ __forceinline__ static void operation_logx(
        arith_t &arith,
        bn_t &gas_limit,
        bn_t &gas_used,
        uint32_t &error_code,
        uint32_t &pc,
        stack_t &stack,
        uint8_t &opcode
    )
    {
        uint8_t log_index = opcode & 0x0F;
        error_code = ERR_NOT_IMPLEMENTED;
    }

    __host__ __device__ __forceinline__ static void operation_stop(
        return_data_t &return_data,
        uint32_t &error_code
    )
    {
        return_data.set(
            NULL,
            0);
        error_code = ERR_RETURN;
    }


    __host__ __device__ void run(
        uint32_t error_code
    )
    {
        // get the first message call from transaction
        _message_ptrs[_depth] = _transaction->get_message_call();
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
        uint32_t trace_pc;
        uint32_t execution_step = 0;
        while (
            ((_pcs[_depth] < _code_size) ||
            (_depth > 0)) &&
            (execution_step < MAX_EXECUTION_STEPS)
        )
        {

            _opcode = _bytecode[_pcs[_depth]];
            ONE_THREAD_PER_INSTANCE(
                printf("pc: %d opcode: %d\n", pc, opcode);)
#ifdef TRACER
            trace_pc = pc;
#endif
            // PUSHX
            if (((_opcode & 0xF0) == 0x60) || ((_opcode & 0xF0) == 0x70))
            {
                operation_pushx();
            }
            else if ((_opcode & 0xF0) == 0x80) // DUPX
            {
                operation_dupx();
            }
            else if ((opcode & 0xF0) == 0x90) // SWAPX
            {
                operation_swapx();
            }
            else if ((opcode & 0xF0) == 0xA0) // LOGX
            {
                operation_logx();
            }
            else
            {
                switch (opcode)
                {
                case OP_STOP: // STOP
                {
                    operation_stop();
                }
                break;
                case OP_ADD: // ADD
                {
                    cgbn_set_ui32(_arith._env, gas_cost, 3);
                    stack.add(error_code);
                }
                break;
                case OP_MUL: // MUL
                {
                    cgbn_set_ui32(_arith._env, gas_cost, 5);
                    stack.mul(error_code);
                }
                break;
                case OP_SUB: // SUB
                {
                    cgbn_set_ui32(_arith._env, gas_cost, 3);
                    stack.sub(error_code);
                }
                break;
                case OP_DIV: // DIV
                {
                    cgbn_set_ui32(_arith._env, gas_cost, 5);
                    stack.div(error_code);
                }
                break;
                case OP_SDIV: // SDIV
                {
                    cgbn_set_ui32(_arith._env, gas_cost, 5);
                    stack.sdiv(error_code);
                }
                break;
                case OP_MOD: // MOD
                {
                    cgbn_set_ui32(_arith._env, gas_cost, 5);
                    stack.mod(error_code);
                }
                break;
                case OP_SMOD: // SMOD
                {
                    cgbn_set_ui32(_arith._env, gas_cost, 5);
                    stack.smod(error_code);
                }
                break;
                case OP_ADDMOD: // ADDMOD
                {
                    cgbn_set_ui32(_arith._env, gas_cost, 8);
                    stack.addmod(error_code);
                }
                break;
                case OP_MULMOD: // MULMOD
                {
                    cgbn_set_ui32(_arith._env, gas_cost, 8);
                    stack.mulmod(error_code);
                }
                break;
                case OP_EXP: // EXP
                {
                    cgbn_set_ui32(_arith._env, gas_cost, 10);
                    stack.exp(error_code, gas_cost);
                }
                break;
                case OP_SIGNEXTEND: // SIGNEXTEND
                {
                    cgbn_set_ui32(_arith._env, gas_cost, 5);
                    stack.signextend(error_code);
                }
                break;
                case OP_LT: // LT
                {
                    cgbn_set_ui32(_arith._env, gas_cost, 3);
                    stack.lt(error_code);
                }
                break;
                case OP_GT: // GT
                {
                    cgbn_set_ui32(_arith._env, gas_cost, 3);
                    stack.gt(error_code);
                }
                break;
                case OP_SLT: // SLT
                {
                    cgbn_set_ui32(_arith._env, gas_cost, 3);
                    stack.slt(error_code);
                }
                break;
                case OP_SGT: // SGT
                {
                    cgbn_set_ui32(_arith._env, gas_cost, 3);
                    stack.sgt(error_code);
                }
                break;
                case OP_EQ: // EQ
                {
                    cgbn_set_ui32(_arith._env, gas_cost, 3);
                    stack.eq(error_code);
                }
                break;
                case OP_ISZERO: // ISZERO
                {
                    cgbn_set_ui32(_arith._env, gas_cost, 3);
                    stack.iszero(error_code);
                }
                break;
                case OP_AND: // AND
                {
                    cgbn_set_ui32(_arith._env, gas_cost, 3);
                    stack.bitwise_and(error_code);
                }
                break;
                case OP_OR: // OR
                {
                    cgbn_set_ui32(_arith._env, gas_cost, 3);
                    stack.bitwise_or(error_code);
                }
                break;
                case OP_XOR: // XOR
                {
                    cgbn_set_ui32(_arith._env, gas_cost, 3);
                    stack.bitwise_xor(error_code);
                }
                break;
                case OP_NOT: // NOT
                {
                    cgbn_set_ui32(_arith._env, gas_cost, 3);
                    stack.bitwise_not(error_code);
                }
                break;
                case OP_BYTE: // BYTE
                {
                    cgbn_set_ui32(_arith._env, gas_cost, 3);
                    stack.get_byte(error_code);
                }
                break;
                case OP_SHL: // SHL
                {
                    cgbn_set_ui32(_arith._env, gas_cost, 3);
                    stack.shl(error_code);
                }
                break;
                case OP_SHR: // SHR
                {
                    cgbn_set_ui32(_arith._env, gas_cost, 3);
                    stack.shr(error_code);
                }
                break;
                case OP_SAR: // SAR
                {
                    cgbn_set_ui32(_arith._env, gas_cost, 3);
                    stack.sar(error_code);
                }
                break;
                case OP_SHA3: // SHA3
                {
                    cgbn_set_ui32(_arith._env, gas_cost, 30);

                    // get the values from stack
                    stack.pop(offset, error_code);
                    stack.pop(length, error_code);
                    length_s = _arith.from_cgbn_to_size_t(length);

                    // dynamic cost on size
                    minimum_word_size = (length_s + 31) / 32;
                    _arith.from_size_t_to_cgbn(aux_gas_cost, minimum_word_size);
                    cgbn_mul_ui32(_arith._env, aux_gas_cost, aux_gas_cost, 6);
                    cgbn_add(_arith._env, gas_cost, gas_cost, aux_gas_cost);

                    // get data from memory and hash
                    byte_data = memory.get(
                        offset,
                        length,
                        gas_cost,
                        remaining_gas,
                        error_code);

                    if (error_code == ERR_NONE)
                    {
                        _keccak.sha3(byte_data, length_s, &(tmp_memory[0]), HASH_BYTES);
                        _arith.from_memory_to_cgbn(value, &(tmp_memory[0]));
                        stack.push(value, error_code);
                    }
                }
                break;
                case OP_ADDRESS: // ADDRESS
                {
                    cgbn_set_ui32(_arith._env, gas_cost, 2);
                    stack.push(contract_address, error_code);
                }
                break;
                case OP_BALANCE: // BALANCE
                {
                    cgbn_set_ui32(_arith._env, gas_cost, 0);
                    stack.pop(address, error_code);
                    write_state.get_account_balance(
                        address,
                        value,
                        _global_state,
                        access_state,
                        parents_state,
                        gas_cost);
                    stack.push(value, error_code);
                }
                break;
                case OP_ORIGIN: // ORIGIN
                {
                    cgbn_set_ui32(_arith._env, gas_cost, 2);
                    msg.get_tx_origin(address);
                    stack.push(address, error_code);
                }
                break;
                case OP_CALLER: // CALLER
                {
                    cgbn_set_ui32(_arith._env, gas_cost, 2);
                    msg.get_caller(address);
                    stack.push(address, error_code);
                }
                break;
                case OP_CALLVALUE: // CALLVALUE
                {
                    cgbn_set_ui32(_arith._env, gas_cost, 2);
                    msg.get_value(value);
                    stack.push(value, error_code);
                }
                break;
                case OP_CALLDATALOAD: // CALLDATALOAD
                {
                    cgbn_set_ui32(_arith._env, gas_cost, 3);
                    stack.pop(index, error_code);
                    index_s = _arith.from_cgbn_to_size_t(index);

                    byte_data = msg.get_data(index_s, 32, size_s);
                    byte_data = expand_memory(byte_data, size_s, 32);
                    // TODO: reverse the byte_data
                    _arith.from_memory_to_cgbn(value, byte_data);
                    ONE_THREAD_PER_INSTANCE(
                        if (byte_data != NULL) {
                            free(byte_data);
                        })
                    stack.push(value, error_code);
                }
                break;
                case OP_CALLDATASIZE: // CALLDATASIZE
                {
                    cgbn_set_ui32(_arith._env, gas_cost, 2);
                    _arith.from_size_t_to_cgbn(length, msg.get_data_size());
                    stack.push(length, error_code);
                }
                break;
                case OP_CALLDATACOPY: // CALLDATACOPY
                {
                    cgbn_set_ui32(_arith._env, gas_cost, 3);

                    // get the values from stack
                    stack.pop(offset, error_code);
                    stack.pop(index, error_code);
                    stack.pop(length, error_code);
                    index_s = _arith.from_cgbn_to_size_t(index);
                    length_s = _arith.from_cgbn_to_size_t(length);

                    // dynamic cost on size
                    minimum_word_size = (length_s + 31) / 32;
                    _arith.from_size_t_to_cgbn(aux_gas_cost, minimum_word_size);
                    cgbn_mul_ui32(_arith._env, aux_gas_cost, aux_gas_cost, 3);
                    cgbn_add(_arith._env, gas_cost, gas_cost, aux_gas_cost);

                    // get data from msg and set on memory
                    byte_data = msg.get_data(index_s, length_s, size_s);
                    byte_data = expand_memory(byte_data, size_s, length_s);
                    memory.set(
                        byte_data,
                        offset,
                        length,
                        gas_cost,
                        remaining_gas,
                        error_code);

                    ONE_THREAD_PER_INSTANCE(
                        if (byte_data != NULL) {
                            free(byte_data);
                        })
                }
                break;
                case OP_CODESIZE: // CODESIZE
                {
                    cgbn_set_ui32(_arith._env, gas_cost, 2);
                    _arith.from_size_t_to_cgbn(length, contract->code_size);
                    stack.push(length, error_code);
                }
                break;
                case OP_CODECOPY: // CODECOPY
                {
                    cgbn_set_ui32(_arith._env, gas_cost, 3);

                    // get the values from stack
                    stack.pop(offset, error_code);
                    stack.pop(index, error_code);
                    stack.pop(length, error_code);

                    memory.grow_cost(
                        offset,
                        length,
                        gas_cost,
                        remaining_gas,
                        error_code);

                    if (error_code == ERR_NONE)
                    {
                        // transform and verify for overflow later
                        dst_offset_s = _arith.from_cgbn_to_size_t(offset);
                        index_s = _arith.from_cgbn_to_size_t(index);
                        length_s = _arith.from_cgbn_to_size_t(length);

                        bn_t MAX_SIZE_T;
                        cgbn_set_ui32(_arith._env, MAX_SIZE_T, 1);
                        cgbn_shift_left(_arith._env, MAX_SIZE_T, MAX_SIZE_T, 64);
                        // veirfy if is a jumpoint dest
                        if (cgbn_compare(_arith._env, index, MAX_SIZE_T) >= 0)
                        {
                            byte_data = expand_memory(NULL, 0, length_s);
                        }
                        else if (index_s >= contract->code_size)
                        {
                            byte_data = expand_memory(NULL, 0, length_s);
                        }
                        else
                        {
                            byte_data = expand_memory(contract->bytecode + index_s, contract->code_size - index_s, length_s);
                        }

                        memory.set(
                            byte_data,
                            offset,
                            length,
                            dummy_gas,
                            remaining_gas,
                            error_code);
                        ONE_THREAD_PER_INSTANCE(
                            if (byte_data != NULL) {
                                free(byte_data);
                            })
                    }
                }
                break;
                case OP_GASPRICE: // GASPRICE
                {
                    cgbn_set_ui32(_arith._env, gas_cost, 2);
                    msg.get_tx_gasprice(value);
                    stack.push(value, error_code);
                }
                break;
                case OP_EXTCODESIZE: // EXTCODESIZE
                {
                    cgbn_set_ui32(_arith._env, gas_cost, 0);
                    stack.pop(address, error_code);
                    length_s = write_state.get_account_code_size(
                        address,
                        _global_state,
                        access_state,
                        parents_state,
                        gas_cost);
                    _arith.from_size_t_to_cgbn(length, length_s);
                    stack.push(length, error_code);
                }
                break;
                case OP_EXTCODECOPY: // EXTCODECOPY
                {
                    cgbn_set_ui32(_arith._env, gas_cost, 0);
                    stack.pop(address, error_code);
                    stack.pop(offset, error_code);
                    stack.pop(index, error_code);
                    stack.pop(length, error_code);
                    dst_offset_s = _arith.from_cgbn_to_size_t(offset);
                    index_s = _arith.from_cgbn_to_size_t(index);
                    length_s = _arith.from_cgbn_to_size_t(length);
                    tmp_contract = write_state.get_account(
                        address,
                        _global_state,
                        access_state,
                        parents_state,
                        gas_cost,
                        4 // code
                    );

                    memory.grow_cost(
                        offset,
                        length,
                        gas_cost,
                        remaining_gas,
                        error_code);

                    if (error_code == ERR_NONE)
                    {
                        if (index_s >= tmp_contract->code_size)
                        {
                            byte_data = expand_memory(NULL, 0, length_s);
                        }
                        else
                        {
                            byte_data = expand_memory(tmp_contract->bytecode + index_s, tmp_contract->code_size - index_s, length_s);
                        }
                        memory.set(
                            byte_data,
                            offset,
                            length,
                            gas_cost,
                            remaining_gas,
                            error_code);
                        ONE_THREAD_PER_INSTANCE(
                            if (byte_data != NULL) {
                                free(byte_data);
                            })
                    }
                }
                break;
                case OP_RETURNDATASIZE: // RETURNDATASIZE
                {
                    cgbn_set_ui32(_arith._env, gas_cost, 2);
                    _arith.from_size_t_to_cgbn(length, external_return_data.size());
                    stack.push(length, error_code);
                }
                break;
                case OP_RETURNDATACOPY: // RETURNDATACOPY
                {
                    cgbn_set_ui32(_arith._env, gas_cost, 3);
                    stack.pop(offset, error_code);
                    stack.pop(index, error_code);
                    stack.pop(length, error_code);
                    dst_offset_s = _arith.from_cgbn_to_size_t(offset);
                    index_s = _arith.from_cgbn_to_size_t(index);
                    length_s = _arith.from_cgbn_to_size_t(length);
                    minimum_word_size = (length_s + 31) / 32;
                    _arith.from_size_t_to_cgbn(aux_gas_cost, 3 * minimum_word_size);
                    cgbn_add(_arith._env, gas_cost, gas_cost, aux_gas_cost);
                    byte_data = external_return_data.get(index_s, length_s, error_code);
                    memory.set(
                        byte_data,
                        offset,
                        length,
                        gas_cost,
                        remaining_gas,
                        error_code);
                }
                break;
                case OP_BLOCKHASH: // BLOCKHASH
                {
                    cgbn_set_ui32(_arith._env, gas_cost, 20);
                    stack.pop(index, error_code);
                    _current_block.get_previous_hash(index, value, error_code);
                    stack.push(value, error_code);
                }
                break;
                case OP_COINBASE: // COINBASE
                {
                    cgbn_set_ui32(_arith._env, gas_cost, 2);
                    _current_block.get_coin_base(value);
                    stack.push(value, error_code);
                }
                break;
                case OP_TIMESTAMP: // TIMESTAMP
                {
                    cgbn_set_ui32(_arith._env, gas_cost, 2);
                    _current_block.get_time_stamp(value);
                    stack.push(value, error_code);
                }
                break;
                case OP_NUMBER: // NUMBER
                {
                    cgbn_set_ui32(_arith._env, gas_cost, 2);
                    _current_block.get_number(value);
                    stack.push(value, error_code);
                }
                break;
                case OP_DIFFICULTY: // DIFFICULTY
                {
                    cgbn_set_ui32(_arith._env, gas_cost, 2);
                    _current_block.get_difficulty(value);
                    stack.push(value, error_code);
                }
                break;
                case OP_GASLIMIT: // GASLIMIT
                {
                    cgbn_set_ui32(_arith._env, gas_cost, 2);
                    _current_block.get_gas_limit(value);
                    stack.push(value, error_code);
                }
                break;
                case OP_CHAINID: // POP
                {
                    cgbn_set_ui32(_arith._env, gas_cost, 2);
                    _current_block.get_chain_id(value);
                    stack.push(value, error_code);
                }
                break;
                case OP_SELFBALANCE: // SELFBALANCE
                {
                    cgbn_set_ui32(_arith._env, gas_cost, 2);
                    stack.push(contract_balance, error_code);
                }
                break;
                case OP_BASEFEE: // BASEFEE
                {
                    cgbn_set_ui32(_arith._env, gas_cost, 2);
                    _current_block.get_base_fee(value);
                    stack.push(value, error_code);
                }
                break;
                case OP_POP: // POP
                {
                    cgbn_set_ui32(_arith._env, gas_cost, 2);
                    stack.pop(value, error_code);
                }
                break;
                case OP_MLOAD: // MLOAD
                {
                    cgbn_set_ui32(_arith._env, gas_cost, 3);
                    stack.pop(offset, error_code);
                    cgbn_set_ui32(_arith._env, length, 32);
                    byte_data = memory.get(
                        offset,
                        length,
                        gas_cost,
                        remaining_gas,
                        error_code);
                    if (error_code == ERR_NONE)
                    {
                        _arith.from_memory_to_cgbn(value, byte_data);
                        stack.push(value, error_code);
                    }
                }
                break;
                case OP_MSTORE: // MSTORE
                {
                    cgbn_set_ui32(_arith._env, gas_cost, 3);
                    stack.pop(offset, error_code);
                    offset_s = _arith.from_cgbn_to_size_t(offset);
                    stack.pop(value, error_code);
                    _arith.from_cgbn_to_memory(&(tmp_memory[0]), value);
                    cgbn_set_ui32(_arith._env, length, 32);
                    memory.set(
                        &(tmp_memory[0]),
                        offset,
                        length,
                        gas_cost,
                        remaining_gas,
                        error_code);
                }
                break;
                case OP_MSTORE8: // MSTORE8
                {
                    cgbn_set_ui32(_arith._env, gas_cost, 3);
                    stack.pop(offset, error_code);
                    offset_s = _arith.from_cgbn_to_size_t(offset);
                    stack.pop(value, error_code);
                    _arith.from_cgbn_to_memory(&(tmp_memory[0]), value);
                    cgbn_set_ui32(_arith._env, length, 1);

                    memory.set(
                        &(tmp_memory[WORD_BYTES - 1]),
                        offset,
                        length,
                        gas_cost,
                        remaining_gas,
                        error_code);
                }
                break;
                case OP_SLOAD: // SLOAD
                {
                    cgbn_set_ui32(_arith._env, gas_cost, 0);
                    stack.pop(key, error_code);
                    write_state.get_value(
                        storage_address,
                        key,
                        value,
                        _global_state,
                        access_state,
                        parents_state,
                        gas_cost);
                    stack.push(value, error_code);
                }
                break;
                case OP_SSTORE: // SSTORE
                {
                    cgbn_set_ui32(_arith._env, gas_cost, 0);
                    cgbn_set_ui32(_arith._env, gas_refund, 0);
                    stack.pop(key, error_code);
                    stack.pop(value, error_code);
                    write_state.set_value(
                        storage_address,
                        key,
                        value,
                        _global_state,
                        access_state,
                        parents_state,
                        gas_cost,
                        gas_refund);
                }
                break;
                // TODO: for jump verify is PUSHX the jump destination
                // needs testing to see if it is correct
                case OP_JUMP: // JUMP
                {
                    cgbn_set_ui32(_arith._env, gas_cost, 8);
                    stack.pop(index, error_code);
                    int32_t overflow = _arith.size_t_from_cgbn(index_s, index);
                    // veirfy if is a jumpoint dest
                    if (error_code == ERR_NONE)
                    {
                        if (overflow)
                        {
                            error_code = ERR_INVALID_JUMP_DESTINATION;
                        }
                        else if (index_s >= contract->code_size)
                        {
                            error_code = ERR_INVALID_JUMP_DESTINATION;
                        }
                        else if (D_jumps.has(index_s) == 0)
                        {
                            error_code = ERR_INVALID_JUMP_DESTINATION;
                        }
                        else
                        {
                            pc = index_s - 1;
                        }
                    }
                }
                break;
                case OP_JUMPI: // JUMPI
                {
                    cgbn_set_ui32(_arith._env, gas_cost, 10);
                    stack.pop(index, error_code);
                    stack.pop(value, error_code);
                    int32_t overflow = _arith.size_t_from_cgbn(index_s, index);
                    if ((cgbn_compare_ui32(_arith._env, value, 0) != 0) &&
                        (error_code == ERR_NONE))
                    {
                        if (overflow)
                        {
                            error_code = ERR_INVALID_JUMP_DESTINATION;
                        }
                        else if (index_s >= contract->code_size)
                        {
                            error_code = ERR_INVALID_JUMP_DESTINATION;
                        }
                        else if (D_jumps.has(index_s) == 0)
                        {
                            error_code = ERR_INVALID_JUMP_DESTINATION;
                        }
                        else
                        {
                            pc = index_s - 1;
                        }
                    }
                }
                break;
                case OP_PC: // PC
                {
                    cgbn_set_ui32(_arith._env, gas_cost, 2);
                    cgbn_set_ui32(_arith._env, value, pc);
                    stack.push(value, error_code);
                }
                break;
                case OP_MSIZE: // MSIZE
                {
                    cgbn_set_ui32(_arith._env, gas_cost, 2);
                    _arith.from_size_t_to_cgbn(length, memory.size());
                    stack.push(length, error_code);
                }
                break;
                case OP_GAS: // GAS
                {
                    // cgbn_add_ui32(_arith._env, gas_cost, gas_cost, 2);
                    cgbn_set_ui32(_arith._env, gas_cost, 2);
                    cgbn_set(_arith._env, value, remaining_gas);
                    cgbn_sub(_arith._env, value, value, gas_cost);
                    stack.push(value, error_code);
                }
                break;
                case OP_JUMPDEST: // JUMPDEST
                {
                    cgbn_set_ui32(_arith._env, gas_cost, 1);
                    // do nothing
                    pc = pc;
                }
                break;
                case OP_PUSH0: // PUSH0
                {
                    cgbn_set_ui32(_arith._env, gas_cost, 2);
                    cgbn_set_ui32(_arith._env, value, 0);
                    stack.push(value, error_code);
                }
                break;
                case OP_CREATE: // CREATE
                {
                    // TODO: implement
                    // make the rhl function
                    error_code = ERR_NOT_IMPLEMENTED;
                }
                break;
                case OP_CALL: // CALL
                {
                    call(
                        contract_address,
                        storage_address,
                        msg,
                        stack,
                        external_return_data,
                        memory,
                        access_state,
                        parents_state,
                        write_state,
                        remaining_gas,
#ifdef TRACER
                        tracer,
                        trace_pc,
                        opcode,
#endif
                        OP_CALL,
                        error_code);
                    msg.get_to(to);
                    contract = write_state.get_account(to, _global_state, access_state, parents_state, dummy_gas, 4);
                }
                break;
                case OP_CALLCODE: // CALLCODE
                {
                    call(
                        contract_address,
                        storage_address,
                        msg,
                        stack,
                        external_return_data,
                        memory,
                        access_state,
                        parents_state,
                        write_state,
                        remaining_gas,
#ifdef TRACER
                        tracer,
                        trace_pc,
                        opcode,
#endif
                        OP_CALLCODE,
                        error_code);
                    msg.get_to(to);
                    contract = write_state.get_account(to, _global_state, access_state, parents_state, dummy_gas, 4);
                }
                break;
                case OP_RETURN: // RETURN
                {
                    cgbn_set_ui32(_arith._env, gas_cost, 0);
                    stack.pop(offset, error_code);
                    stack.pop(length, error_code);
                    length_s = _arith.from_cgbn_to_size_t(length);
                    byte_data = memory.get(
                        offset,
                        length,
                        gas_cost,
                        remaining_gas,
                        error_code);
                    if (error_code == ERR_NONE)
                    {
                        returns.set(byte_data, length_s);
                        error_code = ERR_RETURN;
                    }
                }
                break;
                case OP_DELEGATECALL: // DELEGATECALL
                {

                    call(
                        contract_address,
                        storage_address,
                        msg,
                        stack,
                        external_return_data,
                        memory,
                        access_state,
                        parents_state,
                        write_state,
                        remaining_gas,
#ifdef TRACER
                        tracer,
                        trace_pc,
                        opcode,
#endif
                        OP_DELEGATECALL,
                        error_code);
                    msg.get_to(to);
                    contract = write_state.get_account(to, _global_state, access_state, parents_state, dummy_gas, 4);
                }
                break;
                case OP_CREATE2: // CREATE2
                {
                    // TODO: implement
                    error_code = ERR_NOT_IMPLEMENTED;
                }
                break;
                case OP_STATICCALL: // STATICCALL
                {
                    call(
                        contract_address,
                        storage_address,
                        msg,
                        stack,
                        external_return_data,
                        memory,
                        access_state,
                        parents_state,
                        write_state,
                        remaining_gas,
#ifdef TRACER
                        tracer,
                        trace_pc,
                        opcode,
#endif
                        OP_STATICCALL,
                        error_code);
                    msg.get_to(to);
                    contract = write_state.get_account(to, _global_state, access_state, parents_state, dummy_gas, 4);
                }
                break;
                case OP_REVERT: // REVERT
                {
                    cgbn_set_ui32(_arith._env, gas_cost, 0);
                    stack.pop(offset, error_code);
                    stack.pop(length, error_code);
                    length_s = _arith.from_cgbn_to_size_t(length);
                    byte_data = memory.get(
                        offset,
                        length,
                        gas_cost,
                        remaining_gas,
                        error_code);
                    returns.set(byte_data, length_s);
                    error_code = ERR_REVERT;
                }
                break;
                case OP_INVALID: // INVALID
                {
                    error_code = ERR_NOT_IMPLEMENTED;
                }
                break;
                case OP_SELFDESTRUCT: // SELFDESTRUCT
                {
                    // TODO: implement
                    error_code = ERR_NOT_IMPLEMENTED;
                }
                break;
                default:
                {
                    error_code = ERR_NOT_IMPLEMENTED;
                }
                break;
                }
            }
#ifdef TRACER
            if (opcode != OP_CALL && opcode != OP_CALLCODE && opcode != OP_DELEGATECALL && opcode != OP_STATICCALL)
                tracer.push(contract_address, trace_pc, opcode, &stack);
#endif
            if (cgbn_compare(_arith._env, remaining_gas, gas_cost) == -1)
            {
                error_code = ERR_OUT_OF_GAS;
            }
            if (error_code == ERR_OUT_OF_GAS)
            {
                cgbn_set(_arith._env, gas_cost, remaining_gas);
            }
            cgbn_sub(_arith._env, remaining_gas, remaining_gas, gas_cost);
            execution_step = execution_step + 1;

            error = error_code;

            if ((error == ERR_RETURN) || (error == ERR_REVERT))
            {
                break;
            }
            else if (error != ERR_NONE)
            {
                returns.set(NULL, 0);

                if (msg.get_depth() > 0)
                {
                    error = ERR_REVERT;
                }

                break;
            }
            else
            {
                pc = pc + 1;
            }
        }
        if (pc >= contract->code_size)
        {
            returns.set(NULL, 0);
#ifdef TRACER
            tracer.push(contract_address, pc, OP_STOP, &stack);
#endif
        }
        if (msg.get_depth() == 0)
        {
            stack.to_stack_data_t(call_stack, 0);
            memory.copy_info(call_memory);
        }
        else
        {

            // memory.free_memory();
            if ((memory._content->alocated_size > 0) && (memory._content->data != NULL))
            {
                ONE_THREAD_PER_INSTANCE(
                    free(memory._content->data);)
            }
        }
        cgbn_store(_arith._env, call_gas_left, remaining_gas);
    }

    __host__ __device__ void call(
        bn_t &contract_address,
        bn_t &storage_address,
        message_t &msg,
        stack_t &stack,
        return_data_t &returns,
        memory_t &memory,
        state_t &access_state,
        state_t &parents_state,
        state_t &write_state,
        bn_t &remaining_gas,
#ifdef TRACER
        tracer_t &tracer,
        uint32_t trace_pc,
        uint8_t opcode,
#endif
        uint32_t call_type,
        uint32_t &error_code)
    {
        SHARED_MEMORY state_data_t *external_call_parents_write_state;
        SHARED_MEMORY state_data_t *external_call_write_state;
        SHARED_MEMORY message_content_t *external_call_msg;

        ONE_THREAD_PER_INSTANCE(
            external_call_msg = (message_content_t *)malloc(sizeof(message_content_t));
            external_call_parents_write_state = (state_data_t *)malloc(sizeof(state_data_t));
            external_call_write_state = (state_data_t *)malloc(sizeof(state_data_t));)
        // get the values from the stack
        bn_t gas;
        bn_t to;
        bn_t value;
        bn_t caller;
        bn_t nonce;
        bn_t tx_origin;
        bn_t tx_gasprice;
        bn_t storage;
        bn_t dummy_gas_cost;
        bn_t gas_cost;
        bn_t capped_gas;
        bn_t return_value;
        // evm_word_t tmp_evm_word;;
        cgbn_set_ui32(_arith._env, gas_cost, 0);
        stack.pop(gas, error_code);

        stack.pop(to, error_code);
        // make the cost for accesing the state
        contract_t *contract = write_state.get_account(
            to,
            _global_state,
            access_state,
            parents_state,
            gas_cost,
            4 // code
        );

        if (call_type == OP_CALL)
        {
            stack.pop(value, error_code);
            cgbn_set(_arith._env, caller, contract_address);
            cgbn_set_ui32(_arith._env, storage, 0);
            // if th empty account is called
            bn_t balance;
            contract = write_state.get_account(
                to,
                _global_state,
                access_state,
                parents_state,
                dummy_gas_cost,
                3 // balance nonce
            );
            cgbn_load(_arith._env, balance, &(contract->balance));
            cgbn_load(_arith._env, nonce, &(contract->nonce));
            if ((cgbn_compare_ui32(_arith._env, balance, 0) == 0) &&
                (cgbn_compare_ui32(_arith._env, nonce, 0) == 0) &&
                (contract->code_size == 0))
            {
                cgbn_add_ui32(_arith._env, gas_cost, gas_cost, 25000);
            }
        }
        else if (call_type == OP_CALLCODE)
        {
            stack.pop(value, error_code);
            cgbn_set(_arith._env, caller, contract_address);
            cgbn_set(_arith._env, storage, storage_address);
        }
        else if (call_type == OP_DELEGATECALL)
        {
            msg.get_value(value);
            msg.get_caller(caller);
            cgbn_set(_arith._env, storage, storage_address);
        }
        else if (call_type == OP_STATICCALL)
        {
            cgbn_set_ui32(_arith._env, value, 0);
            cgbn_set(_arith._env, caller, contract_address);
            cgbn_set_ui32(_arith._env, storage, 0);
        }
        else
        {
            error_code = ERR_NOT_IMPLEMENTED;
            return;
        }
        // positive value cost
        if ((call_type == OP_CALL) || (call_type == OP_CALLCODE))
        {
            if (cgbn_compare_ui32(_arith._env, value, 0) == 1)
            {
                if (msg.get_call_type() == OP_STATICCALL)
                {
                    error_code = ERR_STATIC_CALL_CONTEXT;
                }
                cgbn_add_ui32(_arith._env, gas_cost, gas_cost, 9000);
                // TODO: something with fallback function refund 2300
            }
        }

        write_state.get_account_nonce(
            caller,
            nonce,
            _global_state,
            access_state,
            parents_state,
            dummy_gas_cost);

        msg.get_tx_origin(tx_origin);
        msg.get_tx_gasprice(tx_gasprice);
        // setup the message
        cgbn_store(_arith._env, &(external_call_msg->caller), caller);
        cgbn_store(_arith._env, &(external_call_msg->value), value);
        cgbn_store(_arith._env, &(external_call_msg->to), to);
        cgbn_store(_arith._env, &(external_call_msg->nonce), nonce);
        cgbn_store(_arith._env, &(external_call_msg->tx.gasprice), tx_gasprice);
        cgbn_store(_arith._env, &(external_call_msg->tx.origin), tx_origin);
        external_call_msg->depth = msg.get_depth() + 1;
        external_call_msg->call_type = call_type;
        cgbn_store(_arith._env, &(external_call_msg->storage), storage);
        // msg call data
        bn_t offset, length;
        size_t offset_s, length_s;
        uint8_t *byte_data;

        stack.pop(offset, error_code);
        stack.pop(length, error_code);
        length_s = _arith.from_cgbn_to_size_t(length);
        byte_data = memory.get(
            offset,
            length,
            gas_cost,
            remaining_gas,
            error_code);

        byte_data = expand_memory(byte_data, length_s, length_s);
        external_call_msg->data.size = length_s;
        external_call_msg->data.data = byte_data;
        // return data offset
        stack.pop(offset, error_code);
        stack.pop(length, error_code);
        offset_s = _arith.from_cgbn_to_size_t(offset);
        length_s = _arith.from_cgbn_to_size_t(length);
        // make the union of parents writes
        // and the new child write state
        external_call_parents_write_state->contracts == NULL;
        external_call_parents_write_state->no_contracts = 0;
        external_call_write_state->contracts == NULL;
        external_call_write_state->no_contracts = 0;
        state_t external_parents_write_state(_arith, external_call_parents_write_state);
        external_parents_write_state.copy_from_state_t(parents_state);
        external_parents_write_state.copy_from_state_t(write_state);
        state_t external_write_state(_arith, external_call_write_state);

        memory.grow_cost(
            offset,
            length,
            gas_cost,
            remaining_gas,
            error_code);
        if (cgbn_compare(_arith._env, remaining_gas, gas_cost) == -1)
        {
            error_code = ERR_OUT_OF_GAS;
            return;
        }
        cgbn_sub(_arith._env, remaining_gas, remaining_gas, gas_cost);
        cgbn_div_ui32(_arith._env, capped_gas, remaining_gas, 64);
        cgbn_sub(_arith._env, capped_gas, remaining_gas, capped_gas);
        if (cgbn_compare(_arith._env, gas, capped_gas) == 1)
        {
            cgbn_set(_arith._env, gas, capped_gas);
        }
        cgbn_sub(_arith._env, remaining_gas, remaining_gas, gas);

        cgbn_store(_arith._env, &(external_call_msg->gas), gas);

        message_t external_msg(_arith, external_call_msg);
        // error_code
        uint32_t external_error_code;
        // gas left
        evm_word_t call_gas_left;
        // cgbn_set_ui32(_arith._env, gas, 0);
        cgbn_store(_arith._env, &call_gas_left, gas);

#ifdef TRACER
        tracer.push(contract_address, trace_pc, opcode, &stack);
#endif

        // if no code size special case
        if (contract->code_size > 0)
        {

            // make the call TODO: look on gas
            run(
                external_call_msg,
                NULL,
                returns._content,
                NULL,
                access_state._content,
                external_call_parents_write_state,
                external_call_write_state,
                &call_gas_left,
#ifdef TRACER
                tracer._content,
#endif
                external_error_code);
        }
        else
        {
            returns.set(NULL, 0);
            external_error_code = ERR_NONE;
        }
        // TODO: maybe here an erorr if size is less than return data size
        uint32_t tmp_error_code;
        byte_data = returns.get(0, returns.size(), tmp_error_code);
        byte_data = expand_memory(byte_data, returns.size(), length_s);
        uint8_t *tmp_memory = byte_data;
        memory.set(
            byte_data,
            offset,
            length,
            dummy_gas_cost,
            remaining_gas,
            error_code);

        cgbn_load(_arith._env, gas, &call_gas_left);
        cgbn_add(_arith._env, remaining_gas, remaining_gas, gas);
        if (cgbn_compare(_arith._env, remaining_gas, gas_cost) == -1)
        {
            external_error_code = ERR_OUT_OF_GAS;
        }

        if ((external_error_code == ERR_NONE) || (external_error_code == ERR_RETURN))
        {
            // save the state
            write_state.copy_from_state_t(external_write_state);
            cgbn_set_ui32(_arith._env, return_value, 1);
        }
        else
        {
            cgbn_set_ui32(_arith._env, return_value, 0);
        }
        stack.push(return_value, error_code);
#ifdef TRACER
        // modify the stack with curret stack
        tracer.modify_last_stack(&stack);
#endif

        ONE_THREAD_PER_INSTANCE(
            if (tmp_memory != NULL)
                free(tmp_memory);)
        external_msg.free_memory();
        external_parents_write_state.free_memory();
        external_write_state.free_memory();
        // state_t::free_instance(external_call_parents_write_state);
        // state_t::free_instance(external_call_write_state);
        //  TODO: free the other allocated memory
    }

    __host__ static void get_instances(
        evm_instances_t &instances,
        const cJSON *test)
    {
        instances.msgs = message_t::get_messages(test, instances.count);
        instances.stacks = stack_t::get_stacks(instances.count);
        instances.return_datas = return_data_t::get_returns(instances.count);
        instances.memories = memory_t::get_memories_info(instances.count);
        instances.access_states = state_t::get_local_states(instances.count);
        instances.parents_write_states = state_t::get_local_states(instances.count);
        instances.write_states = state_t::get_local_states(instances.count);
        // keccak parameters
        instances.sha3_parameters = keccak_t::get_cpu_instances(instances.count);
        instances.world_state = state_t::get_global_state(test);
        instances.block = block_t::get_instance(test);
        instances.gas_left_a = (evm_word_t *)malloc(sizeof(evm_word_t) * instances.count);
        // TODO: maybe it works with memset
        for (size_t idx = 0; idx < instances.count; idx++)
        {
            for (size_t jdx = 0; jdx < params::BITS / 32; jdx++)
            {
                instances.gas_left_a[idx]._limbs[jdx] = 0;
            }
        }
#ifdef TRACER
        instances.tracers = tracer_t::get_tracers(instances.count);
#endif
        instances.errors = (uint32_t *)malloc(sizeof(uint32_t) * instances.count);
        memset(instances.errors, ERR_NONE, sizeof(uint32_t) * instances.count);
    }

    __host__ static void get_gpu_instances(
        evm_instances_t &gpu_instances,
        evm_instances_t &cpu_instances)
    {
        gpu_instances.count = cpu_instances.count;
        // msg
        gpu_instances.msgs = message_t::get_gpu_messages(cpu_instances.msgs, cpu_instances.count);
        // stack
        gpu_instances.stacks = stack_t::get_gpu_stacks(cpu_instances.stacks, cpu_instances.count);
        // return data
        gpu_instances.return_datas = return_data_t::get_gpu_returns(cpu_instances.return_datas, cpu_instances.count);
        // memory
        gpu_instances.memories = memory_t::get_gpu_memories_info(cpu_instances.memories, cpu_instances.count);
        // state
        gpu_instances.access_states = state_t::get_gpu_local_states(cpu_instances.access_states, cpu_instances.count);
        gpu_instances.parents_write_states = state_t::get_gpu_local_states(cpu_instances.parents_write_states, cpu_instances.count);
        gpu_instances.write_states = state_t::get_gpu_local_states(cpu_instances.write_states, cpu_instances.count);
        // keccak parameters
        gpu_instances.sha3_parameters = keccak_t::get_gpu_instances(cpu_instances.sha3_parameters, cpu_instances.count);
        // block
        gpu_instances.block = block_t::from_cpu_to_gpu(cpu_instances.block);
        // block_t::free_instance(cpu_block);
        gpu_instances.world_state = state_t::from_cpu_to_gpu(cpu_instances.world_state);
        // state_t::free_instance(cpu_world_state);
        cudaMalloc((void **)&gpu_instances.gas_left_a, sizeof(evm_word_t) * cpu_instances.count);
        cudaMemcpy(gpu_instances.gas_left_a, cpu_instances.gas_left_a, sizeof(evm_word_t) * cpu_instances.count, cudaMemcpyHostToDevice);
#ifdef TRACER
        gpu_instances.tracers = tracer_t::get_gpu_tracers(cpu_instances.tracers, cpu_instances.count);
#endif
        cudaMalloc((void **)&gpu_instances.errors, sizeof(uint32_t) * cpu_instances.count);
        cudaMemcpy(gpu_instances.errors, cpu_instances.errors, sizeof(uint32_t) * cpu_instances.count, cudaMemcpyHostToDevice);
    }

    __host__ static void get_cpu_from_gpu_instances(
        evm_instances_t &cpu_instances,
        evm_instances_t &gpu_instances)
    {
        // msgs
        message_t::free_gpu_messages(gpu_instances.msgs, cpu_instances.count);
        // stacks
        stack_t::free_stacks(cpu_instances.stacks, cpu_instances.count);
        cpu_instances.stacks = stack_t::get_cpu_stacks_from_gpu(gpu_instances.stacks, cpu_instances.count);
        stack_t::free_gpu_stacks(gpu_instances.stacks, cpu_instances.count);
        // return datas
        return_data_t::free_host_returns(cpu_instances.return_datas, cpu_instances.count);
        cpu_instances.return_datas = return_data_t::get_cpu_returns_from_gpu(gpu_instances.return_datas, cpu_instances.count);
        // memories
        memory_t::free_memories_info(cpu_instances.memories, cpu_instances.count);
        cpu_instances.memories = memory_t::get_memories_from_gpu(gpu_instances.memories, cpu_instances.count);
        // states
        state_t::free_local_states(cpu_instances.access_states, cpu_instances.count);
        cpu_instances.access_states = state_t::get_local_states_from_gpu(gpu_instances.access_states, cpu_instances.count);
        state_t::free_local_states(cpu_instances.parents_write_states, cpu_instances.count);
        cpu_instances.parents_write_states = state_t::get_local_states_from_gpu(gpu_instances.parents_write_states, cpu_instances.count);
        state_t::free_local_states(cpu_instances.write_states, cpu_instances.count);
        cpu_instances.write_states = state_t::get_local_states_from_gpu(gpu_instances.write_states, cpu_instances.count);
        // keccak
        keccak_t::free_gpu_instances(gpu_instances.sha3_parameters, cpu_instances.count);
        // block
        block_t::free_gpu(gpu_instances.block);
        // world state
        state_t::free_gpu_memory(gpu_instances.world_state);
        cudaMemcpy(cpu_instances.gas_left_a, gpu_instances.gas_left_a, sizeof(evm_word_t) * cpu_instances.count, cudaMemcpyDeviceToHost);
        cudaFree(gpu_instances.gas_left_a);
#ifdef TRACER
        tracer_t::free_tracers(cpu_instances.tracers, cpu_instances.count);
        cpu_instances.tracers = tracer_t::get_cpu_tracers_from_gpu(gpu_instances.tracers, cpu_instances.count);
#endif
        cudaMemcpy(cpu_instances.errors, gpu_instances.errors, sizeof(uint32_t) * cpu_instances.count, cudaMemcpyDeviceToHost);
        cudaFree(gpu_instances.errors);
    }

    __host__ static void free_instances(
        evm_instances_t &cpu_instances)
    {
        message_t::free_messages(cpu_instances.msgs, cpu_instances.count);
        stack_t::free_stacks(cpu_instances.stacks, cpu_instances.count);
        return_data_t::free_host_returns(cpu_instances.return_datas, cpu_instances.count);
        memory_t::free_memory_data(cpu_instances.memories, cpu_instances.count);
        state_t::free_local_states(cpu_instances.access_states, cpu_instances.count);
        state_t::free_local_states(cpu_instances.parents_write_states, cpu_instances.count);
        state_t::free_local_states(cpu_instances.write_states, cpu_instances.count);
        keccak_t::free_cpu_instances(cpu_instances.sha3_parameters, cpu_instances.count);
        block_t::free_instance(cpu_instances.block);
        state_t::free_instance(cpu_instances.world_state);
        free(cpu_instances.gas_left_a);
#ifdef TRACER
        tracer_t::free_tracers(cpu_instances.tracers, cpu_instances.count);
#endif
        free(cpu_instances.errors);
    }

    __host__ __device__ void print_instances(
        evm_instances_t instances)
    {
        printf("Current block\n");
        _current_block.print();
        printf("World state\n");
        _global_state.print();
        printf("Instances\n");
        for (size_t idx = 0; idx < instances.count; idx++)
        {
            printf("Instance %lu\n", idx);
            message_t message(_arith, &(instances.msgs[idx]));
            message.print();
            stack_t stack(_arith, &(instances.stacks[idx]));
            stack.print();
            return_data_t returns(&(instances.return_datas[idx]));
            returns.print();
            memory_t memory(_arith, &(instances.memories[idx]));
            memory.print();
            state_t access_state(_arith, &(instances.access_states[idx]));
            access_state.print();
            state_t parents_state(_arith, &(instances.parents_write_states[idx]));
            parents_state.print();
            state_t write_state(_arith, &(instances.write_states[idx]));
            write_state.print();
            printf("Gas left: ");
            print_bn<params>(instances.gas_left_a[idx]);
            printf("\n");
#ifdef TRACER
            tracer_t tracer(_arith, &(instances.tracers[idx]));
            tracer.print();
#endif
            printf("Error: %u\n", instances.errors[idx]);
        }
    }

    __host__ cJSON *instances_to_json(
        evm_instances_t instances)
    {
        mpz_t mpz_gas_left;
        mpz_init(mpz_gas_left);
        char *hex_string_ptr = (char *)malloc(sizeof(char) * ((params::BITS / 32) * 8 + 3));
        cJSON *root = cJSON_CreateObject();
        cJSON_AddItemToObject(root, "pre", _global_state.to_json());
        cJSON_AddItemToObject(root, "env", _current_block.to_json());
        cJSON *instances_json = cJSON_CreateArray();
        cJSON_AddItemToObject(root, "post", instances_json);
        for (size_t idx = 0; idx < instances.count; idx++)
        {
            cJSON *instance_json = cJSON_CreateObject();
            cJSON_AddItemToArray(instances_json, instance_json);
            message_t message(_arith, &(instances.msgs[idx]));
            cJSON_AddItemToObject(instance_json, "msg", message.to_json());
            stack_t stack(_arith, &(instances.stacks[idx]));
            cJSON_AddItemToObject(instance_json, "stack", stack.to_json());
            return_data_t returns(&(instances.return_datas[idx]));
            cJSON_AddItemToObject(instance_json, "return", returns.to_json());
            memory_t memory(_arith, &(instances.memories[idx]));
            cJSON_AddItemToObject(instance_json, "memory", memory.to_json());
            state_t access_state(_arith, &(instances.access_states[idx]));
            cJSON_AddItemToObject(instance_json, "access_state", access_state.to_json());
            state_t parents_state(_arith, &(instances.parents_write_states[idx]));
            cJSON_AddItemToObject(instance_json, "parents_state", parents_state.to_json());
            state_t write_state(_arith, &(instances.write_states[idx]));
            cJSON_AddItemToObject(instance_json, "write_state", write_state.to_json());
            _arith.from_cgbn_memory_to_hex(instances.gas_left_a[idx], hex_string_ptr);
            cJSON_AddItemToObject(instance_json, "gas_left", cJSON_CreateString(hex_string_ptr));
#ifdef TRACER
            tracer_t tracer(_arith, &(instances.tracers[idx]));
            cJSON_AddItemToObject(instance_json, "traces", tracer.to_json());
#endif
            cJSON_AddItemToObject(instance_json, "error", cJSON_CreateNumber(instances.errors[idx]));
            cJSON_AddItemToObject(instance_json, "success", cJSON_CreateBool((instances.errors[idx] == ERR_NONE) || (instances.errors[idx] == ERR_RETURN) || (instances.errors[idx] == ERR_SUCCESS)));
        }
        free(hex_string_ptr);
        mpz_clear(mpz_gas_left);
        return root;
    }
};

template <class params>
__global__ void kernel_evm(cgbn_error_report_t *report, typename evm_t<params>::evm_instances_t *instances)
{
    uint32_t instance = (blockIdx.x * blockDim.x + threadIdx.x) / params::TPI;

    if (instance >= instances->count)
        return;

    typedef arith_env_t<params> arith_t;
    typedef typename arith_t::bn_t bn_t;
    typedef evm_t<params> evm_t;

    // setup evm
    evm_t evm(cgbn_report_monitor, report, instance, &(instances->sha3_parameters[instance]), instances->block, instances->world_state);

    // run the evm
    evm.run(
        &(instances->msgs[instance]),
        &(instances->stacks[instance]),
        &(instances->return_datas[instance]),
        &(instances->memories[instance]),
        &(instances->access_states[instance]),
        &(instances->parents_write_states[instance]),
        &(instances->write_states[instance]),
        &(instances->gas_left_a[instance]),
#ifdef TRACER
        &(instances->tracers[instance]),
#endif
        instances->errors[instance]);
}

#endif