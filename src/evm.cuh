#ifndef _EVM_H_
#define _EVM_H_

#include "utils.h"
#include "arith.cuh"
#include "stack.cuh"
#include "message.cuh"
#include "memory.cuh"
#include "returndata.cuh"
#include "block.cuh"
#include "tracer.cuh"
#include "contract.cuh"
#include "keccak.cuh"

template<class params>
class evm_t {
    public:
        //arithmetic environment
        typedef arith_env_t<params>                     arith_t;
        typedef typename arith_t::bn_t                  bn_t;
        typedef cgbn_mem_t<params::BITS>                evm_word_t;
        // block information
        typedef block_t<params>                         block_t;
        typedef block_t::block_data_t                   block_data_t;
        // state information
        typedef state_t<params>                         state_t;
        typedef state_t::state_data_t                   state_data_t;
        typedef state_t::contract_t                     contract_t;
        // stack information
        typedef stack_t<params>                         stack_t;
        typedef stack_t::stack_data_t                   stack_data_t;
        typedef stack_t::stack_content_data_t           stack_content_data_t;
        // memory information
        typedef memory_t<params>                        memory_t;
        typedef memory_t::memory_data_t                 memory_data_t;
        // message information
        typedef message_t<params>                       message_t;
        typedef message_t::message_content_t            message_content_t;
        // tracer information
        typedef tracer_t<params>                        tracer_t;
        typedef typename tracer_t::tracer_content_t     tracer_content_t;
        // keccak information
        typedef keccak::keccak_t               keccak_t;
        typedef typename keccak_t::sha3_parameters_t    sha3_parameters_t;

        // constants
        static const uint32_t                           MAX_DEPTH=1024;
        static const uint32_t                           WORD_BITS = params::BITS;
        static const uint32_t                           WORD_BYTES = params::BITS/8;
        static const uint32_t                           MAX_EXECUTION_STEPS = 10000;
        static const uint32_t                           HASH_BYTES = 32;

        typedef struct {
            message_content_t   *msgs;
            stack_data_t        *stacks;
            return_data_t       *return_datas;
            memory_data_t       *memories;
            state_data_t        *access_states;
            state_data_t        *parents_write_states;
            state_data_t        *write_states;
            sha3_parameters_t   *sha3_parameters;
            block_data_t        *block;
            state_data_t        *world_state;
            #ifdef GAS
            evm_word_t          *gas_left_a;
            #endif
            #ifdef TRACER
            tracer_content_t    *tracers;
            #endif
            uint32_t            *errors;
            size_t              count;
        } evm_instances_t;

        

        arith_t     _arith;
        state_t     _global_state;
        block_t     _current_block;
        uint32_t    _instance;
        keccak_t    _keccak;

        //constructor
        __device__  __forceinline__ evm_t(
            cgbn_monitor_t          monitor,
            cgbn_error_report_t     *report,
            uint32_t                instance,
            sha3_parameters_t       *sha3_parameters,
            block_data_t            *block, 
            state_data_t            *world_state
        ) : _arith(monitor, report, instance), _keccak(sha3_parameters->rndc, sha3_parameters->rotc, sha3_parameters->piln, sha3_parameters->state), _current_block(_arith, block), _global_state(_arith, world_state), _instance(instance) {
        }
        
        __host__ evm_t(
            cgbn_monitor_t      monitor,
            uint32_t            instance,
            sha3_parameters_t   *sha3_parameters,
            block_data_t        *block, 
            state_data_t        *world_state
        ) : _arith(monitor, instance), _keccak(sha3_parameters->rndc, sha3_parameters->rotc, sha3_parameters->piln, sha3_parameters->state), _current_block(_arith, block), _global_state(_arith, world_state), _instance(instance) {
        }

        __host__ __device__ void run(
            message_content_t   *call_msg,
            stack_data_t        *call_stack,
            return_data_t       *call_return_data,
            memory_data_t       *call_memory,
            state_data_t        *call_access_state,
            state_data_t        *call_parents_write_state,
            state_data_t        *call_write_state,
            #ifdef GAS
            evm_word_t          *call_gas_left,
            #endif
            #ifdef TRACER
            tracer_content_t    *call_tracer,
            #endif
            uint32_t            &error
        ) {
            if (call_msg->depth > MAX_DEPTH) {
                error=ERR_MAX_DEPTH_EXCEEDED;
                call_return_data->offset=0;
                call_return_data->size=0;
                // probabily revert the state
                return;
            }
            if (call_msg->call_type!=OP_CALL) {
                error=ERR_NOT_IMPLEMENTED;
                call_return_data->offset=0;
                call_return_data->size=0;
                // probabily revert the state
                return;
            }

            // stack initiliasation
            #ifdef __CUDA_ARCH__
            __shared__ stack_content_data_t       stack_content;
            #else
            stack_content_data_t                  stack_content;
            #endif
            stack_data_t                          stack_data;
            stack_data.stack_offset=0;
            stack_data.stack_base=&(stack_content.values[0]);
            stack_t stack(_arith, &stack_data);

            // (heap) memory initiliasation
            #ifdef __CUDA_ARCH__
            __shared__ memory_data_t              memory_data;
            __shared__ uint8_t                    tmp_memory[WORD_BYTES];
            #else
            memory_data_t                         memory_data;
            uint8_t                               tmp_memory[WORD_BYTES];
            #endif
            memory_data.size=0;
            memory_data.alocated_size=0;
            memory_data.data=NULL;
            memory_t memory(_arith, &memory_data);

            // local state initiliasation
            #ifdef __CUDA_ARCH__
            __shared__ state_data_t               local_state;
            #else
            state_data_t                          local_state;
            #endif
            local_state.no_contracts=0;
            local_state.contracts=NULL;
            state_t write_state(_arith, &local_state);
            state_t parents_state(_arith, call_parents_write_state);
            state_t access_state(_arith, call_access_state);

            // msg initiliasation
            message_t msg(_arith, call_msg);

            // tracer initiliasation
            #ifdef TRACER
            tracer_t tracer(_arith, call_tracer);
            #endif

            // evm run internal information
            uint32_t pc;
            uint8_t opcode;
            uint32_t error_code;
            error_code=ERR_SUCCESS;

            // auxiliary variables
            contract_t *contract;
            bn_t contract_address, contract_balance;
            bn_t caller, value, nonce, to, tx_origin, tx_gasprice;
            bn_t address, key, offset, length, index, src_offset, dst_offset;
            size_t dst_offset_s, src_offset_s, length_s, index_s, offset_s;
            bn_t remaining_gas;
            bn_t gas_cost, aux_gas_cost;
            uint8_t *byte_data;
            uint32_t minimum_word_size;


            // TODO: get the current contract
            // go thorugh all of them and don't pay
            msg.get_to(to);
            contract=_global_state.get_local_account(to, error_code);
            cgbn_load(_arith._env, contract_address, &(contract->address));
            cgbn_load(_arith._env, contract_balance, &(contract->balance));
            write_state.set_local_account(contract_address, contract, error_code);
            // verify for null contract

            // get the gas from message
            msg.get_gas(remaining_gas);
            cgbn_set_ui32(_arith._env, gas_cost, 0);


            pc=0;
            uint32_t trace_pc;
            #ifndef GAS
            uint32_t execution_step=0;
            while(pc < contract->code_size && execution_step<MAX_EXECUTION_STEPS)
            #else
            while(pc < contract->code_size)
            #endif
            {

                opcode=contract->bytecode[pc];
                #ifdef TRACER
                trace_pc=pc;
                #endif
                if ( ((opcode&0xF0)==0x60) || ((opcode&0xF0)==0x70) ) {
                    // PUSH
                    cgbn_set_ui32(_arith._env, gas_cost, 3);
                    uint8_t push_size=(opcode&0x0F)+1;
                    byte_data=&(contract->bytecode[pc+1]);
                    pc=pc+push_size;
                    stack.pushx(byte_data, push_size, error_code);
                } else if ( (opcode&0xF0)==0x80) {
                    // DUP
                    cgbn_set_ui32(_arith._env, gas_cost, 3);
                    uint8_t dup_index=(opcode&0x0F)+1;
                    stack.dupx(dup_index, error_code);
                } else if ( (opcode&0xF0)==0x90) {
                    // SWAP
                    cgbn_set_ui32(_arith._env, gas_cost, 3);
                    uint8_t swap_index=(opcode&0x0F)+1;
                    stack.swapx(swap_index, error_code);
                } else if ( (opcode&0xF0)==0xA0) {
                    // LOG
                    // more complex gas cost
                    uint8_t log_index=opcode&0x0F;
                    error_code=ERR_NOT_IMPLEMENTED;
                } else {
                     switch (opcode) {
                        case OP_STOP: // STOP
                            {
                                // only for contracts without code
                                call_return_data->offset=0;
                                call_return_data->size=0;
                                pc=pc;
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
                                src_offset_s=_arith.from_cgbn_to_size_t(offset);
                                length_s=_arith.from_cgbn_to_size_t(length);
                                
                                // dynamic cost on size
                                minimum_word_size = (length_s + 31) / 32;
                                _arith.from_size_t_to_cgbn(aux_gas_cost, minimum_word_size);
                                cgbn_mul_ui32(_arith._env, aux_gas_cost, aux_gas_cost, 6);
                                cgbn_add(_arith._env, gas_cost, gas_cost, aux_gas_cost);

                                // get data from memory and hash
                                byte_data=memory.get(src_offset_s, length_s, gas_cost, error_code);
                                _keccak.sha3(byte_data, length_s, &(tmp_memory[0]), HASH_BYTES);
                                _arith.from_memory_to_cgbn(value, &(tmp_memory[0]));
                                stack.push(value, error_code);
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
                                // TODO: get from lcoal/global state
                                error_code=ERR_NOT_IMPLEMENTED;
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
                                msg.get_value(address);
                                stack.push(value, error_code);
                            }
                            break;
                        case OP_CALLDATALOAD: // CALLDATALOAD
                            {
                                cgbn_set_ui32(_arith._env, gas_cost, 3);
                                stack.pop(index, error_code);
                                index_s=_arith.from_cgbn_to_size_t(index);
                                byte_data=msg.get_data(index_s, 32, error_code);
                                _arith.from_memory_to_cgbn(value, byte_data);
                                free(byte_data);
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
                                dst_offset_s=_arith.from_cgbn_to_size_t(offset);
                                index_s=_arith.from_cgbn_to_size_t(index);
                                length_s=_arith.from_cgbn_to_size_t(length);
                                
                                // dynamic cost on size
                                minimum_word_size = (length_s + 31) / 32;
                                _arith.from_size_t_to_cgbn(aux_gas_cost, minimum_word_size);
                                cgbn_mul_ui32(_arith._env, aux_gas_cost, aux_gas_cost, 3);
                                cgbn_add(_arith._env, gas_cost, gas_cost, aux_gas_cost);

                                // get data from msg and set on memory
                                byte_data=msg.get_data(index_s, length_s, error_code);
                                memory.set(byte_data, dst_offset_s, length_s, gas_cost, error_code);
                                free(byte_data);
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
                                dst_offset_s=_arith.from_cgbn_to_size_t(offset);
                                index_s=_arith.from_cgbn_to_size_t(index);
                                length_s=_arith.from_cgbn_to_size_t(length);
                                
                                // TODO: maybe test the code size and length for adding zeros instead
                                memory.set(contract->bytecode + index_s, dst_offset_s, length_s, gas_cost, error_code);
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
                                cgbn_set_ui32(_arith._env, gas_cost, 100);
                                // TODO: get from local/global state
                                error_code=ERR_NOT_IMPLEMENTED;
                            }
                            break;
                        case OP_EXTCODECOPY: // EXTCODECOPY
                            {
                                cgbn_set_ui32(_arith._env, gas_cost, 100);
                                // TODO: get from local/global state
                                error_code=ERR_NOT_IMPLEMENTED;
                            }
                            break;
                        case OP_RETURNDATASIZE: // RETURNDATASIZE
                            {
                                cgbn_set_ui32(_arith._env, gas_cost, 2);
                                // TODO: make a way of testing if the return data is set
                                error_code=ERR_NOT_IMPLEMENTED;
                            }
                            break;
                        case OP_RETURNDATACOPY: // RETURNDATACOPY
                            {
                                cgbn_set_ui32(_arith._env, gas_cost, 3);
                                // TODO: make a way of testing if the return data is set
                                error_code=ERR_NOT_IMPLEMENTED;
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
                                // TODO: 
                                cgbn_load(_arith._env, value, &(contract->balance));
                                stack.push(value, error_code);
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
                                offset_s=_arith.from_cgbn_to_size_t(offset);
                                _arith.from_memory_to_cgbn(value, memory.get(offset_s, 32, gas_cost, error_code));
                                stack.push(value, error_code);
                            }
                            break;
                        case OP_MSTORE: // MSTORE
                            {
                                cgbn_set_ui32(_arith._env, gas_cost, 3);
                                stack.pop(offset, error_code);
                                offset_s=_arith.from_cgbn_to_size_t(offset);
                                stack.pop(value, error_code);
                                _arith.from_cgbn_to_memory(&(tmp_memory[0]), value);
                                memory.set(&(tmp_memory[0]), offset_s, 32, gas_cost, error_code);
                            }
                            break;
                        case OP_MSTORE8: // MSTORE8
                            {
                                cgbn_set_ui32(_arith._env, gas_cost, 3);
                                stack.pop(offset, error_code);
                                offset_s=_arith.from_cgbn_to_size_t(offset);
                                stack.pop(value, error_code);
                                _arith.from_cgbn_to_memory(&(tmp_memory[0]), value);
                                memory.set(&(tmp_memory[0]), offset_s, 1, gas_cost, error_code);
                            }
                            break;
                        case OP_SLOAD: // SLOAD
                            {
                                cgbn_set_ui32(_arith._env, gas_cost, 0);
                                // TODO: use the states
                                error_code=ERR_NOT_IMPLEMENTED;
                            }
                            break;
                        case OP_SSTORE: // SSTORE
                            {
                                cgbn_set_ui32(_arith._env, gas_cost, 0);
                                // TODO: use the states
                                stack.pop(key, error_code);
                                stack.pop(value, error_code);
                                uint32_t tmp_error_code;
                                write_state.set_local_value(contract_address, key, value);
                                //error_code=ERR_NOT_IMPLEMENTED;
                            }
                            break;
                        case OP_JUMP: // JUMP
                            {
                                cgbn_set_ui32(_arith._env, gas_cost, 8);
                                stack.pop(index, error_code);
                                index_s=_arith.from_cgbn_to_size_t(index);
                                // veirfy if is a jumpoint dest
                                if (contract->bytecode[index_s]!=OP_JUMPDEST) {
                                    error_code=ERR_INVALID_JUMP_DESTINATION;
                                    break;
                                } else {
                                    pc=index_s-1;
                                }
                            }
                            break;
                        case OP_JUMPI: // JUMPI
                            {
                                cgbn_set_ui32(_arith._env, gas_cost, 8);
                                stack.pop(index, error_code);
                                index_s=_arith.from_cgbn_to_size_t(index);
                                stack.pop(value, error_code);
                                if (cgbn_compare_ui32(_arith._env, value, 0)!=0) {
                                    if (contract->bytecode[index_s]!=OP_JUMPDEST) {
                                        error_code=ERR_INVALID_JUMP_DESTINATION;
                                        break;
                                    } else {
                                        pc=index_s-1;
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
                                _arith.from_size_t_to_cgbn(length,  memory.size());
                                stack.push(length, error_code);
                            }
                            break;
                        case OP_GAS: // GAS
                            {
                                cgbn_set_ui32(_arith._env, gas_cost, 2);
                                // TODO: reduce or very the gas before
                                cgbn_set(_arith._env, value, remaining_gas);
                                cgbn_sub(_arith._env, value, value, gas_cost);
                                stack.push(value, error_code);
                                // stack.push(remaining_gas, error_code);
                            }
                            break;
                        case OP_JUMPDEST: // JUMPDEST
                            {
                                cgbn_set_ui32(_arith._env, gas_cost, 1);
                                // do nothing
                                pc=pc;
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
                                error_code=ERR_NOT_IMPLEMENTED;
                            }
                            break;
                        case OP_CALL: // CALL
                            {
                                // TODO: implement
                                error_code=ERR_NOT_IMPLEMENTED;
                            }
                            break;
                        case OP_CALLCODE: // CALLCODE
                            {
                                // TODO: implement
                                error_code=ERR_NOT_IMPLEMENTED;
                            }
                            break;
                        case OP_RETURN: // RETURN
                            {
                                cgbn_set_ui32(_arith._env, gas_cost, 0);
                                stack.pop(offset, error_code);
                                offset_s=_arith.from_cgbn_to_size_t(offset);
                                stack.pop(length, error_code);
                                length_s=_arith.from_cgbn_to_size_t(length);
                                call_return_data->offset=offset_s;
                                call_return_data->size=length_s;
                                error_code=ERR_RETURN;
                            }
                            break;
                        case OP_DELEGATECALL: // DELEGATECALL
                            {
                                // TODO: implement
                                error_code=ERR_NOT_IMPLEMENTED;
                            }
                            break;
                        case OP_CREATE2: // CREATE2
                            {
                                // TODO: implement
                                error_code=ERR_NOT_IMPLEMENTED;
                            }
                            break;
                        case OP_STATICCALL: // STATICCALL
                            {
                                // TODO: implement
                                error_code=ERR_NOT_IMPLEMENTED;
                            }
                            break;
                        case OP_REVERT: // REVERT
                            {
                                cgbn_set_ui32(_arith._env, gas_cost, 0);
                                stack.pop(offset, error_code);
                                offset_s=_arith.from_cgbn_to_size_t(offset);
                                stack.pop(length, error_code);
                                length_s=_arith.from_cgbn_to_size_t(length);
                                call_return_data->offset=offset_s;
                                call_return_data->size=length_s;
                                error_code=ERR_NOT_IMPLEMENTED;
                            }
                            break;
                        case OP_INVALID: // INVALID
                            {
                                error_code=ERR_NOT_IMPLEMENTED;
                            }
                            break;
                        case OP_SELFDESTRUCT: // SELFDESTRUCT
                            {
                                // TODO: implement
                                error_code=ERR_NOT_IMPLEMENTED;
                            }
                            break;
                        default:
                            {
                                error_code=ERR_NOT_IMPLEMENTED;
                            }
                            break;
                    }
                }
                #ifdef TRACER
                tracer.push(contract_address, trace_pc, opcode, &stack);
                #endif
                #ifdef GAS
                if (cgbn_compare(_arith._env, remaining_gas, gas_cost)==-1) {
                    error_code=ERR_OUT_OF_GAS;
                    break;
                }
                cgbn_sub(_arith._env, remaining_gas, remaining_gas, gas_cost);
                #else
                execution_step=execution_step+1;
                #endif
                error=error_code;
                if (error!=ERR_NONE)
                    break;
                pc=pc+1;
            }
            if (pc >= contract->code_size) {
                call_return_data->offset=0;
                call_return_data->size=0;
                #ifdef TRACER
                tracer.push(contract_address, pc, OP_STOP, &stack);
                #endif
            }
            stack.copy_stack_data(call_stack, 0);
            memory.copy_info(call_memory);
            // TODO: if not revert, update the state
            write_state.copy_to_state_data_t(call_write_state);
            #ifdef GAS
            cgbn_store(_arith._env, call_gas_left, remaining_gas);
            #endif
        }

        __host__ static void get_instances(
            evm_instances_t &instances,
            const cJSON           *test
        ) {
            instances.msgs=message_t::get_messages(test, instances.count);
            instances.stacks=stack_t::get_stacks(instances.count);
            instances.return_datas=generate_host_returns(instances.count);
            instances.memories=memory_t::get_memories_info(instances.count);
            instances.access_states=state_t::get_local_states(instances.count);
            instances.parents_write_states=state_t::get_local_states(instances.count);
            instances.write_states=state_t::get_local_states(instances.count);
            // keccak parameters
            instances.sha3_parameters=keccak_t::get_cpu_instances(instances.count);
            instances.world_state=state_t::get_global_state(test);
            instances.block=block_t::get_instance(test);
            #ifdef GAS
            instances.gas_left_a= (emv_word_t *) malloc(sizeof(emv_word_t) * instances.count);
            // TODO: maybe it works with memset
            for(size_t idx=0; idx<instances.count; idx++) {
                for(size_t jdx=0; jdx<params::BITS/32; jdx++) {
                    instances.gas_left_a[idx]._limbs[jdx]=0;
                }
            }
            #endif
            #ifdef TRACER
            instances.tracers=tracer_t::get_tracers(instances.count);
            #endif
            instances.errors=(uint32_t *) malloc(sizeof(uint32_t) * instances.count);
            memset(instances.errors, ERR_NONE, sizeof(uint32_t) * instances.count);
        }

        __host__ static void get_gpu_instances(
            evm_instances_t &gpu_instances,
            evm_instances_t &cpu_instances
        ) {
            gpu_instances.count=cpu_instances.count;
            // msg
            gpu_instances.msgs=message_t::get_gpu_messages(cpu_instances.msgs, cpu_instances.count);
            // stack
            gpu_instances.stacks=stack_t::get_gpu_stacks(cpu_instances.stacks, cpu_instances.count);
            // return data
            gpu_instances.return_datas=generate_gpu_returns(cpu_instances.return_datas, cpu_instances.count);
            // memory
            gpu_instances.memories=memory_t::get_gpu_memories_info(cpu_instances.memories, cpu_instances.count);
            // state
            gpu_instances.access_states=state_t::get_gpu_local_states(cpu_instances.access_states, cpu_instances.count);
            gpu_instances.parents_write_states=state_t::get_gpu_local_states(cpu_instances.parents_write_states, cpu_instances.count);
            gpu_instances.write_states=state_t::get_gpu_local_states(cpu_instances.write_states, cpu_instances.count);
            // keccak parameters
            gpu_instances.sha3_parameters=keccak_t::get_gpu_instances(cpu_instances.sha3_parameters, cpu_instances.count);
            // block
            gpu_instances.block=block_t::from_cpu_to_gpu(cpu_instances.block);
            //block_t::free_instance(cpu_block);
            gpu_instances.world_state=state_t::from_cpu_to_gpu(cpu_instances.world_state);
            //state_t::free_instance(cpu_world_state);
            #ifdef GAS
            cudaMalloc((void **)&gpu_instances.gas_left_a, sizeof(emv_word_t) * cpu_instances.count);
            cudaMemcpy(gpu_instances.gas_left_a, cpu_instances.cpu_gas_left_a, sizeof(emv_word_t) * cpu_instances.count, cudaMemcpyHostToDevice);
            #endif
            #ifdef TRACER
            gpu_instances.tracers=tracer_t::get_gpu_tracers(cpu_instances.tracers, cpu_instances.count);
            #endif
            cudaMalloc((void **)&gpu_instances.errors, sizeof(uint32_t) * cpu_instances.count);
            cudaMemcpy(gpu_instances.errors, cpu_instances.errors, sizeof(uint32_t) * cpu_instances.count, cudaMemcpyHostToDevice);
        }
        

        __host__ static void get_cpu_from_gpu_instances(
            evm_instances_t &cpu_instances,
            evm_instances_t &gpu_instances
        ) {
            // msgs
            message_t::free_gpu_messages(gpu_instances.msgs, cpu_instances.count);
            // stacks
            stack_t::free_stacks(cpu_instances.stacks, cpu_instances.count);
            cpu_instances.stacks=stack_t::get_cpu_stacks_from_gpu(gpu_instances.stacks, cpu_instances.count);
            stack_t::free_gpu_stacks(gpu_instances.stacks, cpu_instances.count);
            // return datas
            cudaMemcpy(cpu_instances.return_datas, gpu_instances.return_datas, sizeof(return_data_t) * cpu_instances.count, cudaMemcpyDeviceToHost);
            free_gpu_returns(gpu_instances.return_datas, cpu_instances.count);
            // memories
            memory_t::free_memories_info(cpu_instances.memories, cpu_instances.count);
            cpu_instances.memories=memory_t::get_memories_from_gpu(gpu_instances.memories, cpu_instances.count);
            // states
            state_t::free_local_states(cpu_instances.access_states, cpu_instances.count);
            cpu_instances.access_states=state_t::get_local_states_from_gpu(gpu_instances.access_states, cpu_instances.count);
            state_t::free_local_states(cpu_instances.parents_write_states, cpu_instances.count);
            cpu_instances.parents_write_states=state_t::get_local_states_from_gpu(gpu_instances.parents_write_states, cpu_instances.count);
            state_t::free_local_states(cpu_instances.write_states, cpu_instances.count);
            cpu_instances.write_states=state_t::get_local_states_from_gpu(gpu_instances.write_states, cpu_instances.count);
            // keccak
            keccak_t::free_gpu_instances(gpu_instances.sha3_parameters, cpu_instances.count);
            // block
            block_t::free_gpu(gpu_instances.block);
            // world state
            state_t::free_gpu_memory(gpu_instances.world_state);
            #ifdef GAS
            cudaMemcpy(cpu_instances.gas_left_a, gpu_instances.gas_left_a, sizeof(evm_word_t) * cpu_instances.count, cudaMemcpyDeviceToHost);
            cudaFree(gpu_instances.gas_left_a);
            #endif
            #ifdef TRACER
            tracer_t::free_tracers(cpu_instances.tracers, cpu_instances.count);
            cpu_instances.tracers=tracer_t::get_cpu_tracers_from_gpu(gpu_instances.tracers, cpu_instances.count);
            #endif
            cudaMemcpy(cpu_instances.errors, gpu_instances.errors, sizeof(uint32_t) * cpu_instances.count, cudaMemcpyDeviceToHost);
            cudaFree(gpu_instances.errors);
        }

        __host__ static void free_instances(
            evm_instances_t &cpu_instances
        ) {
            message_t::free_messages(cpu_instances.msgs, cpu_instances.count);
            stack_t::free_stacks(cpu_instances.stacks, cpu_instances.count);
            free_host_returns(cpu_instances.return_datas, cpu_instances.count);
            memory_t::free_memory_data(cpu_instances.memories, cpu_instances.count);
            state_t::free_local_states(cpu_instances.access_states, cpu_instances.count);
            state_t::free_local_states(cpu_instances.parents_write_states, cpu_instances.count);
            state_t::free_local_states(cpu_instances.write_states, cpu_instances.count);
            keccak_t::free_cpu_instances(cpu_instances.sha3_parameters, cpu_instances.count);
            block_t::free_instance(cpu_instances.block);
            state_t::free_instance(cpu_instances.world_state);
            #ifdef GAS
            free(cpu_instances.gas_left_a);
            #endif
            #ifdef TRACER
            tracer_t::free_tracers(cpu_instances.tracers, cpu_instances.count);
            #endif
            free(cpu_instances.errors);
        }

        __host__ __device__ void print_instances(
            evm_instances_t instances
        ) {
            printf("Current block\n");
            _current_block.print();
            printf("World state\n");
            _global_state.print();
            printf("Instances\n");
            for(size_t idx=0; idx<instances.count; idx++) {
                printf("Instance %lu\n", idx);
                message_t message(_arith, &(instances.msgs[idx]));
                message.print();
                stack_t stack(_arith, &(instances.stacks[idx]));
                stack.print();
                printf("Return data offset: %lu, size: %lu\n", instances.return_datas[idx].offset, instances.return_datas[idx].size);
                memory_t memory(_arith, &(instances.memories[idx]));
                memory.print();
                state_t access_state(_arith, &(instances.access_states[idx]));
                access_state.print();
                state_t parents_state(_arith, &(instances.parents_write_states[idx]));
                parents_state.print();
                state_t write_state(_arith, &(instances.write_states[idx]));
                write_state.print();
                #ifdef GAS
                printf("Gas left: ");
                print_bn<params>(instances.gas_left_a[idx]);
                printf("\n");
                #endif
                #ifdef TRACER
                tracer_t tracer(_arith, &(instances.tracers[idx]));
                tracer.print();
                #endif
                printf("Error: %u\n", instances.errors[idx]);
            }
        }

        __host__ cJSON *instances_to_json(
            evm_instances_t instances
        ) {
            mpz_t mpz_gas_left;
            mpz_init(mpz_gas_left);
            char hex_string[67]="0x";
            cJSON *root = cJSON_CreateObject();
            cJSON_AddItemToObject(root, "pre", _global_state.to_json());
            cJSON_AddItemToObject(root, "env", _current_block.to_json());
            cJSON *instances_json = cJSON_CreateArray();
            cJSON_AddItemToObject(root, "post", instances_json);
            for(size_t idx=0; idx<instances.count; idx++) {
                cJSON *instance_json = cJSON_CreateObject();
                cJSON_AddItemToArray(instances_json, instance_json);
                message_t message(_arith, &(instances.msgs[idx]));
                cJSON_AddItemToObject(instance_json, "msg", message.to_json());
                stack_t stack(_arith, &(instances.stacks[idx]));
                cJSON_AddItemToObject(instance_json, "stack", stack.to_json());
                cJSON *return_json = cJSON_CreateObject();
                cJSON_AddItemToObject(instance_json, "return", return_json);
                cJSON_AddItemToObject(return_json, "offset", cJSON_CreateNumber(instances.return_datas[idx].offset));
                cJSON_AddItemToObject(return_json, "size", cJSON_CreateNumber(instances.return_datas[idx].size));
                memory_t memory(_arith, &(instances.memories[idx]));
                cJSON_AddItemToObject(instance_json, "memory", memory.to_json());
                state_t access_state(_arith, &(instances.access_states[idx]));
                cJSON_AddItemToObject(instance_json, "access_state", access_state.to_json());
                state_t parents_state(_arith, &(instances.parents_write_states[idx]));
                cJSON_AddItemToObject(instance_json, "parents_state", parents_state.to_json());
                state_t write_state(_arith, &(instances.write_states[idx]));
                cJSON_AddItemToObject(instance_json, "write_state", write_state.to_json());
                #ifdef GAS
                to_mpz(mpz_gas_left, instances.gas_left_a[idx]._limbs, params::BITS/32);
                strcpy(hex_string+2, mpz_get_str(NULL, 16, mpz_stack_value));
                cJSON_AddItemToObject(instance_json, "gas_left", cJSON_CreateString(hex_string));
                #endif
                #ifdef TRACER
                tracer_t tracer(_arith, &(instances.tracers[idx]));
                cJSON_AddItemToObject(instance_json, "traces", tracer.to_json());
                #endif
                cJSON_AddItemToObject(instance_json, "error", cJSON_CreateNumber(instances.errors[idx]));
                cJSON_AddItemToObject(instance_json, "success", cJSON_CreateBool(
                    (instances.errors[idx]==ERR_NONE) ||
                    (instances.errors[idx]==ERR_RETURN) ||
                    (instances.errors[idx]==ERR_SUCCESS)
                ));
            }
            mpz_clear(mpz_gas_left);
            return root;
        }
};


template<class params>
__global__ void kernel_evm(cgbn_error_report_t *report, typename evm_t<params>::evm_instances_t *instances) {
  uint32_t instance=(blockIdx.x*blockDim.x + threadIdx.x)/params::TPI;

  if(instance >= instances->count)
    return;

  typedef arith_env_t<params> arith_t;
  typedef typename arith_t::bn_t  bn_t;
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
    #ifdef GAS
    &(instances->gas_left_a[instance]),
    #endif
    #ifdef TRACER
    &(instances->tracers[instance]),
    #endif
    instances->errors[instance]
  );
}

#endif