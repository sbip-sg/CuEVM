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
        // constants
        static const uint32_t                           MAX_DEPTH=1024;
        static const uint32_t                           WORD_BITS = params::BITS;
        static const uint32_t                           WORD_BYTES = params::BITS/8;
        static const uint32_t                           MAX_EXECUTION_STEPS = 10000;

        typedef struct {
            message_t       *msgs;
            stack_data_t    *stacks;
            return_data_t   *return_datas;
            memory_data_t   *local_memories;
            state_data_t    *read_states;
            state_data_t    *write_states;
            block_data_t    *block;
            state_data_t    *world_state;
            #ifdef GAS
            evm_word_t      *gas_left_a;
            #endif
            #ifdef TRACER
            tracer_content_t *tracers;
            #endif
            uint32_t        *errors;
            uint32_t        instance_count;
        } evm_instances_t;

        

        arith_t     _arith;
        state_t     _global_state;
        block_t     _current_block;
        uint32_t    _instance;

        //constructor
        __device__  __forceinline__ emv_t(
            cgbn_monitor_t          monitor,
            cgbn_error_report_t     *report,
            uint32_t                instance,
            block_data_t            *block, 
            state_data_t            *world_state
        ) : _arith(monitor, report, instance), _current_block(_arith, block), _global_state(_arith, world_state), _instance(instance) {
        }
        
        __host__ emv_t(
            cgbn_monitor_t  monitor,
            uint32_t        instance,
            block_data_t    *block, 
            state_data_t    *world_state
        ) : _arith(monitor, instance), _current_block(_arith, block), _global_state(_arith, world_state), _instance(instance) {
        }

        __host__ __device__ void run(
            message_t       *msg,
            stack_data_t    *stack,
            return_data_t   *return_data,
            memory_data_t   *local_memory,
            state_data_t    *caller_local_state,
            #ifdef GAS
            evm_word_t      *gas_left,
            #endif
            #ifdef TRACER
            tracer_content_t *tracer,
            #endif
            uint32_t        &error
        ) {
            if (msg->depth > MAX_DEPTH) {
                error=ERR_MAX_DEPTH_EXCEEDED;
                return_data->offset=0;
                return_data->size=0;
                // probabily revert the state
                return;
            }

            // stack initiliasation
            __shared__ stack_content_data_t       stack_content;
            stack_data_t                          stack_data;
            stack_data.stack_offset=0;
            stack_data.stack_base=&(stack_content.values[0]);
            stack_t stack(_arith, &stack_data);

            // (heap) memory initiliasation
            __shared__ memory_data_t              memory_data;
            __shared__ uint8_t                    tmp_memory[WORD_BYTES];
            memory_data.size=0;
            memory_data.alocated_size=0;
            memory_data.data=NULL;
            memory_t memory(_arith, &memory_data);

            // local state initiliasation
            __shared__ state_data_t               local_state;
            local_state.no_contracts=0;
            local_state.contracts=NULL;
            state_t local(_arith, &local_state);

            // msg initiliasation
            message_t msg(_arith, msg);

            // tracer initiliasation
            #ifdef TRACER
            tracer_t tracer(arith, tracer);
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
            bn_t address, key, value, offset, length, index, src_offset, dst_offset;
            size_t dst_offset_s, src_offset_s, length_s, index_s, offset_s;
            bn_t remaining_gas;
            bn_t gas_cost, aux_gas_cost;
            uint8_t *byte_data;
            uin32_t minimum_word_size;


            // TODO: get the current contract
            contract=msgs[_instance].contract;
            // verify for null contract

            // get the gas from message
            msg.get_gas_limit(remaining_gas);
            cgbn_set_ui32(_arith._env, gas_cost, 0);


            #ifndef GAS
            uint32_t execution_step=0;
            while(pc < contract->code_size && execution_step<MAX_EXECUTION_STEPS)
            #else
            while(pc < contract->code_size)
            #endif
            {
                opcode=contract->bytecode[pc];

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
                                return_data->offset=0;
                                return_data->size=0;
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
                                error_code=ERR_NOT_IMPLEMENTED;
                            }
                            break;
                        case OP_ADDRESS: // ADDRESS
                            {
                                cgbn_set_ui32(_arith._env, gas_cost, 2);
                                // TODO: without load
                                cgbn_load(arith._env, address, &(contract->address));
                                stack.push(address, error_code);
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
                                message.get_tx_origin(address);
                                stack.push(address, error_code);
                            }
                            break;
                        case OP_CALLER: // CALLER
                            {
                                cgbn_set_ui32(_arith._env, gas_cost, 2);
                                message.get_caller(address);
                                stack.push(address, error_code);
                            }
                            break;
                        case OP_CALLVALUE: // CALLVALUE
                            {
                                cgbn_set_ui32(_arith._env, gas_cost, 2);
                                message.get_value(address);
                                stack.push(value, error_code);
                            }
                            break;
                        case OP_CALLDATALOAD: // CALLDATALOAD
                            {
                                cgbn_set_ui32(_arith._env, gas_cost, 3);
                                stack.pop(index, error_code);
                                index_s=arith.from_cgbn_to_size_t(index);
                                byte_data=message.get_data(index, 32, error_code);
                                arith.from_memory_to_cgbn(value, byte_data);
                                free(byte_data);
                                stack.push(value, error_code);
                            }
                            break;
                        case OP_CALLDATASIZE: // CALLDATASIZE
                            {
                                cgbn_set_ui32(_arith._env, gas_cost, 2);
                                arith.from_size_t_to_cgbn(length, message.get_data_size());
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
                                dst_offset_s=arith.from_cgbn_to_size_t(offset);
                                index_s=arith.from_cgbn_to_size_t(index);
                                length_s=arith.from_cgbn_to_size_t(length);
                                
                                // dynamic cost on size
                                minimum_word_size; = (length_s + 31) / 32
                                airht.from_size_t_to_cgbn(aux_gas_cost, minimum_word_size);
                                cgbn_mul_ui32(_arith._env, aux_gas_cost, aux_gas_cost, 3);
                                cgbn_add(_arith._env, gas_cost, gas_cost, aux_gas_cost);

                                // get data from msg and set on memory
                                byte_data=message.get_data(index, length_s, error_code);
                                memory.set(byte_data, dst_offset_s, length_s, gas_cost, error_code);
                                free(byte_data);
                            }
                            break;
                        case OP_CODESIZE: // CODESIZE
                            {
                                cgbn_set_ui32(_arith._env, gas_cost, 2);
                                arith.from_size_t_to_cgbn(length, contract->code_size);
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
                                dst_offset_s=arith.from_cgbn_to_size_t(offset);
                                index_s=arith.from_cgbn_to_size_t(index);
                                length_s=arith.from_cgbn_to_size_t(length);
                                
                                // TODO: maybe test the code size and length for adding zeros instead
                                memory.set(contract->bytecode + index_s, dst_offset_s, length_s, gas_cost, error_code);
                            }
                            break;
                        case OP_GASPRICE: // GASPRICE
                            {
                                cgbn_set_ui32(_arith._env, gas_cost, 2);
                                message.get_tx_gasprice(value);
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
                                _current_block.get_previous_block_hash(index, value, error_code);
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
                                cgbn_load(arith._env, value, &(contract->balance));
                                stack.push(value, error_code);
                            }
                            break;
                        case OP_BASEFEE: // BASEFEE
                            {
                                cgbn_set_ui32(_arith._env, gas_cost, 2);
                                _current_bloc.get_base_fee(value);
                                stack.push(value, error_code);
                            }
                            break;
                        case OP_POP: // POP
                            {
                                cgbn_set_ui32(_arith._env, gas_cost, 2);
                                stack.pop(value);
                            }
                            break;
                        case OP_MLOAD: // MLOAD
                            {
                                cgbn_set_ui32(_arith._env, gas_cost, 3);
                                stack.pop(offset, error_code);
                                offset_s=arith.from_cgbn_to_size_t(offset);
                                arith.from_memory_to_cgbn(value, memory.get(offset_s, 32, gas_cost, error_code));
                                stack.push(value, error_code);
                            }
                            break;
                        case OP_MSTORE: // MSTORE
                            {
                                cgbn_set_ui32(_arith._env, gas_cost, 3);
                                stack.pop(offset, error_code);
                                offset_s=arith.from_cgbn_to_size_t(offset);
                                stack.pop(value, error_code);
                                arith.from_cgbn_to_memory(&(tmp[0]), value);
                                memory.set(offset_s, 32, &(tmp[0]));
                            }
                            break;
                        case OP_MSTORE8: // MSTORE8
                            {
                                cgbn_set_ui32(_arith._env, gas_cost, 3);
                                stack.pop(offset, error_code);
                                offset_s=arith.from_cgbn_to_size_t(offset);
                                stack.pop(value, error_code);
                                arith.from_cgbn_to_memory(&(tmp[0]), value);
                                memory.set(offset_s, 1, &(tmp[0]));
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
                                error_code=ERR_NOT_IMPLEMENTED;
                            }
                            break;
                        case OP_JUMP: // JUMP
                            {
                                cgbn_set_ui32(_arith._env, gas_cost, 8);
                                stack.pop(index, error_code);
                                index_s=arith.from_cgbn_to_size_t(index);
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
                                index_s=arith.from_cgbn_to_size_t(index);
                                stack.pop(value, error_code);
                                if (cgbn_compare_ui32(arith._env, value, 0)!=0) {
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
                                cgbn_set_ui32(arith._env, value, pc);
                                stack.push(value, error_code);
                            }
                            break;
                        case OP_MSIZE: // MSIZE
                            {
                                cgbn_set_ui32(_arith._env, gas_cost, 2);
                                arith.from_size_t_to_cgbn(length,  memory.size());
                                stack.push(length, error_code);
                            }
                            break;
                        case OP_GAS: // GAS
                            {
                                cgbn_set_ui32(_arith._env, gas_cost, 2);
                                // TODO: reduce or very the gas before
                                cgbn_set_ui32(arith._env, value, remaining_gas);
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
                                cgbn_set_ui32(arith._env, value, 0);
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
                                stack.pop(offset);
                                offset_s=arith.from_cgbn_to_size_t(offset);
                                stack.pop(length);
                                length_s=arith.from_cgbn_to_size_t(length);
                                return_data->offset=offset_s;
                                return_data->size=length_s;
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
                                stack.pop(offset);
                                offset_s=arith.from_cgbn_to_size_t(offset);
                                stack.pop(length);
                                length_s=arith.from_cgbn_to_size_t(length);
                                return_data->offset=offset_s;
                                return_data->size=length_s;
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
                tracer.push(address, pc, opcode, &stack);
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
            stack.copy_stack_data(stack_data, 0);
            memory.copy_info(memory_data);
            // TODO: if not revert, update the state
            local.copy_state(caller_local_state);
            #ifdef GAS
            cgbn_store(_arith._env, gas_left, remaining_gas);
            #endif
        }

        __host__ static void get_instances(
            message_t       *&msgs,
            stack_data_t    *&stacks,
            return_data_t   *&return_datas,
            memory_data_t   *&local_memories,
            state_data_t    *&caller_local_states,
            block_data_t    *&block, 
            state_data_t    *&world_state,
            #ifdef GAS
            evm_word_t      *&gas_left_a,
            #endif
            #ifdef TRACER
            tracer_content_t *&tracers,
            #endif
            uint32_t        *&errors,
            cJSON           *test,
            uint32_t        &instance_count
        ) {
            msgs=message_t::get_messages(test, instances_count);
            stacks=stack_t::get_stacks(instance_count);
            return_datas=generate_host_returns(instance_count);
            local_memories=memory_t::get_memories_info(instance_count);
            caller_local_states=state_t::get_local_states(instance_count);
            world_state=state_t::get_global_state(test)
            block=block_t::get_instance(test)
            #ifdef GAS
            gas_left_a= (emv_word_t *) malloc(sizeof(emv_word_t) * instance_count);
            // TODO: maybe it works with memset
            for(size_t idx=0; idx<instance_count; idx++) {
                for(size_t jdx=0; jdx<params::BITS/32; jdx++) {
                    gas_left_a[idx]._limbs[jdx]=0;
                }
            }
            #endif
            #ifdef TRACER
            tracers=tracer_t::get_tracers(instance_count);
            #endif
            errors=(uint32_t *) malloc(sizeof(uint32_t) * instance_count);
            memset(errors, ERR_NONE, sizeof(uint32_t) * instance_count);
        }

        __host__ static void get_gpu_instances(
            message_t       *&msgs,
            stack_data_t    *&stacks,
            return_data_t   *&return_datas,
            memory_data_t   *&local_memories,
            state_data_t    *&caller_local_states,
            block_data_t    *&block, 
            state_data_t    *&world_state,
            #ifdef GAS
            evm_word_t      *&gas_left_a,
            #endif
            #ifdef TRACER
            tracer_content_t *&tracers,
            #endif
            uint32_t        *&errors,
            cJSON           *test,
            uint32_t        &instance_count
        ) {
            message_t       *cpu_msgs,
            stack_data_t    *cpu_stacks,
            return_data_t   *cpu_return_datas,
            memory_data_t   *cpu_local_memories,
            state_data_t    *cpu_caller_local_states,
            block_data_t    *cpu_block, 
            state_data_t    *cpu_world_state
            #ifdef GAS
            evm_word_t      *cpu_gas_left_a,
            #endif
            #ifdef TRACER
            tracer_content_t *cpu_tracers,
            #endif
            uint32_t        *cpu_errors,
            get_instances(
                cpu_msgs,
                cpu_stacks,
                cpu_return_datas,
                cpu_local_memories,
                cpu_caller_local_states,
                cpu_block,
                cpu_world_state,
            #ifdef GAS
                cpu_gas_left_a,
            #endif
            #ifdef TRACER
                cpu_tracers,
            #endif
                cpu_errors,
                test,
                instance_count
            );

            msgs=message_t::get_gpu_messages(cpu_msgs, instance_count);
            // keep them for json
            //message_t::free_messages(cpu_msgs, instance_count);
            stacks=stack_t::get_gpu_stacks(cpu_stacks, instance_count);
            stack_t::free_stacks(cpu_stacks, instance_count);
            return_datas=generate_gpu_returns(cpu_return_datas, instance_count);
            free_host_returns(cpu_return_datas, instance_count);
            local_memories=memory_t::get_gpu_memories_info(cpu_local_memories, instance_count);
            memory_t::free_memories_info(cpu_memories, instance_count);
            caller_local_states=state_t::get_gpu_local_states(cpu_caller_local_states, instance_count);
            state_t::free_local_states(cpu_caller_local_states, instance_count);
            block=block_t::from_cpu_to_gpu(cpu_block);
            //block_t::free_instance(cpu_block);
            world_state=state_t::from_cpu_to_gpu(cpu_world_state);
            //state_t::free_instance(cpu_world_state);
            #ifdef GAS
            cudaMalloc((void **)&gas_left_a, sizeof(emv_word_t) * instance_count);
            cudaMemcpy(gas_left_a, cpu_gas_left_a, sizeof(emv_word_t) * instance_count, cudaMemcpyHostToDevice);
            free(cpu_gas_left_a);
            #endif
            #ifdef TRACER
            tracers=tracer_t::get_gpu_tracers(cpu_tracers, instance_count);
            tracer_t::free_tracers(cpu_tracers, instance_count);
            #endif
            cudaMalloc((void **)&errors, sizeof(uint32_t) * instance_count);
            cudaMemcpy(errors, cpu_errors, sizeof(uint32_t) * instance_count, cudaMemcpyHostToDevice);
            free(cpu_errors);
        }
        

        __host__ static void get_cpu_from_gpu_instances(
            message_t       *&msgs,
            message_t       *gpu_msgs,
            stack_data_t    *&stacks,
            stack_data_t    *gpu_stacks,
            return_data_t   *&return_datas,
            return_data_t   *gpu_return_datas,
            memory_data_t   *&local_memories,
            memory_data_t   *gpu_local_memories,
            state_data_t    *&caller_local_states,
            state_data_t    *gpu_caller_local_states,
            block_data_t    *&block, 
            block_data_t    *gpu_block, 
            state_data_t    *&world_state
            state_data_t    *gpu_world_state
            #ifdef GAS
            evm_word_t      *&gas_left_a,
            evm_word_t      *gpu_gas_left_a,
            #endif
            #ifdef TRACER
            tracer_content_t *&tracers,
            tracer_content_t *gpu_tracers,
            #endif
            uint32_t        *&errors,
            uint32_t        *gpu_errors,
            uint32_t        instance_count
        ) {
            message_t::free_gpu_messages(gpu_messages, messages_count);
            stacks=stack_t::get_cpu_stacks_from_gpu(gpu_stacks, instance_count);
            stack_t::free_gpu_stacks(gpu_stacks, instance_count);
            cudaMemcpy(return_datas, gpu_return_datas, sizeof(return_data_t) * instance_count, cudaMemcpyDeviceToHost);
            free_gpu_returns(gpu_return_datas, instance_count);
            local_memories=memory_t::get_memories_from_gpu(gpu_local_memories, instance_count);
            caller_local_states=state_t::get_local_states_from_gpu(gpu_caller_local_states, instance_count);
            block_t::free_gpu(gpu_block);
            state_t::free_gpu_memory(gpu_world_state);
            #ifdef GAS
            cudaMemcpy(gas_left_a, gpu_gas_left_a, sizeof(evm_word_t) * instance_count, cudaMemcpyDeviceToHost);
            cudaFree(gpu_gas_left_a);
            #endif
            #ifdef TRACER
            tracers=tracer_t::get_cpu_tracers_from_gpu(gpu_tracers, instance_count);
            #endif
            cudaMemcpy(errors, gpu_errors, sizeof(uint32_t) * instance_count, cudaMemcpyDeviceToHost);
            cudaFree(gpu_errors);
        }

        __host__ static void free_instances(
            message_t       *&msgs,
            stack_data_t    *&stacks,
            return_data_t   *&return_datas,
            memory_data_t   *&local_memories,
            state_data_t    *&caller_local_states,
            block_data_t    *&block, 
            state_data_t    *&world_state,
            #ifdef GAS
            evm_word_t      *&gas_left_a,
            #endif
            #ifdef TRACER
            tracer_content_t *&tracers,
            #endif
            uint32_t        *&errors,
            uint32_t        &instance_count
        ) {
            message_t::free_messages(msgs, instance_count);
            stack_t::free_stacks(stacks, instance_count);
            free_host_returns(return_datas, instance_count);
            memory_t::free_memory_data(local_memories, instance_count);
            state_t::free_local_states(caller_local_states, instance_count);
            block_t::free_instance(block);
            state_t::free_instance(world_state);
            #ifdef GAS
            free(gas_left_a);
            #endif
            #ifdef TRACER
            tracer_t::free_tracers(tracers, instance_count);
            #endif
            free(errors);
        }

        __host__ __device__ void print_instances(
            message_t       *msgs,
            stack_data_t    *stacks,
            return_data_t   *return_datas,
            memory_data_t   *local_memories,
            state_data_t    *caller_local_states,
            #ifdef GAS
            evm_word_t      *gas_left_a,
            #endif
            #ifdef TRACER
            tracer_content_t *tracers,
            #endif
            uint32_t        *errors,
            uint32_t        instance_count
        ) {
            printf("Current block\n");
            _current_block.print();
            printf("World state\n");
            _global_state.print();
            printf("Instances\n");
            for(size_t idx=0; idx<instance_count; idx++) {
                printf("Instance %lu\n", idx);
                message_t message(_arith, &(msgs[idx]));
                message.print();
                stack_t stack(_arith, &(stacks[idx]));
                stack.print();
                print("Return data offset: %lu, size: %lu\n", return_datas[idx].offset, return_datas[idx].size);
                memory_t memory(_arith, &(local_memories[idx]));
                memory.print();
                state_t local(_arith, &(caller_local_states[idx]));
                local.print();
                #ifdef GAS
                print("Gas left: ");
                print_bn<params>(gas_left_a[idx]);
                print("\n");
                #endif
                #ifdef TRACER
                tracer_t tracer(_arith, &(tracers[idx]));
                tracer.print();
                #endif
                print("Error: %u\n", errors[idx]);
            }
        }

        __host__ cJSON *instances_to_json(
            message_t       *msgs,
            stack_data_t    *stacks,
            return_data_t   *return_datas,
            memory_data_t   *local_memories,
            state_data_t    *caller_local_states,
            #ifdef GAS
            evm_word_t      *gas_left_a,
            #endif
            #ifdef TRACER
            tracer_content_t *tracers,
            #endif
            uint32_t        *errors,
            uint32_t        instance_count
        ) {
            mpz_t mpz_gas_left;
            mpz_init(mpz_gas_left);
            char hex_string[67]="0x";
            cJSON *root = cJSON_CreateObject();
            cJSON_AddItemToObject(root, "pre", _global_state.to_json());
            cJSON_AddItemToObject(root, "env", _current_block.to_json());
            cJSON *instances = cJSON_CreateArray();
            cJSON_AddItemToObject(root, "post", instances);
            for(size_t idx=0; idx<instance_count; idx++) {
                cJSON *instance = cJSON_CreateObject();
                cJSON_AddItemToArray(instances, instance);
                message_t message(_arith, &(msgs[idx]));
                cJSON_AddItemToObject(instance, "msg", message.to_json());
                stack_t stack(_arith, &(stacks[idx]));
                cJSON_AddItemToObject(instance, "stack", stack.to_json());
                cJSON *return_json = cJSON_CreateObject();
                cJSON_AddItemToObject(instance, "return", return_json);
                cJSON_AddItemToObject(return_json, "offset", cJSON_CreateNumber(return_datas[idx].offset));
                cJSON_AddItemToObject(return_json, "size", cJSON_CreateNumber(return_datas[idx].size));
                memory_t memory(_arith, &(local_memories[idx]));
                cJSON_AddItemToObject(instance, "memory", memory.to_json());
                state_t local(_arith, &(caller_local_states[idx]));
                cJSON_AddItemToObject(instance, "state", local.to_json());
                #ifdef GAS
                to_mpz(mpz_gas_left, gas_left_a[idx]._limbs, params::BITS/32);
                strcpy(hex_string+2, mpz_get_str(NULL, 16, mpz_stack_value));
                cJSON_AddItemToObject(instance, "gas_left", cJSON_CreateString(hex_string));
                #endif
                #ifdef TRACER
                tracer_t tracer(_arith, &(tracers[idx]));
                cJSON_AddItemToObject(instance, "traces", tracer.to_json());
                #endif
                cJSON_AddItemToObject(instance, "error", cJSON_CreateNumber(errors[idx]));
                cJSON_AddItemToObject(instance, "success", cJSON_CreateBool(
                    (errors[idx]==ERR_NONE) ||
                    (errors[idx]==ERR_RETURN) ||
                    (errors[idx]==ERR_SUCCESS)
                ));
            }
            mpz_clear(mpz_gas_left);
            return root;
        }
};


#endif