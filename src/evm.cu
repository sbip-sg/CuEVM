#ifndef _GPU_EVM_H_
#define _GPU_EVM_H_

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda.h>
#include <cjson/cJSON.h>
#include "utils.h"
#include "arith.cuh"
#include "stack.cuh"
#include "message.cuh"
#include "memory.cuh"
#include "storage.cuh"
#include "returndata.cuh"
#include "block.cuh"
#include "opcodes.h"
#include "error_codes.h"

template<class params>
class evm_t {
    public:
        typedef typename gpu_block<params>::gpu_block                   block_t;
        typedef typename gpu_block_hash<params>::gpu_block_hash         block_hash_t;
        typedef typename gpu_memory_t::memory_data_t                    memory_data_t;
        typedef typename gpu_message<params>::gpu_message               message_t;
        typedef typename gpu_stack_t<params>::stack_data_t              stack_data_t;
        typedef arith_env_t<params>                                     arith_t;
        typedef typename arith_env_t<params>::bn_t                      bn_t;
        typedef typename state_t<params>::contract_t                    contract_t;
        typedef gpu_stack_t<params>                                     stack_t;
        typedef cgbn_mem_t<params::BITS>                                evm_word_t;
        typedef gpu_memory_t                                            memory_t;
        typedef gpu_return_data_t                                       return_data_t;
        typedef state_t<params>                                         state_t;
        typedef typename state_t<params>::state_data_t                  state_data_t;
        typedef typename state_t<params>::contract_storage_t            contract_storage_t;
        typedef struct
        {
            uint32_t    pc;
            uint32_t    stack_size;
            uint8_t     opcode;
        } evm_state_t;
        

        arith_t _arith;
        state_t _global_state;
        block_t *_block;
        block_hash_t *_block_hash;
        uint32_t _instance;

        //constructor
        __device__  __forceinline__ emv_t(cgbn_monitor_t monitor, cgbn_error_report_t *report, uint32_t instance, block_t *_block, block_hash_t *_block_hash, state_data_t *global_contracts, uint32_t instance) : _arith(monitor, report, instance), _block(_block), _block_hash(_block_hash), _global_state(_arith, global_contracts), _instance(instance) {}

        __device__ run(
            message_t       *msgs,
            unit32_t        *stacks_size,
            stack_data_t    *stacks,
            return_data_t   *returns,
            memory_data_t   *return_memory,
            state_data_t    *local_state,
            uint32_t        *errors,
            uint32_t        *gas_left,
            #ifdef DEBUG
            uint32_t        *execution_lengths,
            evm_state_t     *debug_states,
            #endif
            uint32_t        depth
        ) {
            if (depth > params::MAX_DEPTH) {
                errors[_instance]=ERR_MAX_DEPTH_EXCEEDED;
                return;
            }

            // stackd data and memory shared inside the same block of threads
            __shared__ evm_word_t       stack_data[params::STACK_SIZE];
            __shared__ memory_data_t    memory_data;
            memory_data._size=0;
            memory_data._data=NULL;
            // temporary data for memory
            __shared__ uint8_t          tmp[params::BITS/8];
            // program counter
            uint32_t                    pc;
            uint32_t                    execution_step;
            pc = 0;

            // get the current contract
            contract_t *contract;
            contract=msgs[_instance].contract;

            // init stack of the execution env
            stack_t stack(_arith, &(stack_data[0]), params::STACK_SIZE);
            // init the memory of the execution env
            memory_t memory(&memory_data);
            // init the local state
            state_t local(_arith, &(local_state[_instance]));
            // auxliary variables for use with instructions
            bn_t address, key, value, offset, length, index, src_offset, dst_offset;
            size_t dst_offset_s, sec_offset_s, length_s;

            uint8_t opcode;
            #ifdef DEBUG
            uint32_t execution_step=0;
            while(pc < contract->code_size && execution_step<params::MAX_EXECUTION) {
            #else
            while(pc < contract->code_size) {
            #endif
                opcode=contract->bytecode[pc];

                if (opcode&0xF0==0x60 || opcode&0xF0==0x70) {
                    // PUSH
                    uint8_t push_size=(opcode&0x0F)+1;
                    uint8_t *push_data=&(contract->bytecode[pc+1]);
                    pc=pc+push_size;
                    stack.pushx(push_data, push_size);
                } else if (opcode&0xF0==0x80) {
                    // DUP
                    uint8_t dup_index=(opcode&0x0F)+1;
                    stack.dupx(dup_index);
                } else if (opcode&0xF0==0x90) {
                    // SWAP
                    uint8_t swap_index=(opcode&0x0F)+1;
                    stack.swapx(swap_index);
                } else if (opcode&0xF0==0xA0) {
                    // LOG
                    uint8_t log_index=opcode&0x0F;
                    errors[_instance]=ERR_NOT_IMPLEMENTED;
                } else {
                     switch (opcode) {
                        case OP_STOP: // STOP
                            pc=pc;
                            break;
                        case OP_ADD: // ADD
                            {
                                stack.add();
                            }
                            break;
                        case OP_MUL: // MUL
                            {
                                stack.mul();
                            }
                            break;
                        case OP_SUB: // SUB
                            {
                                stack.sub();
                            }
                            break;
                        case OP_DIV: // DIV
                            {
                                stack.div();
                            }
                            break;
                        case OP_SDIV: // SDIV
                            {
                                stack.sdiv();
                            }
                            break;
                        case OP_MOD: // MOD
                            {
                                stack.mod();
                            }
                            break;
                        case OP_SMOD: // SMOD
                            {
                                stack.smod();
                            }
                            break;
                        case OP_ADDMOD: // ADDMOD
                            {
                                stack.addmod();
                            }
                            break;
                        case OP_MULMOD: // MULMOD
                            {
                                stack.mulmod();
                            }
                            break;
                        case OP_EXP: // EXP
                            {
                                stack.exp();
                            }
                            break;
                        case OP_SIGNEXTEND: // SIGNEXTEND
                            {
                                stack.signextend();
                            }
                            break;
                        case OP_LT: // LT
                            {
                                stack.lt();
                            }
                            break;
                        case OP_GT: // GT
                            {
                                stack.gt();
                            }
                            break;
                        case OP_SLT: // SLT
                            {
                                stack.slt();
                            }
                            break;
                        case OP_SGT: // SGT
                            {
                                stack.sgt();
                            }
                            break;
                        case OP_EQ: // EQ
                            {
                                stack.eq();
                            }
                            break;
                        case OP_ISZERO: // ISZERO
                            {
                                stack.iszero();
                            }
                            break;
                        case OP_AND: // AND
                            {
                                stack.bitwise_and();
                            }
                            break;
                        case OP_OR: // OR
                            {
                                stack.bitwise_or();
                            }
                            break;
                        case OP_XOR: // XOR
                            {
                                stack.bitwise_xor();
                            }
                            break;
                        case OP_NOT: // NOT
                            {
                                stack.bitwise_not();
                            }
                            break;
                        case OP_BYTE: // BYTE
                            {
                                stack.get_byte();
                            }
                            break;
                        case OP_SHL: // SHL
                            {
                                stack.shl();
                            }
                            break;
                        case OP_SHR: // SHR
                            {
                                stack.shr();
                            }
                            break;
                        case OP_SAR: // SAR
                            {
                                stack.sar();
                            }
                            break;
                        case OP_SHA3: // SHA3
                            {
                                errors[instance]=ERR_NOT_IMPLEMENTED;
                            }
                            break;
                        case OP_ADDRESS: // ADDRESS
                            {
                                cgbn_load(arith._env, address, &(contract->address));
                                stack.push(address);
                            }
                            break;
                        case OP_BALANCE: // BALANCE
                            {
                                errors[instance]=ERR_NOT_IMPLEMENTED;
                            }
                            break;
                        case OP_ORIGIN: // ORIGIN
                            {
                                cgbn_load(arith._env, address, &(msgs[instance].tx.origin));
                                stack.push(address);
                            }
                            break;
                        case OP_CALLER: // CALLER
                            {
                                cgbn_load(arith._env, address, &(msgs[instance].caller));
                                stack.push(address);
                            }
                            break;
                        case OP_CALLVALUE: // CALLVALUE
                            {
                                cgbn_load(arith._env, value, &(msgs[instance].value));
                                stack.push(value);
                            }
                            break;
                        case OP_CALLDATALOAD: // CALLDATALOAD
                            {
                                stack.pop(index);
                                arith.from_memory_to_cgbn(value, msgs[instance].data.data + cgbn_get_ui32(arith._env, index));
                                stack.push(value);
                            }
                            break;
                        case OP_CALLDATASIZE: // CALLDATASIZE
                            {
                                arith.from_size_t_to_cgbn(length, msgs[instance].data.size);
                                stack.push(length);
                            }
                            break;
                        case OP_CALLDATACOPY: // CALLDATACOPY
                            {
                                stack.pop(offset);
                                stack.pop(index);
                                stack.pop(length);
                                destOffset=arith.from_cgbn_to_size_t(offset);
                                srcOffset=arith.from_cgbn_to_size_t(index);
                                size_length=arith.from_cgbn_to_size_t(length);
                                memory.set(destOffset, size_length, msgs[instance].data.data + srcOffset);
                            }
                            break;
                        case OP_CODESIZE: // CODESIZE
                            {
                                arith.from_size_t_to_cgbn(length, contract->code_size);
                                stack.push(length);
                            }
                            break;
                        case OP_CODECOPY: // CODECOPY
                            {
                                stack.pop(offset);
                                stack.pop(index);
                                stack.pop(length);
                                destOffset=arith.from_cgbn_to_size_t(offset);
                                srcOffset=arith.from_cgbn_to_size_t(index);
                                size_length=arith.from_cgbn_to_size_t(length);
                                memory.set(destOffset, size_length, contract->bytecode + srcOffset);
                            }
                            break;
                        case OP_GASPRICE: // GASPRICE
                            {
                                cgbn_load(arith._env, value, &(msgs[instance].tx.gasprice));
                                stack.push(value);
                            }
                            break;
                        case OP_EXTCODESIZE: // EXTCODESIZE
                            {
                                errors[instance]=ERR_NOT_IMPLEMENTED;
                            }
                            break;
                        case OP_EXTCODECOPY: // EXTCODECOPY
                            {
                                errors[instance]=ERR_NOT_IMPLEMENTED;
                            }
                            break;
                        case OP_RETURNDATASIZE: // RETURNDATASIZE
                            {
                                errors[instance]=ERR_NOT_IMPLEMENTED;
                            }
                            break;
                        case OP_RETURNDATACOPY: // RETURNDATACOPY
                            {
                                errors[instance]=ERR_NOT_IMPLEMENTED;
                            }
                            break;
                        case OP_BLOCKHASH: // BLOCKHASH
                            {
                                cgbn_load(arith._env, value, &(gpu_current_block.number));
                                stack.push(value);
                                stack.sub();
                                stack.pop(index);
                                size_t block_index;
                                block_index=arith.from_cgbn_to_size_t(index)-1;
                                cgbn_load(arith._env, value, &(gpu_last_blocks_hash[block_index].hash));
                                stack.push(value);
                            }
                            break;
                        case OP_COINBASE: // COINBASE
                            {
                                cgbn_load(arith._env, value, &(gpu_current_block.coin_base));
                                stack.push(value);
                            }
                            break;
                        case OP_TIMESTAMP: // TIMESTAMP
                            {
                                cgbn_load(arith._env, value, &(gpu_current_block.time_stamp));
                                stack.push(value);
                            }
                            break;
                        case OP_NUMBER: // NUMBER
                            {
                                cgbn_load(arith._env, value, &(gpu_current_block.number));
                                stack.push(value);
                            }
                            break;
                        case OP_DIFFICULTY: // DIFFICULTY
                            {
                                cgbn_load(arith._env, value, &(gpu_current_block.difficulty));
                                stack.push(value);
                            }
                            break;
                        case OP_GASLIMIT: // GASLIMIT
                            {
                                cgbn_load(arith._env, value, &(gpu_current_block.gas_limit));
                                stack.push(value);
                            }
                            break;
                        case OP_CHAINID: // POP
                            {
                                cgbn_load(arith._env, value, &(gpu_current_block.chain_id));
                                stack.push(value);
                            }
                            break;
                        case OP_SELFBALANCE: // SELFBALANCE
                            {
                                cgbn_load(arith._env, value, &(contract->balance));
                                stack.push(value);
                            }
                            break;
                        case OP_BASEFEE: // BASEFEE
                            {
                                cgbn_load(arith._env, value, &(gpu_current_block.base_fee));
                                stack.push(value);
                            }
                            break;
                        case OP_POP: // POP
                            {
                                stack.pop(value);
                            }
                            break;
                        case OP_MLOAD: // MLOAD
                            {
                                stack.pop(offset);
                                arith.from_memory_to_cgbn(value, memory.get(arith.from_cgbn_to_size_t(offset), 32));
                                stack.push(value);
                            }
                            break;
                        case OP_MSTORE: // MSTORE
                            {
                                stack.pop(offset);
                                stack.pop(value);
                                arith.from_cgbn_to_memory(&(tmp[0]), value);
                                memory.set(arith.from_cgbn_to_size_t(offset), 32, &(tmp[0]));
                            }
                            break;
                        case OP_MSTORE8: // MSTORE8
                            {
                                stack.pop(offset);
                                stack.pop(value);
                                arith.from_cgbn_to_memory(&(tmp[0]), value);
                                memory.set(arith.from_cgbn_to_size_t(offset), 1, &(tmp[0]));
                            }
                            break;
                        case OP_SLOAD: // SLOAD
                            {
                                errors[instance]=ERR_NOT_IMPLEMENTED;
                            }
                            break;
                        case OP_SSTORE: // SSTORE
                            {
                                errors[instance]=ERR_NOT_IMPLEMENTED;
                            }
                            break;
                        case OP_JUMP: // JUMP
                            {
                                stack.pop(index);
                                pc=arith.from_cgbn_to_size_t(index)-1;
                            }
                            break;
                        case OP_JUMPI: // JUMPI
                            {
                                stack.pop(index);
                                stack.pop(value);
                                if (cgbn_compare_ui32(arith._env, value, 0)!=0) {
                                    pc=arith.from_cgbn_to_size_t(index)-1;
                                }
                            }
                            break;
                        case OP_PC: // PC
                            {
                                cgbn_set_ui32(arith._env, value, pc);
                                stack.push(value);
                            }
                            break;
                        case OP_MSIZE: // MSIZE
                            {
                                arith.from_size_t_to_cgbn(length,  memory.size());
                                stack.push(length);
                            }
                            break;
                        case OP_GAS: // GAS
                            {
                                cgbn_load(arith._env, value, &(msgs[instance].gas));
                                stack.push(value);
                            }
                            break;
                        case OP_JUMPDEST: // JUMPDEST
                            {
                                // do nothing
                                pc=pc;
                            }
                            break;
                        case OP_PUSH0: // PUSH0
                            {
                                cgbn_set_ui32(arith._env, value, 0);
                                stack.push(value);
                            }
                            break;
                        case OP_CREATE: // CREATE
                            {
                                errors[instance]=ERR_NOT_IMPLEMENTED;
                            }
                            break;
                        case OP_CALL: // CALL
                            {
                                errors[instance]=ERR_NOT_IMPLEMENTED;
                            }
                            break;
                        case OP_CALLCODE: // CALLCODE
                            {
                                errors[instance]=ERR_NOT_IMPLEMENTED;
                            }
                            break;
                        case OP_RETURN: // RETURN
                            {
                                stack.pop(offset);
                                stack.pop(length);
                                destOffset=arith.from_cgbn_to_size_t(offset);
                                size_length=arith.from_cgbn_to_size_t(length);
                                returns[instance].offset=destOffset;
                                returns[instance].length=size_length;
                            }
                            break;
                        case OP_DELEGATECALL: // DELEGATECALL
                            {
                                errors[instance]=ERR_NOT_IMPLEMENTED;
                            }
                            break;
                        case OP_CREATE2: // CREATE2
                            {
                                errors[instance]=ERR_NOT_IMPLEMENTED;
                            }
                            break;
                        case OP_STATICCALL: // STATICCALL
                            {
                                errors[instance]=ERR_NOT_IMPLEMENTED;
                            }
                            break;
                        case OP_REVERT: // REVERT
                            {
                                stack.pop(offset);
                                stack.pop(length);
                                destOffset=arith.from_cgbn_to_size_t(offset);
                                size_length=arith.from_cgbn_to_size_t(length);
                                returns[instance].offset=destOffset;
                                returns[instance].length=size_length;
                                errors[instance]=ERR_NOT_IMPLEMENTED;
                            }
                            break;
                        case OP_INVALID: // INVALID
                            {
                                errors[instance]=ERR_NOT_IMPLEMENTED;
                            }
                            break;
                        case OP_SELFDESTRUCT: // SELFDESTRUCT
                            {
                                errors[instance]=ERR_NOT_IMPLEMENTED;
                            }
                            break;
                        default:
                            {
                                errors[instance]=ERR_NOT_IMPLEMENTED;
                            }
                            break;
                    }
                }
                #ifdef DEBUG
                execution_state[execution_step].pc=pc;
                execution_state[execution_step].stack_size=stack.size();
                execution_state[execution_step].opcode=opcode;
                stack.copy_stack_data(&(stacks[instance * params::MAX_EXECUTION + execution_step].values[0]));
                execution_step=execution_step+1;
                #endif
                if (errors[instance]!=ERR_NONE)
                    break;
                pc=pc+1;
            }
        }


};


#endif