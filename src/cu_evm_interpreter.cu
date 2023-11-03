#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <cuda.h>
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


typedef typename gpu_block<utils_params>::gpu_block gpu_block_t;
typedef typename gpu_block_hash<utils_params>::gpu_block_hash gpu_block_hash_t;
typedef typename gpu_memory_t::memory_data_t memory_data_t;

// blocks and blocks hash are read only elements
__device__ __constant__ gpu_block_t gpu_current_block;
__device__ __constant__ gpu_block_hash_t gpu_last_blocks_hash[256];

typedef struct {
    uint32_t    pc;
    uint32_t    stack_size;
    uint8_t     opcode;
} interpreter_state_t;

#define MAX_EXECUTION_SIZE 1000

__device__ __global__ interpreter_state_t execution_state[MAX_EXECUTION_SIZE];

template<class params>
__global__ void cu_evm_interpreter_kernel(cgbn_error_report_t *report, typename gpu_message<params>::gpu_message *msgs, typename gpu_stack_t<params>::stack_data_t *stacks, memory_data_t *debug_memory, return_data_t *returns, uint32_t instance_count, unit32_t *execution_lengths, uint32_t *errors)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t instance=(blockIdx.x*blockDim.x + threadIdx.x)/params::TPI;
    __shared__ cgbn_mem_t<params::BITS> stack_data[params::STACK_SIZE];
    __shared__ memory_data_t memory_data;
    __shared__ uint8_t       tmp[params::BITS/8];
    uint32_t pc;
    uint32_t execution_step;
    memory_data._size=0;
    memory_data._data=NULL;

    if(instance>=instance_count)
        return;

    // initiliase the arithmetics on 256 bits
    typedef arith_env_t<params> local_arith_t;
    typedef typename arith_env_t<params>::bn_t  bn_t;
    local_arith_t arith(cgbn_report_monitor, report, instance);

    // get the contract
    typedef typename gpu_global_storage_t<params>::gpu_contract_t gpu_contract_t;
    gpu_contract_t *contract;
    contract=msgs[instance].contract;

    // initialiase the stack
    typedef gpu_stack_t<params> local_stack_t;
    local_stack_t  stack(arith, &(stack_data[0]), params::STACK_SIZE);

    // initialiase the memory
    gpu_memory_t  memory(&memory_data);

    pc = 0;
    execution_step;
    uint8_t opcode;
        /*execution_state[execution_step].pc=pc;
        execution_state[execution_step].stack_size=stack.size();
        execution_state[execution_step].opcode=contract->bytecode[pc];
        pc++;*/
    // auxiliary variables
    bn_t address, key, value, offset, length, index;
    size_t destOffset, srcOffset, size_length;
    for (execution_step=0; execution_step<MAX_EXECUTION_SIZE; execution_step++) {
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
            errors[instance]=ERR_NOT_IMPLEMENTED
        } else {
            switch (opcode) {
                case OP_STOP: // STOP
                    pc=pc;
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
                        stack.byte();
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
                        errors[instance]=ERR_NOT_IMPLEMENTED
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
                        errors[instance]=ERR_NOT_IMPLEMENTED
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
                        airth.from_memory_to_cgbn(value, msgs[instance].data + cgbn_get_uint32(arith._env, index));
                        stack.push(value);
                    }
                    break;
                case OP_CALLDATASIZE: // CALLDATASIZE
                    {
                        airth.from_size_t_to_cgbn(length, msgs[instance].data_size);
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
                        memory.set(destOffset, msgs[instance].data + srcOffset, size_length);
                    }
                    break;
                case OP_CODESIZE: // CODESIZE
                    {
                        airth.from_size_t_to_cgbn(length, contract->code_size);
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
                        memory.set(destOffset, contract->bytecode + srcOffset, size_length);
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
                        errors[instance]=ERR_NOT_IMPLEMENTED
                    }
                    break;
                case OP_EXTCODECOPY: // EXTCODECOPY
                    {
                        errors[instance]=ERR_NOT_IMPLEMENTED
                    }
                    break;
                case OP_RETURNDATASIZE: // RETURNDATASIZE
                    {
                        errors[instance]=ERR_NOT_IMPLEMENTED
                    }
                    break;
                case OP_RETURNDATACOPY: // RETURNDATACOPY
                    {
                        errors[instance]=ERR_NOT_IMPLEMENTED
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
                        memory.set(arith.from_cgbn_to_size_t(offset), arith.from_cgbn_to_uint8(value), 32);
                    }
                    break;
                case OP_MSTORE8: // MSTORE8
                    {
                        stack.pop(offset);
                        stack.pop(value);
                        arith.from_cgbn_to_memory(&(tmp[0]), value);
                        memory.set(arith.from_cgbn_to_size_t(offset), &(tmp[0]), 1);
                    }
                    break;
                case OP_SLOAD: // SLOAD
                    {
                        errors[instance]=ERR_NOT_IMPLEMENTED
                    }
                    break;
                case OP_SSTORE: // SSTORE
                    {
                        errors[instance]=ERR_NOT_IMPLEMENTED
                    }
                    break;
                case OP_JUMP: // JUMP
                    {
                        stack.pop(index);
                        pc=arith.from_cgbn_to_uint32(index)-1;
                    }
                    break;
                case OP_JUMPI: // JUMPI
                    {
                        stack.pop(index);
                        stack.pop(value);
                        if (cgbn_compare_ui32(arith._env, value, 0)!=0) {
                            pc=arith.from_cgbn_to_uint32(index)-1;
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
                        airth.from_size_t_to_cgbn(length,  memory.size());
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
                        errors[instance]=ERR_NOT_IMPLEMENTED
                    }
                    break;
                case OP_CALL: // CALL
                    {
                        errors[instance]=ERR_NOT_IMPLEMENTED
                    }
                    break;
                case OP_CALLCODE: // CALLCODE
                    {
                        errors[instance]=ERR_NOT_IMPLEMENTED
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
                        errors[instance]=ERR_NOT_IMPLEMENTED
                    }
                    break;
                case OP_CREATE2: // CREATE2
                    {
                        errors[instance]=ERR_NOT_IMPLEMENTED
                    }
                    break;
                case OP_STATICCALL: // STATICCALL
                    {
                        errors[instance]=ERR_NOT_IMPLEMENTED
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
                        errors[instance]=ERR_NOT_IMPLEMENTED
                    }
                    break;
                case OP_INVALID: // INVALID
                    {
                        errors[instance]=ERR_NOT_IMPLEMENTED
                    }
                    break;
                case OP_SELFDESTRUCT: // SELFDESTRUCT
                    {
                        errors[instance]=ERR_NOT_IMPLEMENTED
                    }
                    break;
            }
            execution_state[execution_step].pc=pc;
            execution_state[execution_step].stack_size=stack.size();
            execution_state[execution_step].opcode=opcode;
            stack.copy_stack_data(&(stacks[execution_step].values[0]));
            if (errors[instance]!=ERR_NONE)
                break;
            pc=pc+1;
        }
    }
    execution_lengths[instance]=execution_step;
    memory.copy_memory_data(&(debug_memory[instance]));
}

template<class params>
void cuEVM(unsigned char *bytecode, unsigned char *input, size_t bytecode_len, size_t input_len)
{
    typedef typename gpu_global_storage_t<params>::gpu_contract_t gpu_contract_t;
    typedef typename gpu_message<params>::gpu_message gpu_message_t;
    typedef typename gpu_stack_t<params>::stack_data_t stack_data_t;
  
    stack_data_t            *cpu_stacks, *gpu_stacks;
    cgbn_error_report_t     *report;
    gpu_block_t             *cpu_blocks;
    gpu_block_hash_t        *cpu_blocks_hash;
    gpu_contract_t          *cpu_contract, *gpu_contract;
    gpu_message_t           *cpu_messages, *gpu_messages;
    return_data_t           *cpu_returns, *gpu_returns;
    memory_data_t           *cpu_memory, *gpu_memory;
    uint32_t                *cpu_execution_lengths, *cpu_errors;
    uint32_t                *gpu_execution_lengths, *gpu_errors;
    
    // current block and last 256 blocks hash
    printf("Genereating blocks\n");
    cpu_blocks=generate_cpu_blocks<params>(1);
    cpu_blocks_hash=generate_cpu_blocks_hash<params>(256);
    
    printf("Copying blocks to GPU\n");
    CUDA_CHECK(cudaMemcpyToSymbol(gpu_current_block, cpu_blocks, sizeof(gpu_block_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(gpu_last_blocks_hash, cpu_blocks_hash, sizeof(gpu_block_hash_t)*256));
    
    // make the contract on cpu
    cpu_contract=gpu_global_storage_t<params>::generate_global_storage(1);
    // we initialise the bytecode and storage space
    // I consider that it does not have any storage space
    free(cpu_contract[0].bytecode);
    cpu_contract[0].storage_size=0;
    cpu_contract[0].bytecode_size=bytecode_len;
    cpu_contract[0].bytecode=(unsigned char *)malloc(bytecode_len);
    memcpy(cpu_contract[0].bytecode, bytecode, bytecode_len);
    // make the message on CPU and initiliase with the input data
    cpu_messages=generate_host_messages<params>(1);
    free(cpu_messages[0].data);
    cpu_messages[0].data_size=input_len;
    cpu_messages[0].data=(unsigned char *)malloc(input_len);
    memcpy(cpu_messages[0].data, input, input_len);

    // make the gpu contract and gpu message
    gpu_contract=gpu_global_storage_t<params>::generate_gpu_global_storage(cpu_contract, 1);
    cpu_messages[0].contract=&(gpu_contract[0]);
    gpu_messages=generate_gpu_messages<params>(cpu_messages, 1);
    
    // allocate the return information
    cpu_returns=generate_host_returns(1)
    gpu_returns=generate_gpu_returns(cpu_returns, 1);

    // allocate the execution lengths and errors
    cpu_execution_lengths=(uint32_t *)malloc(sizeof(uint32_t) * 1);
    cpu_execution_lengths[0]=0;
    cpu_errors=(uint32_t *)malloc(sizeof(uint32_t) * 1);
    cpu_errors[0]=ERR_NONE;
    CUDA_CHECK(cudaMalloc((void **)&gpu_execution_lengths, sizeof(uint32_t) * 1));
    CUDA_CHECK(cudaMalloc((void **)&gpu_errors, sizeof(uint32_t) * 1));
    CUDA_CHECK(cudaMemcpy(gpu_execution_lengths, cpu_execution_lengths, sizeof(uint32_t) * 1, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_errors, cpu_errors, sizeof(uint32_t) * 1, cudaMemcpyHostToDevice));

    // allocate the stacks
    cpu_stacks=gpu_stack_t<params>::generate_stack_data(MAX_EXECUTION_SIZE);
    CUDA_CHECK(cudaMalloc((void **)&gpu_stacks, sizeof(stack_data_t)*MAX_EXECUTION_SIZE));
    CUDA_CHECK(cudaMemcpy(gpu_stacks, cpu_stacks, sizeof(stack_data_t)*MAX_EXECUTION_SIZE, cudaMemcpyHostToDevice));
  
    // allocate the debug memory
    cpu_memory=gpu_memory_t::generate_memory_info_data(1);
    gpu_memory=gpu_memory_t::generate_gpu_memory_info_data(cpu_memory, 1);
    
    // create a cgbn_error_report for CGBN to report back errors
    CUDA_CHECK(cgbn_error_report_alloc(&report)); 

    // call the kernel
    printf("Calling the kernel\n");
    cu_evm_interpreter_kernel<params><<<1, 1>>>(report, gpu_messages, gpu_stacks, gpu_memory, gpu_returns, 1, gpu_execution_lengths, gpu_errors);

    // error report uses managed memory, so we sync the device (or stream) and check for cgbn errors
    CUDA_CHECK(cudaDeviceSynchronize());
    CGBN_CHECK(report);

    get_memory_from_gpu(cpu_instances, gpu_instances, instance_count);

    // CLEANUP
    printf("Freeing memory\n");
    // blocks
    free_host_blocks(cpu_blocks, 1);
    free_host_blocks_hash(cpu_blocks_hash, 256);
    gpu_global_storage_t<params>::free_global_storage(cpu_contract, 1);
    gpu_global_storage_t<params>::free_gpu_global_storage(gpu_contract, 1);
    free_host_messages<params>(cpu_messages, 1);
    free_gpu_messages<params>(gpu_messages, 1);
    free_host_returns(cpu_returns, 1);
    free_gpu_returns(gpu_returns, 1);
    free(cpu_execution_lengths);
    free(cpu_errors);
    CUDA_CHECK(cudaFree(gpu_execution_lengths));
    CUDA_CHECK(cudaFree(gpu_errors));
    free(cpu_stacks);
    CUDA_CHECK(cudaFree(gpu_stacks));
    CUDA_CHECK(cgbn_error_report_free(report));
}


int main(int argc, char *argv[])
{

    char *byte_code_hex = NULL;
    char *input_hex = NULL;

    static struct option long_options[] = {
        {"bytecode", required_argument, 0, 'b'},
        {"input", required_argument, 0, 'i'},
        {0, 0, 0, 0}};

    int opt;
    int option_index = 0;
    while ((opt = getopt_long(argc, argv, "b:i:", long_options, &option_index)) != -1)
    {
        switch (opt)
        {
        case 'b':
            byte_code_hex = optarg;
            break;
        case 'i':
            input_hex = optarg;
            break;
        default:
            fprintf(stderr, "Usage: %s --bytecode <hexstring> --input <hexstring>\n", argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    if (!byte_code_hex || !input_hex)
    {
        fprintf(stderr, "Both --bytecode and --input flags are required\n");
        exit(EXIT_FAILURE);
    }

    int bytecode_len = adjustedLength(&byte_code_hex);
    int input_len = adjustedLength(&input_hex);

    unsigned char *byte_code = (unsigned char *)malloc(bytecode_len);
    unsigned char *input = (unsigned char *)malloc(input_len);

    hexStringToByteArray(byte_code_hex, byte_code, bytecode_len * 2);
    hexStringToByteArray(input_hex, input, input_len * 2);


    // call my function
    cuEVM<utils_params>(byte_code, input, bytecode_len, input_len);

    printf("Pass conversion\n");
    unsigned char *d_bytecode, *d_input;
    cudaMalloc((void **)&d_bytecode, bytecode_len);
    cudaMalloc((void **)&d_input, input_len);

    cudaMemcpy(d_bytecode, byte_code, bytecode_len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, input, input_len, cudaMemcpyHostToDevice);
    printf("Pass allocation and memcpy\n");



    int blockSize = 256;
    int numBlocks = (NUMTHREAD + blockSize - 1) / blockSize;
    cuEVM<<<numBlocks, blockSize>>>(d_bytecode, d_input, bytecode_len, input_len, NUMTHREAD);
    printf("RUN\n");

    cudaDeviceSynchronize();
    printf("Syncrhronize\n");

    // clean up
    cudaFree(d_bytecode);
    cudaFree(d_input);
    free(byte_code);
    free(input);

    return 0;
}
