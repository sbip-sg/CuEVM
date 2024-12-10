#ifndef TRACE_DATA_H
#define TRACE_DATA_H

#include <cuda_runtime.h>
#include <CuEVM/core/stack.cuh>
#include <CuEVM/utils/arith.cuh>
#include <CuEVM/utils/evm_defines.cuh>


struct managed_trace_data_t { // need to align memory?
  uint32_t pc;
  uint8_t op;
  uint64_t gas; // gas left before current operation
  uint64_t gas_cost; // gas cost of the current operation
  uint32_t stack_index;
  uint32_t stack_size;
  uint32_t depth;
};


extern __managed__ managed_trace_data_t *trace_data_buf;
extern __managed__ uint32_t trace_data_buf_capacity;
extern __managed__ uint32_t trace_data_buf_index;

extern __managed__ CuEVM::evm_word_t *trace_stack_buf;
extern __managed__ uint32_t trace_stack_buf_capacity;
extern __managed__ uint32_t trace_stack_buf_index;



extern __host__ __device__ void on_operation_start(CuEVM::ArithEnv &arith,
                                                   const uint32_t pc,
                                                   const uint8_t op,
                                                   const CuEVM::evm_stack_t &stack,
                                                   const uint32_t depth,
                                                   const CuEVM::bn_t &gas_limit,
                                                   const CuEVM::bn_t &gas_used);

extern __device__ void on_operation_finish();

extern __device__ void on_transaction_finish();

extern __host__ void dump_trace();

#endif
