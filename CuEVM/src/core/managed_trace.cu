#include <CuEVM/utils/error_codes.cuh>
#include <CuEVM/utils/opcodes.cuh>
#include <iostream>
#include <string>
#include <CuEVM/managed_trace.cuh>
#include <cassert>
#include <sstream>
#include <iomanip>


__managed__ managed_trace_data_t *trace_data_buf = nullptr;
__managed__ uint32_t trace_data_buf_capacity = 1024 * 1024;
__managed__ uint32_t trace_data_buf_index = 0;

__managed__ CuEVM::evm_word_t *trace_stack_buf = nullptr;
__managed__ uint32_t trace_stack_buf_capacity = 100 * 1024;
__managed__ uint32_t trace_stack_buf_index = 0;

__host__ __device__ uint64_t limbs_to_uint64(const uint32_t *cgbn) {
  return (static_cast<uint64_t>(cgbn[0])) ;
}


__host__ __device__ void on_operation_start(CuEVM::ArithEnv &arith,
                                   const uint32_t pc,
                                   const uint8_t op,
                                   const CuEVM::evm_stack_t &stack,
                                   const uint32_t depth,
                                   const CuEVM::bn_t &gas_limit,
                                   const CuEVM::bn_t &gas_used){

#ifdef __CUDA_ARCH__
  CuEVM::bn_t gas;
  cgbn_sub(arith.env, gas, gas_limit, gas_used);
  if (threadIdx.x == 0) { // multiple instance to be handled
    managed_trace_data_t *trace = trace_data_buf + trace_data_buf_index;
    trace->pc = pc;
    trace->op = op;
    trace->depth = depth;
    trace->stack_size = stack.size();
    trace->gas = limbs_to_uint64(gas._limbs);
    trace->gas_cost = limbs_to_uint64(gas_used._limbs);
    trace_data_buf_index++;

    if (stack.size() > 0){
      trace->stack_index = trace_stack_buf_index;
      stack.extract_data(trace_stack_buf + trace_stack_buf_index);
      trace_stack_buf_index += stack.size();
    }
  }
  // __syncthreads();

#else
  assert(0); // Not supported on CPU
#endif
}


void on_operation_finish() {
}

void on_transaction_finish() {
}


void print_stack_as_hex(const uint32_t *limbs) {
    std::ostringstream hex_stream;
    for (int i = 0; i < 8; ++i) {
         hex_stream << std::setw(8) << std::setfill('0') << std::hex << limbs[i];
    }
    std::cout << hex_stream.str();
}

__host__ void dump_trace() {
  for (auto i = 0; i < trace_data_buf_index; i++){
    auto trace = trace_data_buf[i];
    auto pc = trace.pc;
    auto op = trace.op;
    auto gas = trace.gas;
    auto gas_cost = trace.gas_cost;
    auto depth = trace.depth;

    std::fprintf(stdout, "{\"pc\":%d,\"op\":%d,", pc, op);
    std::fprintf(stdout, "\"gas\":\"0x%llx\",", static_cast<unsigned long long>(gas));
    std::fprintf(stdout, "\"gasCost\":\"0x%llx\",", static_cast<unsigned long long>(gas_cost));

    std::fprintf(stdout, "\"stack\": [");
    for (auto j = 0; j < trace.stack_size; j++){
      auto stack = trace_stack_buf[j + trace.stack_index];
      stack.print_as_compact_hex();
      if (j != trace.stack_size - 1){
        std::fprintf(stdout, ",");
      }
    }
    std::fprintf(stdout, "],");

    std::fprintf(stdout, "\"depth\":%d", depth);

    std::fprintf(stdout, "}\n");
  }

  std::fprintf(stdout, "{\"stateRoot\":\"0x\"}\n");
}
