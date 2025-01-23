#include <CuEVM/tracer.cuh>

namespace CuEVM::utils {

__device__ void simplified_trace_data::start_operation(const uint32_t pc, const uint8_t op,
                                                       const CuEVM::evm_stack_t &stack_ptr) {
    if (no_events >= MAX_TRACE_EVENTS) return;
    events[no_events].pc = pc;
    events[no_events].op = op;
    if (op != OP_INVALID && op != OP_SELFDESTRUCT) {
        // printf("add new operation, src data %d \n", THREADIDX);
        // printf("stack size %d\n", stack_ptr.size());

        events[no_events].operand_1 = *stack_ptr.get_address_at_index(1);
        events[no_events].operand_2 = *stack_ptr.get_address_at_index(2);
    }
}

__device__ void simplified_trace_data::record_branch(uint32_t pc_src, uint32_t pc_dst, uint32_t pc_missed) {
    if (no_branches >= MAX_BRANCHES_TRACING) no_branches = 0;
    branches[no_branches].pc_src = pc_src;
    branches[no_branches].pc_dst = pc_dst;
    branches[no_branches].pc_missed = pc_missed;
    branches[no_branches].distance = last_distance;
    // printf("record branch pc_src %u pc_dst %u distance %s\n", pc_src, pc_dst,
    // branches[no_branches].distance.to_hex());
    no_branches++;
}

__device__ void simplified_trace_data::record_distance(uint8_t op, const CuEVM::evm_stack_t &stack_ptr) {
    evm_word_t distance, op1, op2;
    uint32_t stack_size = stack_ptr.size();

    op1 = *stack_ptr.get_address_at_index(1);
    op2 = *stack_ptr.get_address_at_index(2);

    if (uint256_cmp(&op1, &op2) >= 1)
        uint256_sub(&distance, &op1, &op2);
    else
        uint256_sub(&distance, &op2, &op1);

    if (op != OP_EQ) uint256_add_word(&distance, &distance, 1);

    last_distance = distance;
}

__device__ void simplified_trace_data::finish_operation(const CuEVM::evm_stack_t &stack_ptr, uint32_t error_code) {
    if (no_events >= MAX_TRACE_EVENTS) return;
    if (events[no_events].op < OP_REVERT && events[no_events].op != OP_SSTORE)
        events[no_events].res = *stack_ptr.get_address_at_index(1);
    no_events++;
}
__device__ void simplified_trace_data::start_call(uint32_t pc, evm_call_context_t *call_context_ptr) {
    if (no_calls >= MAX_CALLS_TRACING) return;
    // add address and increment current_address_idx
    // addresses[current_address_idx] = cached_call_state->addresses[cached_call_state->current_address_idx];
    // printf("start call simplified trace data pc %d op %d\n", pc, message_call_ptr->call_type);
    calls[no_calls].sender = call_context_ptr->from;
    calls[no_calls].receiver = call_context_ptr->to;
    calls[no_calls].pc = pc;
    calls[no_calls].op = call_context_ptr->call_type;
    calls[no_calls].value = call_context_ptr->value;

    no_calls++;
}
__device__ void simplified_trace_data::finish_call(uint8_t success) {
    if (no_calls > MAX_CALLS_TRACING) return;

    // printf("no_calls %u \n", no_calls);
    for (int i = no_calls - 1; i >= 0; i--) {
        if (calls[i].success == UINT8_MAX) {
            calls[i].success = success;
            break;
        }
    }
}
__device__ void simplified_trace_data::print() {
    printf("no_events %u\n", no_events);
    printf("no_calls %u\n", no_calls);
    printf("events\n");
    for (uint32_t i = 0; i < no_events; i++) {
        printf("pc %u op %u operand_1 %s operand_2 %s res %s\n", events[i].pc, events[i].op,
               events[i].operand_1.to_hex(), events[i].operand_2.to_hex(), events[i].res.to_hex());
    }
    printf("calls\n");
    for (uint32_t i = 0; i < no_calls; i++) {
        printf("pc %u op %u sender %s receiver %s value %s success %u\n", calls[i].pc, calls[i].op,
               calls[i].sender.to_hex(), calls[i].receiver.to_hex(), calls[i].value.to_hex(), calls[i].success);
    }
    printf("branches\n");
    for (uint32_t i = 0; i < no_branches; i++) {
        printf("pc_src %u pc_dst %u distance %s\n", branches[i].pc_src, branches[i].pc_dst,
               branches[i].distance.to_hex());
    }
}

__device__ void trace_data_t::print_err() {
    printf("{\"pc\":%d,\"op\":%d,", pc, op);
    printf("\"gas\":\"0x%lx\",", gas);

    printf("\"gasCost\":\"0x%lx\",", gas_cost);

    printf("\"memSize\":%u,", mem_size);

    printf("\"stack\":");

    // print uint256 stack values
    printf("[");
    for (uint32_t i = 0; i < stack_size; i++) {
        char *val = stack[i].to_hex(nullptr, true);
        if (i != stack_size - 1) {
            printf("\"%s\",", val);
        } else {
            printf("\"%s\"", val);
        }
    }
    printf("],");

    printf("\"depth\":%d,", depth);

    printf("\"refund\":%u", refund);

    printf("}\n");
}

__device__ tracer_t::tracer_t() : data(nullptr), size(0), capacity(0) {}

__device__ tracer_t::~tracer_t() {
    if (data != nullptr) {
        for (uint32_t i = 0; i < size; i++) {
            delete[] data[i].stack;
        }
        delete[] data;
    }
}

__device__ void tracer_t::grow() {
    trace_data_t *new_data = new trace_data_t[capacity + 128];
    if (data != nullptr) {
        memcpy(new_data, data, sizeof(trace_data_t) * size);
        delete[] data;
    }
    data = new_data;
    capacity += 128;
}

__device__ void tracer_t::start_operation(const uint32_t pc, const uint8_t op, const CuEVM::evm_memory_t *memory,
                                          const CuEVM::evm_stack_t *stack, const uint32_t depth,
                                          const byte_array_t *return_data, const CuEVM::gas_t &gas_limit,
                                          const CuEVM::gas_t &gas_used) {
    if (size == capacity) {
        grow();
    }

    data[size].pc = pc;
    data[size].op = op;
    data[size].mem_size = memory->size;

    gas_t gas = gas_limit - gas_used;
    data[size].gas = gas;
    data[size].gas_cost = gas_used;

    data[size].stack_size = stack->stack_offset;
    if (data[size].stack_size > 0) {
        data[size].stack = new evm_word_t[data[size].stack_size];
        stack->extract_data(data[size].stack);
    }

    // for (int i = 0; i < data[size].stack_size; i++) {
    //     data[size].stack[i].print();
    // }

    data[size].depth = depth;
    // TODO: fix this
    // data[size].return_data = new byte_array_t();
    // data[size].return_data = return_data;
    size++;
}

__device__ void tracer_t::finish_operation(const CuEVM::gas_t &gas_used, const CuEVM::gas_t &gas_refund) {
    if (size == 0) return;
    data[size - 1].gas_cost = gas_used - data[size - 1].gas_cost;
    data[size - 1].refund = gas_refund;
}

__device__ void tracer_t::finish_transaction(const byte_array_t *return_data, const CuEVM::gas_t &gas_used,
                                             uint32_t error_code) {
    // TODO: fix this
    this->return_data = new byte_array_t();
    this->gas_used = gas_used;
    this->status = error_code;
}

__device__ void tracer_t::print() {
    for (uint32_t i = 0; i < size; i++) {
        printf("PC: %d\n", data[i].pc);
        printf("Opcode: %d\n", data[i].op);
        printf("Gas: %d\n", data[i].gas);
        printf("Gas cost: %d\n", data[i].gas_cost);
        printf("Stack: ");
        for (uint32_t j = 0; j < data[i].stack_size; j++) {
            data[i].stack[j].print();
        }
        printf("Depth: %d\n", data[i].depth);
        printf("Memory size: %d\n", data[i].mem_size);
        // printf("Return data: ");
        // data[i].return_data->print();
        printf("Refund: %d\n", data[i].refund);
#ifdef EIP_3155_OPTIONAL
        printf("Error code: %d\n", data[i].error_code);
        printf("Memory: ");
        for (uint32_t j = 0; j < data[i].mem_size; j++) {
            printf("%02x", data[i].memory[j]);
        }
        printf("\n");
// printf("Storage: ");
// data[i].storage.print();
#endif
    }
}

__device__ void tracer_t::print_err() {
    for (uint32_t i = 0; i < size; i++) {
        data[i].print_err();
    }
    printf("{\"stateRoot\":\"0x\",");

    char *return_data_hex = nullptr;  // return_data->to_hex();

    if (return_data_hex != nullptr) {
        if (return_data_hex[2] != '\0') {  // more than `0x` stored in the string
            printf("\"output\":\"%s\",", return_data_hex);
        } else {
            printf("\"output\":\"\",");
        }
        delete[] return_data_hex;
    } else {
        printf("\"output\":\"\",");
    }
    printf("\"gasUsed\":\"0x%lx\",", gas_used);

    printf("\"pass\":\"%s\",", (status == ERROR_RETURN) || (status == ERROR_REVERT) ? "true" : "false");

    // fprintf(stderr, "\"fork\":%s,", "\"\"\"\"");

    printf("\"time\":%u}\n", 2);
}

}  // namespace CuEVM::utils
// EIP-3155
