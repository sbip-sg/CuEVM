// CuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation
// Programme) Author: Stefan-Dan Ciocirlan Data: 2023-11-30
// SPDX-License-Identifier: MIT

#include <CuEVM/tracer.cuh>
#include <CuEVM/utils/error_codes.cuh>
#include <CuEVM/utils/opcodes.cuh>
#include <iostream>
#include <string>

namespace CuEVM::utils {

__host__ __device__ void simplified_trace_data::start_operation(const uint32_t pc, const uint8_t op,
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

__host__ __device__ void simplified_trace_data::record_branch(uint32_t pc_src, uint32_t pc_dst, uint32_t pc_missed) {
    if (no_branches >= MAX_BRANCHES_TRACING) no_branches = 0;
    branches[no_branches].pc_src = pc_src;
    branches[no_branches].pc_dst = pc_dst;
    branches[no_branches].pc_missed = pc_missed;
    branches[no_branches].distance = last_distance;
    // printf("record branch pc_src %u pc_dst %u distance %s\n", pc_src, pc_dst,
    // branches[no_branches].distance.to_hex());
    no_branches++;
}

__host__ __device__ void simplified_trace_data::record_distance(ArithEnv &arith, uint8_t op,
                                                                const CuEVM::evm_stack_t &stack_ptr) {
    bn_t distance, op1, op2;
    uint32_t stack_size = stack_ptr.size();

    cgbn_load(arith.env, op1, stack_ptr.get_address_at_index(1));
    cgbn_load(arith.env, op2, stack_ptr.get_address_at_index(2));

    if (cgbn_compare(arith.env, op1, op2) >= 1)
        cgbn_sub(arith.env, distance, op1, op2);
    else
        cgbn_sub(arith.env, distance, op2, op1);

    if (op != OP_EQ) cgbn_add_ui32(arith.env, distance, distance, 1);

    cgbn_store(arith.env, &last_distance, distance);
}

__host__ __device__ void simplified_trace_data::finish_operation(const CuEVM::evm_stack_t &stack_ptr,
                                                                 uint32_t error_code) {
    if (no_events >= MAX_TRACE_EVENTS) return;
    if (events[no_events].op < OP_REVERT && events[no_events].op != OP_SSTORE)
        events[no_events].res = *stack_ptr.get_address_at_index(1);
    no_events++;
}
__host__ __device__ void simplified_trace_data::start_call(uint32_t pc, evm_message_call_t *message_call_ptr) {
    if (no_calls >= MAX_CALLS_TRACING) return;
    // add address and increment current_address_idx
    // addresses[current_address_idx] = cached_call_state->addresses[cached_call_state->current_address_idx];
    // printf("start call simplified trace data pc %d op %d\n", pc, message_call_ptr->call_type);
    calls[no_calls].sender = message_call_ptr->sender;
    calls[no_calls].receiver = message_call_ptr->recipient;
    calls[no_calls].pc = pc;
    calls[no_calls].op = message_call_ptr->call_type;
    calls[no_calls].value = message_call_ptr->value;
    __ONE_GPU_THREAD_WOSYNC_BEGIN__
    no_calls++;
    __ONE_GPU_THREAD_END__
}
__host__ __device__ void simplified_trace_data::finish_call(uint8_t success) {
    if (no_calls > MAX_CALLS_TRACING) return;
    __ONE_GPU_THREAD_WOSYNC_BEGIN__

    // printf("no_calls %u \n", no_calls);
    for (int i = no_calls - 1; i >= 0; i--) {
        if (calls[i].success == UINT8_MAX) {
            calls[i].success = success;
            break;
        }
    }
    __ONE_GPU_THREAD_END__
}
__host__ __device__ void simplified_trace_data::print() {
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

__host__ cJSON *trace_data_t::to_json() {
    char *hex_string_ptr = new char[CuEVM::word_size * 2 + 3];
    cJSON *json = cJSON_CreateObject();
    cJSON_AddNumberToObject(json, "pc", pc);
    cJSON_AddNumberToObject(json, "op", op);
    cJSON_AddStringToObject(json, "gas", gas.to_hex(hex_string_ptr));

    cJSON_AddStringToObject(json, "gasCost", gas_cost.to_hex(hex_string_ptr));
    cJSON_AddNumberToObject(json, "memSize", mem_size);
    cJSON *stack_json = cJSON_CreateArray();
    for (uint32_t i = 0; i < stack_size; i++) {
        cJSON_AddItemToArray(stack_json, cJSON_CreateString(stack[i].to_hex(hex_string_ptr)));
    }
    cJSON_AddItemToObject(json, "stack", stack_json);
    cJSON_AddNumberToObject(json, "depth", depth);
    cJSON_AddItemToObject(json, "returnData", return_data->to_json());
    cJSON_AddStringToObject(json, "refund", refund.to_hex(hex_string_ptr));
#ifdef EIP_3155_OPTIONAL
    cJSON_AddNumberToObject(json, "errorCode", error_code);
    CuEVM::byte_array_t memory_array(memory, mem_size);
    cJSON_AddItemToObject(json, "memory", memory_array.to_json());
// cJSON_AddItemToObject(json, "touchState", touch_state.json());
#endif
    delete[] hex_string_ptr;
    return json;
}

// __device__ void tracer_t::print_tracer_data(trace_data_t *data, uint32_t size) {
//     for (uint32_t i = 0; i < size; i++) {
//         printf("pc %u op %u\n", data[i].pc, data[i].op);
//     }
// }
// New device function to copy and print tracer data
__device__ void tracer_t::print_device_err() {}

__host__ __device__ void trace_data_t::print_err(char *hex_string_ptr) {
    char *tmp = nullptr;
    if (hex_string_ptr == nullptr) {
        tmp = new char[CuEVM::word_size * 2 + 3];
        hex_string_ptr = tmp;
    }

    printf("{\"pc\":%d,\"op\":%d,", pc, op);

    printf("\"gas\":\"%s\",", gas.to_hex(hex_string_ptr, 1));

    printf("\"gasCost\":\"%s\",", gas_cost.to_hex(hex_string_ptr, 1));

    printf("\"memSize\":%u,", mem_size);

    printf("\"stack\":");

    // print uint256 stack values
    printf("[");
    for (uint32_t i = 0; i < stack_size; i++) {
        stack[i].print_as_compact_hex();
        if (i != stack_size - 1) {
            printf(",");
        }
    }
    printf("],");

    printf("\"depth\":%d,", depth);

    // TODO: strupid to just show the least significant 32 bits
    // correct way is to show the whole 256 bits
    // fprintf(stderr, "\"refund\":\"%s\"}\n", refund.to_hex(hex_string_ptr,
    // 1));
    printf("\"refund\":%u", refund._limbs[0]);
#ifdef EIP_3155_OPTIONAL
    printf(",\"error\":%u", error_code);
    printf(",\"memory\":\"0x");
    for (uint32_t j = 0; j < mem_size; j++) {
        printf("%02x", memory[j]);
    }
    printf("\"");

#endif
    printf("}\n");
    if (tmp != nullptr) {
        delete[] tmp;
    }
}

__host__ __device__ tracer_t::tracer_t() : data(nullptr), size(0), capacity(0) {}

__host__ __device__ tracer_t::~tracer_t() {
    if (data != nullptr) {
        for (uint32_t i = 0; i < size; i++) {
            delete[] data[i].stack;
#ifdef EIP_3155_OPTIONAL
            delete[] data[i].memory;
            delete data[i].return_data;
#endif
        }
        delete[] data;
    }
}

__host__ __device__ void tracer_t::grow() {
    __SHARED_MEMORY__ trace_data_t *new_data[CGBN_IBP];
    __ONE_GPU_THREAD_WOSYNC_BEGIN__
    new_data[INSTANCE_IDX_PER_BLOCK] = new trace_data_t[capacity + 128];
    if (data != nullptr) {
        memcpy(new_data[INSTANCE_IDX_PER_BLOCK], data, sizeof(trace_data_t) * size);
        delete[] data;
    }
    __ONE_GPU_THREAD_END__
    data = new_data[INSTANCE_IDX_PER_BLOCK];
    __SYNC_THREADS__
    capacity += 128;
}

__host__ __device__ uint32_t tracer_t::start_operation(ArithEnv &arith, const uint32_t pc, const uint8_t op,
                                                       const CuEVM::evm_memory_t &memory,
                                                       const CuEVM::evm_stack_t &stack, const uint32_t depth,
                                                       const CuEVM::evm_return_data_t &return_data,
                                                       const bn_t &gas_limit, const bn_t &gas_used) {
    if (size == capacity) {
        grow();
    }

    // printf("tracer op %d idx %d size %d after grow\n", op, THREADIDX, size);

    __ONE_GPU_THREAD_WOSYNC_BEGIN__
    data[size].pc = pc;
    data[size].op = op;
    data[size].mem_size = memory.size;
    __ONE_GPU_THREAD_END__
    bn_t gas;
    cgbn_sub(arith.env, gas, gas_limit, gas_used);
    cgbn_store(arith.env, (cgbn_evm_word_t_ptr) & (data[size].gas), gas);
    cgbn_store(arith.env, (cgbn_evm_word_t_ptr) & (data[size].gas_cost), gas_used);
    // #ifdef __CUDA_ARCH__
    //     printf("tracer op %d idx %d after storing gas cost\n", op, threadIdx.x);
    // #endif
    __ONE_GPU_THREAD_WOSYNC_BEGIN__
    data[size].stack_size = stack.size();
    if (data[size].stack_size > 0) {
        data[size].stack = new evm_word_t[data[size].stack_size];
        stack.extract_data(data[size].stack);
        // // std::copy(stack.stack_base, stack.stack_base + stack.size(), data[size].stack);
        // memcpy(data[size].stack, stack.stack_base, sizeof(evm_word_t) * data[size].stack_size);
    }

    data[size].depth = depth;
    __ONE_GPU_THREAD_END__  // sync here
#ifndef GPU                 // reduce complication in gpu code
        data[size]
            .return_data = new byte_array_t(return_data);
#endif

#ifdef EIP_3155_OPTIONAL
    data[size].memory = new uint8_t[data[size].mem_size];
    // std::copy(memory.data.data, memory.data.data + data[size].mem_size, data[size].memory);
    memcpy(data[size].memory, memory.data.data, data[size].mem_size);
#endif

    return size++;
}

__host__ __device__ void tracer_t::finish_operation(ArithEnv &arith, const uint32_t idx, const bn_t &gas_used,
                                                    const bn_t &gas_refund
#ifdef EIP_3155_OPTIONAL
                                                    ,
                                                    const uint32_t error_code
// , const CuEVM::contract_storage_t &storage
#endif
) {
    bn_t gas_cost;
    cgbn_load(arith.env, gas_cost, (cgbn_evm_word_t_ptr) & (data[idx].gas_cost));
    cgbn_sub(arith.env, gas_cost, gas_used, gas_cost);
    cgbn_store(arith.env, (cgbn_evm_word_t_ptr) & (data[idx].gas_cost), gas_cost);
    cgbn_store(arith.env, (cgbn_evm_word_t_ptr) & (data[idx].refund), gas_refund);
#ifdef EIP_3155_OPTIONAL
    __ONE_GPU_THREAD_WOSYNC_BEGIN__
    data[idx].error_code = error_code;
    __ONE_GPU_THREAD_WOSYNC_END__
// data[idx].storage = storage;
#endif
}

__host__ __device__ void tracer_t::finish_transaction(ArithEnv &arith, const CuEVM::evm_return_data_t &return_data,
                                                      const bn_t &gas_used, uint32_t error_code) {
    this->return_data = return_data;
    cgbn_store(arith.env, (cgbn_evm_word_t_ptr) & (this->gas_used), gas_used);
    this->status = error_code;
}

__host__ __device__ void tracer_t::print(ArithEnv &arith) {
    for (uint32_t i = 0; i < size; i++) {
        printf("PC: %d\n", data[i].pc);
        printf("Opcode: %d\n", data[i].op);
        printf("Gas: ");
        data[i].gas.print();
        printf("Gas cost: ");
        data[i].gas_cost.print();
        printf("Stack: ");
        for (uint32_t j = 0; j < data[i].stack_size; j++) {
            data[i].stack[j].print();
        }
        printf("Depth: %d\n", data[i].depth);
        printf("Memory size: %d\n", data[i].mem_size);
        printf("Return data: ");
        data[i].return_data->print();
        printf("Refund: ");
        data[i].refund.print();
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

__host__ __device__ void tracer_t::print_err() {
    char *hex_string_ptr = new char[CuEVM::word_size * 2 + 3];
    for (uint32_t i = 0; i < size; i++) {
        data[i].print_err(hex_string_ptr);
    }
    printf("{\"stateRoot\":\"0x\",");

    char *return_data_hex = return_data.to_hex();

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

    printf("\"gasUsed\":\"%s\",", gas_used.to_hex(hex_string_ptr, 1));

    printf("\"pass\":\"%s\",", (status == ERROR_RETURN) || (status == ERROR_REVERT) ? "true" : "false");

    // fprintf(stderr, "\"fork\":%s,", "\"\"\"\"");

    printf("\"time\":%u}\n", 2);
    delete[] hex_string_ptr;
}

__host__ cJSON *tracer_t::to_json() {
    cJSON *json = cJSON_CreateArray();
    for (uint32_t i = 0; i < size; i++) {
        cJSON_AddItemToArray(json, data[i].to_json());
    }
    return json;
}

}  // namespace CuEVM::utils
// EIP-3155
