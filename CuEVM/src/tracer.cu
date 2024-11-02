// CuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation
// Programme) Author: Stefan-Dan Ciocirlan Data: 2023-11-30
// SPDX-License-Identifier: MIT

#include <CuEVM/tracer.cuh>
#include <CuEVM/utils/error_codes.cuh>
#include <iostream>
#include <string>

namespace CuEVM::utils {
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
    trace_data_t *new_data;
    __ONE_GPU_THREAD_WOSYNC_BEGIN__
    new_data = new trace_data_t[capacity + 128];
    if (data != nullptr) {
        memcpy(new_data, data, sizeof(trace_data_t) * size);
        delete[] data;
    }
    __ONE_GPU_THREAD_END__
    data = new_data;
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
    // #ifdef __CUDA_ARCH__
    //     printf("tracer op %d idx %d after  grow\n", op, threadIdx.x);
    // #endif
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
