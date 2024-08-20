// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2023-11-30
// SPDX-License-Identifier: MIT

#include "include/tracer.cuh"
#include <string>

namespace cuEVM::utils {
    __host__ cJSON* trace_data_t::to_json() {
        char *hex_string_ptr = new char[cuEVM::word_size * 2 + 3];
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
        cJSON_AddItemToObject(json, "returnData", return_data.to_json());
        cJSON_AddStringToObject(json, "refund", refund.to_hex(hex_string_ptr));
        #ifdef EIP_3155_OPTIONAL
        cJSON_AddNumberToObject(json, "errorCode", error_code);
        cuEVM::byte_array_t memory_array(memory, mem_size);
        cJSON_AddItemToObject(json, "memory", memory_array.to_json());
        //cJSON_AddItemToObject(json, "touchState", touch_state.json());
        #endif
        delete [] hex_string_ptr;
        return json;
    }

    __host__ void trace_data_t::print_err() {
        std::string stack_str;
        if (stack_size > 0){
            stack_str += "\"";
            for (auto index =0; index<stack_size; index++){
                std::string temp = stack[index].to_hex();
                stack_str += temp;
                if (index == stack_size - 1) {
                    stack_str += "\"";
                } else {
                    stack_str += "\",\"";
                }
            }
        }
        fprintf(stderr, "{\"pc\":%d,\"op\":%d,\"gas\":\"%s\",\"gasCost\":\"%s\",\"stack\":[%s],\"depth\":%d,\"memSize\":%lu}\n",
            pc, op, gas.to_hex(), gas_cost.to_hex(), stack_str.c_str(), depth, mem_size);
    }

    __host__ __device__ tracer_t::tracer_t() : data(nullptr), size(0), capacity(0) {}

    __host__ __device__ tracer_t::~tracer_t() {
        if (data != nullptr) {
            for (uint32_t i = 0; i < size; i++) {
                delete[] data[i].stack;
                #ifdef EIP_3155_OPTIONAL
                delete[] data[i].memory;
                #endif
            }
            delete[] data;
        }
    }

    __host__ __device__ void tracer_t::grow() {
        trace_data_t *new_data = new trace_data_t[capacity + 128];
        if (data != nullptr) {
            memcpy(new_data, data, sizeof(trace_data_t) * size);
            delete[] data;
        }
        data = new_data;
        capacity += 128;
    }

    __host__ __device__ uint32_t tracer_t::push_init(
        ArithEnv &arith,
        const uint32_t pc,
        const uint8_t op,
        const cuEVM::evm_memory_t &memory,
        const cuEVM::evm_stack_t &stack,
        const uint32_t depth,
        const cuEVM::evm_return_data_t &return_data,
        const bn_t &gas
    ) {
        if (size == capacity) {
            grow();
        }
        data[size].pc = pc;
        data[size].op = op;
        data[size].mem_size = memory.get_size();
        cgbn_store(arith.env, (cgbn_evm_word_t_ptr) &data[size].gas, gas);
        data[size].stack_size = stack.size();
        data[size].stack = new evm_word_t[data[size].stack_size];
        memcpy(data[size].stack, stack.stack_base, sizeof(evm_word_t) * data[size].stack_size);
        data[size].depth = depth;
        data[size].return_data = return_data;
        #ifdef EIP_3155_OPTIONAL
        data[size].memory = new uint8_t[data[size].mem_size];
        memcpy(data[size].memory, memory.data.data, data[size].mem_size);
        #endif
        return size++;
    }

    __host__ __device__ void tracer_t::push_final(
        ArithEnv &arith,
        const uint32_t idx,
        const bn_t &gas_cost,
        const bn_t &refund
        #ifdef EIP_3155_OPTIONAL
        , const uint32_t error_code,
        const cuEVM::state::TouchState &touch_state
        #endif
    ) {
        cgbn_store(arith.env, &data[idx].gas_cost, gas_cost);
        cgbn_store(arith.env, &data[idx].refund, refund);
        #ifdef EIP_3155_OPTIONAL
        data[idx].error_code = error_code;
        data[idx].touch_state = touch_state;
        #endif
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
            data[i].return_data.print();
            printf("Refund: ");
            data[i].refund.print();
            #ifdef EIP_3155_OPTIONAL
            printf("Error code: %d\n", data[i].error_code);
            printf("Memory: ");
            for (uint32_t j = 0; j < data[i].mem_size; j++) {
                printf("%02x", data[i].memory[j]);
            }
            printf("\n");
            printf("Touch state: ");
            cuEVM::state::TouchState::print(data[i].touch_state);
            #endif
        }
    }

    __host__ cJSON* tracer_t::to_json() {
        cJSON *json = cJSON_CreateArray();
        for (uint32_t i = 0; i < size; i++) {
            cJSON_AddItemToArray(json, data[i].to_json());
        }
        return json;
    }

}
// EIP-3155
