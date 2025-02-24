#include <CuEVM/tracer.cuh>
#include <cassert>

namespace CuEVM::utils {
// In evm.cu (host code)
void print_tracer_data(char* h_buffer) {
    size_t offset = 0;
    uint32_t status;
    memcpy(&status, h_buffer + offset, sizeof(uint32_t));
    offset += sizeof(uint32_t);
    printf("status: %d\n", status);
    if (status == -1) {
        printf("Error: Buffer overflow during serialization\n");
        return;
    }

    printf("\n\ninstance 10\n\n");  // Match original output

    // Read number of trace data points
    uint32_t size;
    memcpy(&size, h_buffer + offset, sizeof(uint32_t));
    printf("offset %d size: %d\n", offset, size);

    offset += sizeof(uint32_t);

    // Parse and print each trace data point
    for (uint32_t i = 0; i < size; i++) {
        uint32_t pc;
        memcpy(&pc, h_buffer + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);

        uint32_t op;
        memcpy(&op, h_buffer + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);

        uint64_t gas;
        memcpy(&gas, h_buffer + offset, sizeof(uint64_t));
        offset += sizeof(uint64_t);

        uint64_t gas_cost;
        memcpy(&gas_cost, h_buffer + offset, sizeof(uint64_t));
        offset += sizeof(uint64_t);

        uint32_t mem_size;
        memcpy(&mem_size, h_buffer + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);

        int depth;
        memcpy(&depth, h_buffer + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);

        uint32_t refund;
        memcpy(&refund, h_buffer + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);

        uint32_t stack_size;
        memcpy(&stack_size, h_buffer + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);

        printf(
            "{\"pc\":%u,\"op\":%u,\"gas\":\"0x%lx\",\"gasCost\":\"0x%lx\","
            "\"memSize\":%u,\"stack\":[",
            pc, op, gas, gas_cost, mem_size);
        if (stack_size > 100) {
            printf("stack_size error : %u\n", stack_size);
            break;
        }

        // Parse and print stack elements
        for (uint32_t j = 0; j < stack_size; j++) {
            evm_word_t val;
            memcpy(&val, h_buffer + offset, sizeof(evm_word_t));
            offset += sizeof(evm_word_t);
            char* val_hex = val.to_hex(nullptr, true);
            printf("\"%s\"", val_hex);
            if (j < stack_size - 1) {
                printf(",");
            }
            delete[] val_hex;
        }
        printf("],\"depth\":%d,\"refund\":%u}\n", depth, refund);
    }

    // Parse and print additional fields
    uint64_t gas_used;
    memcpy(&gas_used, h_buffer + offset, sizeof(uint64_t));
    offset += sizeof(uint64_t);

    bool pass;
    memcpy(&pass, h_buffer + offset, sizeof(bool));
    offset += sizeof(bool);

    uint32_t time;
    memcpy(&time, h_buffer + offset, sizeof(uint32_t));
    offset += sizeof(uint32_t);

    printf("{\"stateRoot\":\"0x\",");

    printf("\"gasUsed\":\"0x%lx\",", gas_used);
    printf("\"pass\":\"%s\",", pass ? "true" : "false");
    printf("\"time\":%u}\n", time);
}

__device__ bool trace_data_t::serialize(char* buf, uint32_t buf_size, uint32_t& offset) {
    // Serialize fixed-size fields
    // printf("trace_data_t serialize pc %u offset: %d bufsize: %d\n", pc, offset, buf_size);
    if (offset + sizeof(uint32_t) > buf_size) return false;

    memcpy(buf + offset, &pc, sizeof(uint32_t));
    offset += sizeof(uint32_t);

    if (offset + sizeof(uint32_t) > buf_size) return false;
    memcpy(buf + offset, &op, sizeof(uint32_t));
    offset += sizeof(uint32_t);

    if (offset + sizeof(uint64_t) > buf_size) return false;
    memcpy(buf + offset, &gas, sizeof(uint64_t));
    offset += sizeof(uint64_t);

    if (offset + sizeof(uint64_t) > buf_size) return false;
    memcpy(buf + offset, &gas_cost, sizeof(uint64_t));
    offset += sizeof(uint64_t);

    if (offset + sizeof(uint32_t) > buf_size) return false;
    memcpy(buf + offset, &mem_size, sizeof(uint32_t));
    offset += sizeof(uint32_t);

    if (offset + sizeof(uint32_t) > buf_size) return false;
    memcpy(buf + offset, &depth, sizeof(uint32_t));
    offset += sizeof(uint32_t);

    if (offset + sizeof(uint32_t) > buf_size) return false;
    memcpy(buf + offset, &refund, sizeof(uint32_t));
    offset += sizeof(uint32_t);

    // Serialize stack size
    if (offset + sizeof(uint32_t) > buf_size) return false;
    memcpy(buf + offset, &stack_size, sizeof(uint32_t));
    offset += sizeof(uint32_t);

    // Serialize stack elements
    for (uint32_t i = 0; i < stack_size; i++) {
        if (offset + sizeof(evm_word_t) > buf_size) return false;
        memcpy(buf + offset, &stack[i], sizeof(evm_word_t));
        offset += sizeof(evm_word_t);
    }

    return true;
}

__device__ bool tracer_t::serialize(char* buf, uint32_t buf_size, uint32_t& offset) {
    printf("tracer serialize buff %p offset: %d bufsize: %d\n", buf, offset, buf_size);
    // Serialize number of trace data points
    if (offset + sizeof(uint32_t) > buf_size) return false;

    memcpy(buf + offset, &size, sizeof(uint32_t));

    offset += sizeof(uint32_t);

    printf("size: %u\n", size);
    // Serialize each trace data point
    for (uint32_t i = 0; i < size; i++) {
        if (!data[i].serialize(buf, buf_size, offset)) {
            return false;
        }
    }

    // Serialize additional fields
    if (offset + sizeof(uint64_t) > buf_size) return false;
    memcpy(buf + offset, &gas_used, sizeof(uint64_t));
    offset += sizeof(uint64_t);

    // Serialize status (ERROR_RETURN or ERROR_REVERT)
    bool pass = (status == ERROR_RETURN) || (status == ERROR_REVERT);
    if (offset + sizeof(bool) > buf_size) return false;
    memcpy(buf + offset, &pass, sizeof(bool));
    offset += sizeof(bool);

    // Serialize return_data (assuming it's stored as a hex string)
    // For simplicity, we'll serialize the hex string length and data

    // Serialize time (hardcoded to 2 as in original)
    uint32_t time = 2;
    if (offset + sizeof(uint32_t) > buf_size) return false;
    memcpy(buf + offset, &time, sizeof(uint32_t));
    offset += sizeof(uint32_t);

    return true;
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
        char* val = stack[i].to_hex(nullptr, true);
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
    trace_data_t* new_data = new trace_data_t[capacity + 128];
    if (data != nullptr) {
        memcpy(new_data, data, sizeof(trace_data_t) * size);
        delete[] data;
    }
    data = new_data;
    capacity += 128;
}

__device__ void tracer_t::start_operation(const uint32_t pc, const uint8_t op, const CuEVM::evm_memory_t* memory,
                                          const CuEVM::evm_stack_t* stack, const uint32_t depth,
                                          const byte_array_t* return_data, const CuEVM::gas_t& gas_limit,
                                          const CuEVM::gas_t& gas_used) {
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

__device__ void tracer_t::finish_operation(const CuEVM::gas_t& gas_used, const CuEVM::gas_t& gas_refund) {
    if (size == 0) return;
    data[size - 1].gas_cost = gas_used - data[size - 1].gas_cost;
    data[size - 1].refund = gas_refund;
}

__device__ void tracer_t::finish_transaction(const byte_array_t* return_data, const CuEVM::gas_t& gas_used,
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

    char* return_data_hex = nullptr;  // return_data->to_hex();

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
