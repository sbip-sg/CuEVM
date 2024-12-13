#include <CuEVM/core/stack.cuh>
#include <CuEVM/utils/error_codes.cuh>

namespace CuEVM::stack {

__device__ evm_stack_t::evm_stack_t(evm_word_t *shared_stack_base, uint32_t stack_base_offset)
    : shared_stack_base(shared_stack_base),
      global_stack_base(nullptr),
      stack_base_offset(stack_base_offset),
      stack_offset(0) {}

__device__ evm_stack_t::~evm_stack_t() { free(); }

// __device__ evm_stack_t::evm_stack_t(const evm_stack_t &other) {
//     // free();
//     duplicate(other);
// }

__device__ void evm_stack_t::free() {
    if (global_stack_base != nullptr) {
        delete[] global_stack_base;
    }

    clear();
}

__device__ void evm_stack_t::clear() {
    stack_offset = 0;
    // capacity = 0;
    global_stack_base = nullptr;
}

// __device__ evm_stack_t &evm_stack_t::operator=(const evm_stack_t &other) {
//     if (this != &other) {
//         free();
//         duplicate(other);
//     }
//     return *this;
// }

// __device__ void evm_stack_t::duplicate(const evm_stack_t &other) {
//     __SHARED_MEMORY__ evm_word_t *tmp_stack_base;
//     __ONE_GPU_THREAD_BEGIN__
//     tmp_stack_base = new evm_word_t[other.capacity];
//     if (tmp_stack_base != nullptr) {
//         memcpy(tmp_stack_base, other.global_stack_base, other.stack_offset * sizeof(evm_word_t));
//     }
//     __ONE_GPU_THREAD_END__
//     stack_offset = other.stack_offset;
//     capacity = other.stack_offset;
//     global_stack_base = tmp_stack_base;
// }

// TODO : reimplement
__device__ void evm_stack_t::extract_data(evm_word_t *other) const {
    // printf("Extract data stack offset %d\n", stack_offset);

    if (stack_offset + stack_base_offset < CuEVM::shared_stack_size) {
        memcpy(other, shared_stack_base, stack_offset * sizeof(evm_word_t));
    } else {
        int32_t left_over = CuEVM::shared_stack_size - stack_base_offset;
        if (left_over > 0) {
            memcpy(other, shared_stack_base, left_over * sizeof(evm_word_t));
            memcpy(other + left_over, global_stack_base, (stack_offset - left_over) * sizeof(evm_word_t));
        } else
            memcpy(other, global_stack_base, stack_offset * sizeof(evm_word_t));
    }
    // if (global_stack_base != nullptr) {
    //     memcpy(other, global_stack_base, stack_offset * sizeof(evm_word_t));
    // }
}

__device__ uint32_t evm_stack_t::size() const { return stack_offset; }

__device__ evm_word_t *evm_stack_t::top() {
    if (stack_base_offset + stack_offset < CuEVM::shared_stack_size) {
        // printf("shared stack base %p stack offset %d\n", shared_stack_base, stack_offset);
        return shared_stack_base + stack_offset;
    } else {
        // page size is max stack size
        if ((stack_base_offset + stack_offset - CuEVM::shared_stack_size) % max_stack_size == 0) {
            evm_word_t *new_stack_base = new evm_word_t[max_stack_size];
            if (global_stack_base != nullptr) {
                delete[] global_stack_base;
            }

            global_stack_base = new_stack_base;
        }
        return global_stack_base + stack_base_offset + stack_offset - CuEVM::shared_stack_size;
    }
}

__device__ int32_t evm_stack_t::push(const evm_word_t &value) {
    if (stack_offset < max_stack_size) {
        *top() = value;
        stack_offset++;
        // printf("Stack offset %d pointer %p  top %p\n", stack_offset, shared_stack_base, top());

        return ERROR_SUCCESS;
    } else {
        return ERROR_STACK_OVERFLOW;
    }
}

__device__ int32_t evm_stack_t::push_evm_word_t(const evm_word_t *value) {
    if (stack_offset < max_stack_size) {
        *top() = *value;
        stack_offset++;
        // printf("Stack offset %d pointer %p  top %p\n", stack_offset, shared_stack_base, top());

        return ERROR_SUCCESS;
    } else {
        return ERROR_STACK_OVERFLOW;
    }
}

__device__ int32_t evm_stack_t::pop(evm_word_t &y) {
    if (stack_offset == 0) return ERROR_STACK_UNDERFLOW;

    y = *get_address_at_index(1);
    stack_offset--;
    // printf("Stack pop offset %d idx %d\n", stack_offset, THREADIDX);
    // cgbn_load(arith.env, y, top());
    return ERROR_SUCCESS;
}

__device__ int32_t evm_stack_t::pop_evm_word(evm_word_t *&y) {
    if (stack_offset == 0) return ERROR_STACK_UNDERFLOW;

    y = get_address_at_index(1);
    stack_offset--;
    return ERROR_SUCCESS;
}

__device__ int32_t evm_stack_t::pushx(uint8_t x, uint8_t *src_byte_data, uint8_t src_byte_size) {
    if (stack_offset < max_stack_size) {
        evm_word_t *top_ = top();

        uint256_from_bytes(top_, src_byte_data, src_byte_size);
        stack_offset++;

        return ERROR_SUCCESS;
    } else {
        printf("pushx overflow stack offset %d\n", stack_offset);
        return ERROR_STACK_OVERFLOW;
    }
}

// The caller must check underflow
__device__ evm_word_t *evm_stack_t::get_address_at_index(uint32_t index) const {
    // printf("global stack base %p shared stack base %p\n", global_stack_base, shared_stack_base);
    if (stack_base_offset + stack_offset - index < CuEVM::shared_stack_size)  // stack_offset is after added
        return shared_stack_base + stack_offset - index;
    else
        return global_stack_base + stack_base_offset + stack_offset - index - CuEVM::shared_stack_size;
}

__device__ int32_t evm_stack_t::dupx(uint32_t x) {
    if ((stack_offset < max_stack_size) && (x <= stack_offset)) {
        // cgbn_store(arith.env, top(), value);
        *top() = *get_address_at_index(x);

        stack_offset++;
        return ERROR_SUCCESS;
    } else {
        printf(" dupx overflow or underflow stack offset %d\n", stack_offset);
        return ERROR_STACK_OVERFLOW;  // represent underflow also
    };
}

__device__ int32_t evm_stack_t::swapx(uint32_t x) {
    x++;
    if (x > stack_offset) {
        return ERROR_STACK_UNDERFLOW;
    }
    evm_word_t tmp;
    tmp = *get_address_at_index(x);
    *get_address_at_index(x) = *get_address_at_index(1);
    *get_address_at_index(1) = tmp;

    return ERROR_SUCCESS;
}

__device__ void evm_stack_t::print() {
    printf("Stack size: %d, data:\n", size());
    for (uint32_t idx = 1; idx <= size(); idx++) {
        evm_word_t *elem = get_address_at_index(idx);
        printf("idx %d, elem %p\n", idx, elem);
        elem->print();
    }
}

__host__ cJSON *evm_stack_t::to_json() {
    cJSON *json = cJSON_CreateObject();
    char *hex_string_ptr = new char[CuEVM::word_size * 2 + 3];
    cJSON *stack = cJSON_CreateArray();
    for (uint32_t idx = 0; idx < size(); idx++) {
        global_stack_base[idx].to_hex(hex_string_ptr);
        cJSON_AddItemToArray(stack, cJSON_CreateString(hex_string_ptr));
    }
    cJSON_AddItemToObject(json, "data", stack);
    delete[] hex_string_ptr;
    return json;
}
__host__ evm_stack_t *evm_stack_t::get_cpu(uint32_t count) {
    evm_stack_t *instances = new evm_stack_t[count];
    return instances;
}
__host__ void evm_stack_t::cpu_free(evm_stack_t *instances, uint32_t count) { delete[] instances; }

}  // namespace CuEVM::stack