
#include <CuEVM/core/stack.cuh>
// avoid circular dependency
#include <CuEVM/core/memory_pool.cuh>
#include <CuEVM/utils/error_codes.cuh>
namespace CuEVM::stack {

__device__ evm_stack_t::~evm_stack_t() { free(); }

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

// TODO: reimplement
__device__ void evm_stack_t::extract_data(evm_word_t *other) const {
    if (stack_offset + stack_base_offset < memory_pool_stack_preallocate) {
        memcpy(other, shared_stack_base, stack_offset * sizeof(evm_word_t));
    } else {
        int32_t left_over = memory_pool_stack_preallocate - stack_base_offset;
        if (left_over > 0) {
            memcpy(other, shared_stack_base, left_over * sizeof(evm_word_t));
            memcpy(other + left_over, global_stack_base, (stack_offset - left_over) * sizeof(evm_word_t));
        } else
            memcpy(other, global_stack_base, stack_offset * sizeof(evm_word_t));
    }
}

__device__ uint32_t evm_stack_t::size() const { return stack_offset; }

__device__ void evm_stack_t::reduce_size(uint32_t num_items) {
    if (stack_offset >= num_items) {
        stack_offset -= num_items;
    }
}

__device__ evm_word_t *evm_stack_t::top() {
    if (stack_base_offset + stack_offset < memory_pool_stack_preallocate) {
        // if (THREADIDX == 0) {
        //     printf("shared stack base %p stack offset %d, num instances %d instance idx %d \n", shared_stack_base,
        //            stack_offset, CuEVM::memory_pool::global_memory_pool->num_instances, INSTANCE_GLOBAL_IDX);
        //     printf("real address %p\n", shared_stack_base +
        //                                     (stack_offset - 1) *
        //                                     CuEVM::memory_pool::global_memory_pool->num_instances +
        //                                     INSTANCE_GLOBAL_IDX);
        // }
        // return shared_stack_base + stack_offset;
        return shared_stack_base + stack_offset * CuEVM::memory_pool::global_memory_pool->num_instances +
               INSTANCE_GLOBAL_IDX;
    } else {
        // page size is max stack size
        // if ((stack_base_offset + stack_offset - memory_pool_stack_preallocate) % max_stack_size == 0) {
        //     evm_word_t *new_stack_base = new evm_word_t[max_stack_size];
        //     if (global_stack_base != nullptr) {
        //         delete[] global_stack_base;
        //     }

        //     global_stack_base = new_stack_base;
        // }
        // allocate once
        if (global_stack_base == nullptr) {
            global_stack_base = new evm_word_t[max_stack_size];
        }  // TODO reuse global stack allocation for child calls
        return global_stack_base + stack_base_offset + stack_offset - memory_pool_stack_preallocate;
    }
}
__device__ int32_t evm_stack_t::push_uint32(uint32_t value) {
    if (stack_offset < max_stack_size) {
        *top() = value;
        stack_offset++;
        return ERROR_SUCCESS;
    } else {
        return ERROR_STACK_OVERFLOW;
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
        // if (THREADIDX == 0) printf("pushx top %p, thread %d\n", top_, INSTANCE_GLOBAL_IDX);
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
    if (stack_base_offset + stack_offset - index < memory_pool_stack_preallocate)  // stack_offset is after added
        return shared_stack_base + (stack_offset - index) * CuEVM::memory_pool::global_memory_pool->num_instances +
               INSTANCE_GLOBAL_IDX;
    else
        return global_stack_base + stack_base_offset + stack_offset - index - memory_pool_stack_preallocate;
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

}  // namespace CuEVM::stack
