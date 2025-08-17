
#include <CuEVM/core/evm_word.cuh>
#include <CuEVM/utils/cuda_utils.cuh>
#include <CuEVM/utils/evm_defines.cuh>
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
    for (uint32_t idx = 0; idx < stack_offset; idx++) {
        other[stack_offset - 1 - idx] = *get_address_at_index(idx + 1);
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
        return shared_stack_base + stack_offset * CuEVM::memory_pool::global_memory_pool->num_instances +
               INSTANCE_GLOBAL_IDX;
    } else {
        if (global_stack_base == nullptr) {
#ifdef DEBUG_PERF
            printf("stack content dynamic allocation size %d\n", stack_base_offset + stack_offset);
#endif
            global_stack_base = new evm_word_t[max_stack_size];
            if (stack_base_offset > memory_pool_stack_preallocate) {
                stack_base_offset = memory_pool_stack_preallocate;
            }

        }  // TODO reuse global stack allocation for child calls
        if (stack_base_offset <= memory_pool_stack_preallocate) {
            return global_stack_base + stack_base_offset + stack_offset - memory_pool_stack_preallocate;
        } else {
            return global_stack_base + stack_offset;
        }
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

__device__ int32_t evm_stack_t::push_uint64(uint64_t value) {
    if (stack_offset < max_stack_size) {
        evm_word_t *dest = top();
        dest->words[0] = value;
        dest->words[1] = value >> 32;
        for (uint8_t i = 2; i < UINT256_WORDS; i++) {
            dest->words[i] = 0;
        }
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

__device__ int32_t evm_stack_t::pushx(uint8_t x, const uint8_t __restrict__ *src_byte_data, uint8_t src_byte_size) {
    if (stack_offset < max_stack_size) {
        uint256_from_bytes(top(), src_byte_data, src_byte_size);
        stack_offset++;

        return ERROR_SUCCESS;
    } else {
        printf("pushx overflow stack offset %d\n", stack_offset);
        return ERROR_STACK_OVERFLOW;
    }
}

// The caller must check underflow
__device__ evm_word_t *evm_stack_t::get_address_at_index(uint32_t index) const {
    if (stack_base_offset + stack_offset - index < memory_pool_stack_preallocate)  // stack_offset is after added
        return shared_stack_base + (stack_offset - index) * CuEVM::memory_pool::global_memory_pool->num_instances +
               INSTANCE_GLOBAL_IDX;
    else {
        if (stack_base_offset <= memory_pool_stack_preallocate) {
            return global_stack_base + stack_base_offset + stack_offset - index - memory_pool_stack_preallocate;
        } else {
            return global_stack_base + stack_offset - index;
        }
    }
    // return global_stack_base + stack_base_offset + stack_offset - index - memory_pool_stack_preallocate;
}

__device__ int32_t evm_stack_t::dupx(uint32_t x) {
    if ((stack_offset < max_stack_size) && (x <= stack_offset)) {
        // cgbn_store(arith.env, top(), value);
        *top() = *get_address_at_index(x);

        stack_offset++;
        return ERROR_SUCCESS;
    } else {
        // TODO: check this
        // printf("THREAD %d dupx overflow or underflow stack offset %d\n", INSTANCE_GLOBAL_IDX, stack_offset);
        return ERROR_STACK_OVERFLOW;  // represent underflow also
    };
}

__device__ int32_t evm_stack_t::swapx(uint32_t x) {
    x++;
    if (x > stack_offset) {
        // printf("THREAD %d swap overflow or underflow stack offset %d\n", INSTANCE_GLOBAL_IDX, stack_offset);
        return ERROR_STACK_UNDERFLOW;
    }
    evm_word_t tmp;
    tmp = *get_address_at_index(x);
    *get_address_at_index(x) = *get_address_at_index(1);
    *get_address_at_index(1) = tmp;

    return ERROR_SUCCESS;
}

__device__ void evm_stack_t::print() const {
    printf("Stack size: %d, top 5 data:\n", size());
    for (uint32_t idx = 1; idx <= min(size(), 5); idx++) {
        evm_word_t *elem = get_address_at_index(idx);
        printf("idx %d, elem %p\n", idx, elem);
        elem->print();
    }
}

}  // namespace CuEVM::stack
