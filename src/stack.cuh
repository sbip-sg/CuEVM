// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2023-11-30
// SPDX-License-Identifier: MIT

#ifndef _STACK_H_
#define _STACK_H_

#include "include/utils.h"

/**
 * The kernel to copy the stack data structures.
 * @param[out] dst The destination stack data structure
 * @param[in] src The source stack data structure
 * @param[in] count The number of instances
*/
template <typename T, typename E>
__global__ void kernel_stacks(
    T *dst,
    T *src,
    uint32_t count);

/**
 * The stack class (YP: \f$\mu_{s}\f$)
*/
class stack_t{
public:

  /**
   * The size of the stack.
   */
  static const uint32_t STACK_SIZE = evm_params::STACK_SIZE;

  /**
   * The stack data structure.
   */
  typedef struct
  {
    evm_word_t *stack_base; /**< The stack YP: (YP: \f$\mu_{s}\f$)*/
    uint32_t stack_offset; /**< The stack offset (YP: \f$|\mu_{s}|\f$)*/
  } stack_data_t;

  stack_data_t *_content; /**< The conent of the stack*/
  arith_t _arith; /**< The arithmetical environment*/

  /**
   * The constructor of the stack given the arithmetical environment
   * and the stack data structure.
   * @param[in] arith The arithmetical environment
   * @param[in] content The stack data structure
  */
  __host__ __device__ __forceinline__ stack_t(
      arith_t arith,
      stack_data_t *content) : _arith(arith),
                               _content(content)
  {
  }

  /**
   * The constructor of the stack given the arithmetical environment.
   * @param[in] arith The arithmetical environment
  */
  __host__ __device__ __forceinline__ stack_t(
      arith_t arith) : _arith(arith)
  {
    SHARED_MEMORY stack_data_t *content;
    ONE_THREAD_PER_INSTANCE(
        content = new stack_data_t;
        content->stack_base = new evm_word_t[STACK_SIZE];
        content->stack_offset = 0;)
    _content = content;
  }

  /**
   * The destructor of the stack.
  */
  __host__ __device__ __forceinline__ ~stack_t()
  {
    ONE_THREAD_PER_INSTANCE(
        if (_content->stack_base != NULL) {
          delete[] _content->stack_base;
          _content->stack_base = NULL;
          _content->stack_offset = 0;
        } if (_content != NULL) {
          delete _content;
        })
    _content = NULL;
  }

  /**
   * Get the size of the stack.
   * @return The size of the stack
  */
  __host__ __device__ __forceinline__ uint32_t size()
  {
    return _content->stack_offset;
  }

  /**
   * Get the top of the stack.
   * @return The top of the stack pointer
  */
  __host__ __device__ __forceinline__ evm_word_t *top()
  {
    return _content->stack_base + _content->stack_offset;
  }

  /**
   * Push a value on the stack.
   * @param[in] value The value to be pushed on the stack
   * @param[out] error_code The error code
  */
  __host__ __device__ __forceinline__ void push(const bn_t &value, uint32_t &error_code)
  {
    if (size() >= STACK_SIZE)
    {
      error_code = ERR_STACK_OVERFLOW;
      return;
    }
    cgbn_store(_arith._env, top(), value);
    _content->stack_offset++;
  }

  /**
   * Pop a value from the stack.
   * @param[out] y The value popped from the stack
   * @param[out] error_code The error code
  */
  __host__ __device__ __forceinline__ void pop(bn_t &y, uint32_t &error_code)
  {
    if (size() == 0)
    {
      error_code = ERR_STACK_UNDERFLOW;
      cgbn_set_ui32(_arith._env, y, 0);
      return;
    }
    _content->stack_offset--;
    cgbn_load(_arith._env, y, top());
  }

  /**
   * Push a value on the stack given by a byte array.
   * If the size of the byte array is smaller than x,
   * the value is padded with zeros for the least
   * significant bytes.
   * @param[in] x The number of bytes to be pushed on the stack
   * @param[out] error_code The error code
   * @param[in] src_byte_data The byte array
   * @param[in] src_byte_size The size of the byte array
  */
  __host__ __device__ __forceinline__ void pushx(
      uint8_t x,
      uint32_t &error_code,
      uint8_t *src_byte_data,
      uint8_t src_byte_size)
  {
    if (x > 32)
    {
      error_code = ERROR_STACK_INVALID_PUSHX_X;
      return;
    }
    bn_t r;
    cgbn_set_ui32(_arith._env, r, 0);
    for (uint8_t idx = (x - src_byte_size); idx < x; idx++)
    {
      cgbn_insert_bits_ui32(
          _arith._env,
          r,
          r,
          idx * 8,
          8,
          src_byte_data[x - 1 - idx]);
    }
    push(r, error_code);
  }

  /**
   * Get the pointer to the stack element at a given index from the top.
   * @param[in] index The index from the top of the stack
   * @param[out] error_code The error code
   * @return The pointer to the stack element
  */
  __host__ __device__ __forceinline__ evm_word_t *get_index(
    uint32_t index,
    uint32_t &error_code
  )
  {
    if (size() < index)
    {
      error_code = ERR_STACK_UNDERFLOW;
      return NULL;
    }
    return _content->stack_base + (size() - index);
  }

  /**
   * Duplicate the stack element at a given index from the top,
   * and push it on the stack.
   * @param[in] index The index from the top of the stack
   * @param[out] error_code The error code
  */
  __host__ __device__ __forceinline__ void dupx(
      uint8_t x,
      uint32_t &error_code)
  {
    if ((x < 1) || (x > 16))
    {
      error_code = ERROR_STACK_INVALID_DUPX_X;
      return;
    }
    bn_t r;
    evm_word_t *value = get_index(x, error_code);
    if (value == NULL)
    {
      return;
    }
    cgbn_load(_arith._env, r, value);
    push(r, error_code);
  }

  /**
   * Swap the stack element at a given index from the top with the top element.
   * @param[in] index The index from the top of the stack
   * @param[out] error_code The error code
  */
  __host__ __device__ __forceinline__ void swapx(
    uint32_t index,
    uint32_t &error_code)
  {
    if ((index < 1) || (index > 16))
    {
      error_code = ERR_STACK_INVALID_SIZE;
      return;
    }
    bn_t a, b;
    evm_word_t *value_a = get_index(1, error_code);
    evm_word_t *value_b = get_index(index + 1, error_code);
    if ((value_a == NULL) || (value_b == NULL))
    {
      return;
    }
    cgbn_load(_arith._env, a, value_b);
    cgbn_load(_arith._env, b, value_a);
    cgbn_store(_arith._env, value_a, a);
    cgbn_store(_arith._env, value_b, b);
  }

  /**
   * Copy the stack to a stack data structure.
   * @param[out] dst The stack data structure
  */
  __host__ __device__ __forceinline__ void to_stack_data_t(
      stack_data_t &dst)
  {
    ONE_THREAD_PER_INSTANCE(
        if (
            (dst.stack_offset > 0) &&
            (dst.stack_base != NULL)) {
          delete[] dst.stack_base;
          dst.stack_base = NULL;
        } dst.stack_offset = _content->stack_offset;
        if (dst.stack_offset == 0) {
          dst.stack_base = NULL;
        } else {
          dst.stack_base = new evm_word_t[dst.stack_offset];
          memcpy(
              dst.stack_base,
              _content->stack_base,
              sizeof(evm_word_t) * dst.stack_offset);
        })
  }

  /**
   * Print the stack data structure.
   * @param[in] arith The arithmetical environment
   * @param[in] stack_data The stack data structure
  */
  __host__ __device__ __forceinline__ static void print_stack_data_t(
      arith_t &arith,
      stack_data_t &stack_data)
  {
    printf("Stack size: %d, data:\n", stack_data.stack_offset);
    for (uint32_t idx = 0; idx < stack_data.stack_offset; idx++)
    {
      arith.print_cgbn_memory(stack_data.stack_base[idx]);
    }
  }

  /**
   * Print the stack.
   * @param[in] full If false, prints only active stack elements, otherwise prints the entire stack
  */
  __host__ __device__ void print(
      bool full = false)
  {
    printf("Stack size: %d, data:\n", size());
    uint32_t print_size = full ? STACK_SIZE : size();
    for (uint32_t idx = 0; idx < print_size; idx++)
    {
      _arith.print_cgbn_memory(_content->stack_base[idx]);
    }
  }

  /**
   * Generate a JSON object from a stack data structure.
   * @param[in] arith The arithmetical environment
   * @param[in] stack_data The stack data structure
   * @return The JSON object
  */
  __host__ static cJSON *json_from_stack_data_t(
      arith_t &arith,
      stack_data_t &stack_data)
  {
    char *hex_string_ptr = new char[arith_t::BYTES * 2 + 3];
    cJSON *stack_json = cJSON_CreateObject();

    cJSON *stack_data_json = cJSON_CreateArray();
    for (uint32_t idx = 0; idx < stack_data.stack_offset; idx++)
    {
      arith.hex_string_from_cgbn_memory(hex_string_ptr, stack_data.stack_base[idx]);
      cJSON_AddItemToArray(stack_data_json, cJSON_CreateString(hex_string_ptr));
    }
    cJSON_AddItemToObject(stack_json, "data", stack_data_json);
    delete[] hex_string_ptr;
    return stack_json;
  }

  /**
   * Generate a Python dictionary from a stack data structure.
   * @param[in] arith The arithmetical environment
   * @param[in] stack_data The stack data structure
   * @return The Python dictionary
  */
  __host__ static PyObject* pyobject_from_stack_data_t(arith_t &arith, stack_data_t &stack_data) {
    char* hex_string_ptr = new char[arith_t::BYTES * 2 + 3];
    PyObject* stack_json = PyDict_New();

    PyObject* stack_data_json = PyList_New(0);
    for (uint32_t idx = 0; idx < stack_data.stack_offset; idx++) {
        // Convert stack data to hex string
        arith.hex_string_from_cgbn_memory(hex_string_ptr, stack_data.stack_base[idx]);
        // Create a Python string from the hex string
        PyObject* stack_item = PyUnicode_FromString(hex_string_ptr);

        // Add the Python string to the stack data list
        PyList_Append(stack_data_json, stack_item);

        // Decrement the reference count of stack_item since PyList_Append increases it
        Py_DECREF(stack_item);
    }

    // Add the stack data list to the stack dictionary under the key "data"
    PyDict_SetItemString(stack_json, "data", stack_data_json);

    // Decrement the reference count of stack_data_json since PyDict_SetItemString increases it
    Py_DECREF(stack_data_json);

    delete[] hex_string_ptr;
    return stack_json;
  }

  /**
   * Generate a JSON object from the stack.
   * @param[in] full If false, prints only active stack elements, otherwise prints the entire stack
  */
  __host__ cJSON *json(bool full = false)
  {
    char *hex_string_ptr = new char[arith_t::BYTES * 2 + 3];
    cJSON *stack_json = cJSON_CreateObject();

    cJSON *stack_data_json = cJSON_CreateArray();
    uint32_t print_size = full ? STACK_SIZE : size();
    for (uint32_t idx = 0; idx < print_size; idx++)
    {
      _arith.hex_string_from_cgbn_memory(hex_string_ptr, _content->stack_base[idx]);
      cJSON_AddItemToArray(stack_data_json, cJSON_CreateString(hex_string_ptr));
    }
    cJSON_AddItemToObject(stack_json, "data", stack_data_json);
    delete[] hex_string_ptr;
    return stack_json;
  }

  /**
   * Generate the cpu data structures for the stack.
   * @param[in] count The number of instances
   * @return The cpu data structures
  */
  __host__ static stack_data_t *get_cpu_instances(
      uint32_t count)
  {
    stack_data_t *cpu_instances = new stack_data_t[count];
    for (uint32_t idx = 0; idx < count; idx++)
    {
      cpu_instances[idx].stack_base = NULL;
      cpu_instances[idx].stack_offset = 0;
    }
    return cpu_instances;
  }

  /**
   * Free the cpu data structures for the stack.
   * @param[in] cpu_instances The cpu data structures
   * @param[in] count The number of instances
  */
  __host__ static void free_cpu_instances(
      stack_data_t *cpu_instances,
      uint32_t count)
  {
    for (int index = 0; index < count; index++)
    {
      printf("freeing index %d\n", index);
      if (cpu_instances[index].stack_base != NULL)
      {
        delete[] cpu_instances[index].stack_base;
        cpu_instances[index].stack_base = NULL;
      }
      cpu_instances[index].stack_offset = 0;
    }
    delete[] cpu_instances;
  }

  /**
   * Generate the gpu data structures for the stack.
   * @param[in] cpu_instances The cpu data structures
   * @param[in] count The number of instances
   * @return The gpu data structures
  */
  __host__ static stack_data_t *get_gpu_instances_from_cpu_instances(
      stack_data_t *cpu_instances,
      uint32_t count)
  {
    stack_data_t *gpu_instances, *tmp_cpu_instances;
    tmp_cpu_instances = new stack_data_t[count];
    memcpy(
        tmp_cpu_instances,
        cpu_instances,
        sizeof(stack_data_t) * count);
    for (uint32_t idx = 0; idx < count; idx++)
    {
      if (cpu_instances[idx].stack_offset > 0)
      {
        CUDA_CHECK(cudaMalloc(
            (void **)&tmp_cpu_instances[idx].stack_base,
            sizeof(evm_word_t) * cpu_instances[idx].stack_offset));
        CUDA_CHECK(cudaMemcpy(
            tmp_cpu_instances[idx].stack_base,
            cpu_instances[idx].stack_base,
            sizeof(evm_word_t) * cpu_instances[idx].stack_offset,
            cudaMemcpyHostToDevice));
      }
      else
      {
        tmp_cpu_instances[idx].stack_base = NULL;
      }
    }
    CUDA_CHECK(cudaMalloc(
        (void **)&gpu_instances,
        sizeof(stack_data_t) * count));
    CUDA_CHECK(cudaMemcpy(
        gpu_instances,
        tmp_cpu_instances,
        sizeof(stack_data_t) * count,
        cudaMemcpyHostToDevice));
    delete[] tmp_cpu_instances;
    tmp_cpu_instances = NULL;
    return gpu_instances;
  }

  /**
   * Free the gpu data structures for the stack.
   * @param[in] gpu_instances The gpu data structures
   * @param[in] count The number of instances
  */
  __host__ static void free_gpu_instances(
      stack_data_t *gpu_instances,
      uint32_t count)
  {
    stack_data_t *tmp_cpu_instances;
    tmp_cpu_instances = new stack_data_t[count];
    CUDA_CHECK(cudaMemcpy(
        tmp_cpu_instances,
        gpu_instances,
        sizeof(stack_data_t) * count,
        cudaMemcpyDeviceToHost));
    for (uint32_t idx = 0; idx < count; idx++)
    {
      if (tmp_cpu_instances[idx].stack_base != NULL)
        CUDA_CHECK(cudaFree(tmp_cpu_instances[idx].stack_base));
    }
    delete[] tmp_cpu_instances;
    CUDA_CHECK(cudaFree(gpu_instances));
  }

  /**
   * Generate the cpu data structures for the stack from the gpu data structures.
   * @param[in] gpu_instances The gpu data structures
   * @param[in] count The number of instances
   * @return The cpu data structures
  */
  __host__ static stack_data_t *get_cpu_instances_from_gpu_instances(
      stack_data_t *gpu_instances,
      uint32_t count)
  {
    stack_data_t *cpu_instances, *tmp_cpu_instances, *tmp_gpu_instances;
    cpu_instances = new stack_data_t[count];
    tmp_cpu_instances = new stack_data_t[count];
    CUDA_CHECK(cudaMemcpy(
        cpu_instances,
        gpu_instances,
        sizeof(stack_data_t) * count,
        cudaMemcpyDeviceToHost));
    memcpy(
        tmp_cpu_instances,
        cpu_instances,
        sizeof(stack_data_t) * count);
    for (uint32_t idx = 0; idx < count; idx++)
    {
      tmp_cpu_instances[idx].stack_offset = cpu_instances[idx].stack_offset;
      if (cpu_instances[idx].stack_offset > 0)
      {
        CUDA_CHECK(cudaMalloc(
            (void **)&tmp_cpu_instances[idx].stack_base,
            sizeof(evm_word_t) * cpu_instances[idx].stack_offset));
      }
      else
      {
        tmp_cpu_instances[idx].stack_base = NULL;
      }
    }
    CUDA_CHECK(cudaMalloc(
        (void **)&tmp_gpu_instances,
        sizeof(stack_data_t) * count));
    CUDA_CHECK(cudaMemcpy(
        tmp_gpu_instances,
        tmp_cpu_instances,
        sizeof(stack_data_t) * count,
        cudaMemcpyHostToDevice));
    delete[] tmp_cpu_instances;
    tmp_cpu_instances = NULL;
    kernel_stacks<stack_data_t,  evm_word_t><<<count, 1>>>(
        tmp_gpu_instances,
        gpu_instances,
        count);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(gpu_instances));
    gpu_instances = tmp_gpu_instances;

    CUDA_CHECK(cudaMemcpy(
        cpu_instances,
        gpu_instances,
        sizeof(stack_data_t) * count,
        cudaMemcpyDeviceToHost));
    tmp_cpu_instances = new stack_data_t[count];
    memcpy(
        tmp_cpu_instances,
        cpu_instances,
        sizeof(stack_data_t) * count);

    for (uint32_t idx = 0; idx < count; idx++)
    {
      tmp_cpu_instances[idx].stack_offset = cpu_instances[idx].stack_offset;
      if (cpu_instances[idx].stack_offset > 0)
      {
        tmp_cpu_instances[idx].stack_base = new evm_word_t[cpu_instances[idx].stack_offset];
        CUDA_CHECK(cudaMemcpy(
            tmp_cpu_instances[idx].stack_base,
            cpu_instances[idx].stack_base,
            sizeof(evm_word_t) * cpu_instances[idx].stack_offset,
            cudaMemcpyDeviceToHost));
      }
      else
      {
        tmp_cpu_instances[idx].stack_base = NULL;
      }
    }

    memcpy(
        cpu_instances,
        tmp_cpu_instances,
        sizeof(stack_data_t) * count);
    delete[] tmp_cpu_instances;
    tmp_cpu_instances = NULL;
    free_gpu_instances(gpu_instances, count);
    return cpu_instances;
  }
};

template <typename T, typename E>
__global__ void kernel_stacks(
    T *dst,
    T *src,
    uint32_t count)
{
  uint32_t instance = blockIdx.x * blockDim.x + threadIdx.x;
  if (instance >= count)
  {
    return;
  }
  dst[instance].stack_offset = src[instance].stack_offset;
  if (dst[instance].stack_offset > 0)
  {
    memcpy(
        dst[instance].stack_base,
        src[instance].stack_base,
        sizeof(E) * src[instance].stack_offset);
    delete[] src[instance].stack_base;
    src[instance].stack_base = NULL;
    src[instance].stack_offset = 0;
  }
}

#endif