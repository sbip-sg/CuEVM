// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2023-11-30
// SPDX-License-Identifier: MIT

#include "include/return_data.cuh"
#include "include/utils.h"

namespace cuEVM {
  __host__ __device__ EVMReturnData::EVMReturnData(
      byte_array_t *content) : _content(content) {}

  /**
   * The cosntrctuor without the content
  */
  __host__ __device__ EVMReturnData::EVMReturnData()
  {
    SHARED_MEMORY byte_array_t *tmp_content;
    ONE_THREAD_PER_INSTANCE(
      tmp_content = new byte_array_t;
      tmp_content->size = 0;
      tmp_content->data = NULL;)
    _content = tmp_content;
  }

  /**
   * The destructor
  */
  __host__ __device__ EVMReturnData::~EVMReturnData()
  {
    ONE_THREAD_PER_INSTANCE(
      if (
          (_content->size > 0) &&
          (_content->data != NULL)
      )
      {
        delete[] _content->data;
        _content->size = 0;
        _content->data = NULL;
      }
      delete _content;
    )
    _content = NULL;
  }

  /**
   * Get the size of the return data
   * @return the size of the return data
  */
  __host__ __device__ size_t EVMReturnData::size()
  {
    return _content->size;
  }

  /**
   * Get the content of the return data
   * @param[in] index the index of in the return data
   * @param[in] size the size of the content
   * @param[out] error_code the error code
   * @return the pointer in the return data
  */
  __host__ __device__ uint8_t* EVMReturnData::get(
      size_t index,
      size_t size,
      uint32_t &error_code)
  {
    size_t request_size = index + size;
    if ((request_size < index) || (request_size < size))
    {
      error_code = ERROR_RETURN_DATA_OVERFLOW;
      return _content->data;
    }
    else if (request_size > _content->size)
    {
      error_code = ERROR_RETURN_DATA_INVALID_SIZE;
      return _content->data;
    }
    else
    {
      return _content->data + index;
    }
  }

  /**
   * Get the content of the return data
   * @return the pointer in the return data
  */
  __host__ __device__ byte_array_t* EVMReturnData::get_data()
  {
    return _content;
  }

  /**
   * Set the content of the return data
   * @param[in] data the data to be set
   * @param[in] size the size of the data
  */
  __host__ __device__ void EVMReturnData::set(
      uint8_t *data,
      size_t size)
  {
    ONE_THREAD_PER_INSTANCE(
        if (_content->size > 0) {
          delete[] _content->data;
        } if (size > 0) {
          _content->data = new uint8_t[size];
          memcpy(_content->data, data, size);
        })
    _content->size = size;
  }

  __host__ __device__ void EVMReturnData::to_byte_array_t(
      byte_array_t &data_content)
  {
    ONE_THREAD_PER_INSTANCE(
        if (data_content.size > 0) {
          delete[] data_content.data;
          data_content.data = NULL;
          data_content.size = 0;
        }
        if (_content->size > 0) {
          data_content.data = new uint8_t[_content->size];
          memcpy(data_content.data, _content->data, _content->size);
        } else {
          data_content.data = NULL;
        })
    data_content.size = _content->size;
  }

  /**
   * Print the return data
  */
  __host__ __device__ void EVMReturnData::print()
  {
    byte_array::print_byte_array_t(*_content);
  }

  /**
   * Get the json representation of the return data
   * @return the json representation of the return data
  */
  __host__ cJSON* EVMReturnData::json()
  {
    return byte_array::json_from_byte_array_t(*_content);
  }
}