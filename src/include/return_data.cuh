// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2023-11-30
// SPDX-License-Identifier: MIT

#ifndef _RETURN_DATA_H_
#define _RETURN_DATA_H_

#include "byte_array.cuh"
#include <cuda.h>
#include <cjson/cJSON.h>
#include <stdint.h>
#include <stdlib.h>

namespace cuEVM {
    /**
     * The return data class. (YP: \f$H_{return}(\mu)=H(\mu, I)\f$)
    */
    class EVMReturnData
    {
    public:
      byte_array_t *_content; /**< The content of the return data*/
      /**
       * The constructor with the given content
       * @param[in] content the content of the return data
      */
      __host__ __device__ EVMReturnData(
          byte_array_t *content);

      /**
       * The cosntrctuor without the content
      */
      __host__ __device__ EVMReturnData();

      /**
       * The destructor
      */
      __host__ __device__ ~EVMReturnData();

      /**
       * Get the size of the return data
       * @return the size of the return data
      */
      __host__ __device__ size_t size();

      /**
       * Get the content of the return data
       * @param[in] index the index of in the return data
       * @param[in] size the size of the content
       * @param[out] error_code the error code
       * @return the pointer in the return data
      */
      __host__ __device__ uint8_t *get(
          size_t index,
          size_t size,
          uint32_t &error_code);

      /**
       * Get the content of the return data
       * @return the pointer in the return data
      */
      __host__ __device__ byte_array_t *get_data();

      /**
       * Set the content of the return data
       * @param[in] data the data to be set
       * @param[in] size the size of the data
      */
      __host__ __device__ void set(
          uint8_t *data,
          size_t size);

      __host__ __device__ void to_byte_array_t(
          byte_array_t &data_content);

      /**
       * Print the return data
      */
      __host__ __device__ void print();

      /**
       * Get the json representation of the return data
       * @return the json representation of the return data
      */
      __host__ cJSON *json();
    };
}

#endif