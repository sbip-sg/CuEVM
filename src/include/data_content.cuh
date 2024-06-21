// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2023-11-30
// SPDX-License-Identifier: MIT

#ifndef _DATA_CONTENT_H_
#define _DATA_CONTENT_H_

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda.h>
#include <cjson/cJSON.h>

/**
 * The data content structure.
 * It has the size of the data and a pointer to the data.
*/
typedef struct
{
  size_t size;
  uint8_t *data;
} data_content_t;

/**
 * Get the hex string from a byte array.
 * The hex string is allocated on the heap and needs to be freed.
 * @param[in] bytes The byte array.
 * @param[in] count The number of bytes.
 * @return The hex string.
 */
__host__ char *hex_from_bytes(uint8_t *bytes, size_t count);

/**
 * Print a byte array.
 * @param[in] bytes The byte array.
 * @param[in] count The number of bytes.
*/
__host__ __device__ void print_bytes(uint8_t *bytes, size_t count);

/**
 * Print the data content.
 * @param[in] data_content The data content.
*/
__host__ __device__ void print_data_content_t(data_content_t &data_content);

/**
 * Get the hex string from a data content.
 * The hex string is allocated on the heap and needs to be freed.
 * @param[in] data_content The data content.
 * @return The hex string.
*/
__host__ char *hex_from_data_content(data_content_t &data_content);
/**
 * Get the json object from a data content.
 * @param[in] data_content The data content.
 * @return The json object.
 */
__host__ cJSON *json_from_data_content_t(data_content_t &data_content);


#endif