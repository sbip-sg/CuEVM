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
__host__ char *hex_from_bytes(uint8_t *bytes, size_t count) {
  char *hex_string = new char[count*2+1];
  char *return_string = new char[count*2+1+2];
  for(size_t idx=0; idx<count; idx++)
    sprintf(&hex_string[idx*2], "%02x", bytes[idx]);
  hex_string[count*2]=0;
  strcpy(return_string + 2, hex_string);
  delete[] hex_string;
  hex_string = NULL;
  return_string[0]='0';
  return_string[1]='x';
  return return_string;
}

/**
 * Print a byte array.
 * @param[in] bytes The byte array.
 * @param[in] count The number of bytes.
*/
__host__ __device__ __forceinline__ void print_bytes(uint8_t *bytes, size_t count) {
  printf("data: ");
  for(size_t idx=0; idx<count; idx++)
    printf("%02x", bytes[idx]);
  printf("\n");
}

/**
 * Print the data content.
 * @param[in] data_content The data content.
*/
__host__ __device__ __forceinline__ void print_data_content(data_content_t &data_content) {
  printf("size: %lu\n", data_content.size);
  print_bytes(data_content.data, data_content.size);
}

/**
 * Get the hex string from a data content.
 * The hex string is allocated on the heap and needs to be freed.
 * @param[in] data_content The data content.
 * @return The hex string.
*/
__host__ char *hex_from_data_content(data_content_t &data_content) {
  return hex_from_bytes(data_content.data, data_content.size);
}

/**
 * Get the json object from a data content.
 * @param[in] data_content The data content.
 * @return The json object.
 */
__host__ __forceinline__ cJSON *json_from_data_content(data_content_t &data_content) {
  cJSON *data_json = cJSON_CreateObject();
  char *hex_string;
  //cJSON_AddNumberToObject(json, "size", data_content.size);
  if (data_content.size > 0)
  {
    hex_string = hex_from_data_content(data_content);
    cJSON_AddStringToObject(data_json, "data", hex_string);
    delete[] hex_string;
  } else {
    cJSON_AddStringToObject(data_json, "data", "0x");
  }
  return data_json;
}


#endif