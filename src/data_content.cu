// cuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation Programme)
// Author: Stefan-Dan Ciocirlan
// Data: 2024-06-20
// SPDX-License-Identifier: MIT

#include "include/data_content.cuh"

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

__host__ __device__ void print_bytes(uint8_t *bytes, size_t count) {
  printf("data: ");
  for(size_t idx=0; idx<count; idx++)
    printf("%02x", bytes[idx]);
  printf("\n");
}

__host__ __device__ void print_data_content_t(data_content_t &data_content) {
  printf("size: %lu\n", data_content.size);
  print_bytes(data_content.data, data_content.size);
}

__host__ char *hex_from_data_content(data_content_t &data_content) {
  return hex_from_bytes(data_content.data, data_content.size);
}

__host__ cJSON *json_from_data_content_t(data_content_t &data_content) {
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
