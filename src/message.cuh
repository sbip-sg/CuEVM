#ifndef _GPU_MESSAGE_H_
#define _GPU_MESSAGE_H_

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda.h>
#include <gmp.h>
#ifndef __CGBN_H__
#define __CGBN_H__
#include <cgbn/cgbn.h>
#endif
#include "contract.cuh"
#include "utils.h"

template<class params>
struct tx_t {
  cgbn_mem_t<params::BITS> origin;
  cgbn_mem_t<params::BITS> gasprice;
};

template<class params>
struct message_data_t {
  size_t size;
  uint8_t *data;
};

template<class params>
struct message_t {
  cgbn_mem_t<params::BITS> caller;
  cgbn_mem_t<params::BITS> value;
  cgbn_mem_t<params::BITS> to;
  cgbn_mem_t<params::BITS> nonce;
  tx_t<params> tx;
  cgbn_mem_t<params::BITS> gas;
  uint32_t depth;
  uint32_t call_type;
  message_data_t<params> data;
  typename state_t<params>::contract_t *contract;
};

template<class params>
__host__ message_t<params> *messages_from_json(cJSON *test, size_t &count) {
  const cJSON *messages_json = cJSON_GetObjectItemCaseSensitive(test, "transaction");
  message_t<params> *cpu_messages = NULL;
  mpz_t caller, value, nonce, to, tx_origin, tx_gasprice, gas;
  char *hex_string=NULL;
  mpz_init(caller);
  mpz_init(value);
  mpz_init(to);
  mpz_init(nonce);
  mpz_init(tx_origin);
  mpz_init(tx_gasprice);
  mpz_init(gas);
  const cJSON *data_json = cJSON_GetObjectItemCaseSensitive(messages_json, "data");
  size_t data_counts = cJSON_GetArraySize(data_json);

  const cJSON *gas_limit_json = cJSON_GetObjectItemCaseSensitive(messages_json, "gasLimit");
  size_t gas_limit_counts = cJSON_GetArraySize(gas_limit_json);

  const cJSON *gas_price_json = cJSON_GetObjectItemCaseSensitive(messages_json, "gasPrice");
  hex_string = gas_price_json->valuestring;
  adjusted_length(&hex_string);
  mpz_set_str(tx_gasprice, hex_string, 16);

  const cJSON *nonce_json = cJSON_GetObjectItemCaseSensitive(messages_json, "nonce");
  hex_string = nonce_json->valuestring;
  adjusted_length(&hex_string);
  mpz_set_str(nonce, hex_string, 16);

  const cJSON *to_json = cJSON_GetObjectItemCaseSensitive(messages_json, "to");
  hex_string = to_json->valuestring;
  adjusted_length(&hex_string);
  mpz_set_str(to, hex_string, 16);


  const cJSON *value_json = cJSON_GetObjectItemCaseSensitive(messages_json, "value");
  size_t value_counts = cJSON_GetArraySize(value_json);

  const cJSON *caller_json = cJSON_GetObjectItemCaseSensitive(messages_json, "sender");
  hex_string = caller_json->valuestring;
  adjusted_length(&hex_string);
  mpz_set_str(caller, hex_string, 16);
  mpz_set_str(tx_origin, hex_string, 16);

  count = data_counts * gas_limit_counts * value_counts;
  cpu_messages = (message_t<params> *)malloc(sizeof(message_t<params>)*count);
  size_t idx=0, jdx=0, kdx=0, instance_idx=0;
  size_t data_size;
  uint8_t *data_content;

  for(idx=0; idx<data_counts; idx++) {
    hex_string = cJSON_GetArrayItem(data_json, idx)->valuestring;
    data_size = adjusted_length(&hex_string);
    data_content = (uint8_t *) malloc (data_size);
    hex_to_bytes(hex_string, data_content);

    for(jdx=0; jdx<gas_limit_counts; jdx++) {
      hex_string = cJSON_GetArrayItem(gas_limit_json, jdx)->valuestring;
      adjusted_length(&hex_string);
      mpz_set_str(gas, hex_string, 16);

      for(kdx=0; kdx<value_counts; kdx++) {
        hex_string = cJSON_GetArrayItem(value_json, kdx)->valuestring;
        adjusted_length(&hex_string);
        mpz_set_str(value, hex_string, 16);

        cpu_messages[instance_idx].data.size = data_size;
        cpu_messages[instance_idx].data.data = (uint8_t *) malloc (data_size);
        memcpy(cpu_messages[instance_idx].data.data, data_content, data_size);

        from_mpz(cpu_messages[instance_idx].caller._limbs, params::BITS/32, caller);
        from_mpz(cpu_messages[instance_idx].value._limbs, params::BITS/32, value);
        from_mpz(cpu_messages[instance_idx].to._limbs, params::BITS/32, to);
        from_mpz(cpu_messages[instance_idx].nonce._limbs, params::BITS/32, nonce);
        from_mpz(cpu_messages[instance_idx].tx.origin._limbs, params::BITS/32, tx_origin);
        from_mpz(cpu_messages[instance_idx].tx.gasprice._limbs, params::BITS/32, tx_gasprice);
        from_mpz(cpu_messages[instance_idx].gas._limbs, params::BITS/32, gas);
        cpu_messages[instance_idx].depth=0;
        cpu_messages[instance_idx].call_type=0;
        cpu_messages[instance_idx].contract = NULL;
        instance_idx++;
      }
    }
    free(data_content);
  }
  mpz_clear(caller);
  mpz_clear(value);
  mpz_clear(to);
  mpz_clear(nonce);
  mpz_clear(tx_origin);
  mpz_clear(tx_gasprice);
  mpz_clear(gas);
  return cpu_messages;
}

template<class params>
__host__ void free_host_messages(message_t<params> *cpu_messages, size_t count) {
  for(size_t idx=0; idx<count; idx++) {
    // data
    free(cpu_messages[idx].data.data);
  }
  free(cpu_messages);
}


template<class params>
__host__ message_t<params> *to_gpu(message_t<params> *cpu_messages, size_t count) {
  message_t<params> *gpu_messages, *tmp_cpu_messages;
  tmp_cpu_messages = (message_t<params> *)malloc(count*sizeof(message_t<params>));
  memcpy(tmp_cpu_messages, cpu_messages, count*sizeof(message_t<params>));

  for(size_t idx=0; idx<count; idx++) {
    // data
    cudaMalloc((void **)&(tmp_cpu_messages[idx].data.data), sizeof(uint8_t)*tmp_cpu_messages[idx].data.size);
    cudaMemcpy(tmp_cpu_messages[idx].data.data, cpu_messages[idx].data.data, sizeof(uint8_t)*tmp_cpu_messages[idx].data.size, cudaMemcpyHostToDevice);
  }
  //write_messages<params>(stdout, tmp_cpu_instaces, count);
  cudaMalloc((void **)&gpu_messages, sizeof(message_t<params>)*count);
  cudaMemcpy(gpu_messages, tmp_cpu_messages, sizeof(message_t<params>)*count, cudaMemcpyHostToDevice);
  free(tmp_cpu_messages);
  return gpu_messages;
}

template<class params>
__host__ void free_gpu_messages(message_t<params> *gpu_messages, size_t count) {
  message_t<params> *tmp_cpu_messages;
  tmp_cpu_messages = (message_t<params> *)malloc(count*sizeof(message_t<params>));
  cudaMemcpy(tmp_cpu_messages, gpu_messages, count*sizeof(message_t<params>), cudaMemcpyDeviceToHost);

  for(size_t idx=0; idx<count; idx++) {
    // data
    cudaFree(tmp_cpu_messages[idx].data.data);
  }
  free(tmp_cpu_messages);
  cudaFree(gpu_messages);
}

template<class params>
__host__ void print_messages(FILE *fp, message_t<params> *cpu_messages, size_t count) {
  size_t idx=0, jdx=0;
  for(idx=0; idx<count; idx++) {
    fprintf(fp, "INSTACE: %08x , CALLER: ", idx);
    for(uint32_t jdx=0; jdx<params::BITS/32; jdx++) {
      fprintf(fp, "%08x ", cpu_instances[idx].caller._limbs[jdx]);
    }
    fprintf(fp, ", VALUE: ");
    for(uint32_t jdx=0; jdx<params::BITS/32; jdx++) {
      fprintf(fp, "%08x ", cpu_instances[idx].value._limbs[jdx]);
    }
    fprintf(fp, ", TO: ");
    for(uint32_t jdx=0; jdx<params::BITS/32; jdx++) {
      fprintf(fp, "%08x ", cpu_instances[idx].to._limbs[jdx]);
    }
    fprintf(fp, ", TX_ORIGIN: ");
    for(uint32_t jdx=0; jdx<params::BITS/32; jdx++) {
      fprintf(fp, "%08x ", cpu_instances[idx].tx.origin._limbs[jdx]);
    }
    fprintf(fp, ", TX_GASPRICE: ");
    for(uint32_t jdx=0; jdx<params::BITS/32; jdx++) {
      fprintf(fp, "%08x ", cpu_instances[idx].tx.gasprice._limbs[jdx]);
    }
    fprintf(fp, ", DATA_SIZE: ");
    fprintf(fp, "%lx ", cpu_instances[idx].data.size);
    fprintf(fp, "\n");
  }
}

#endif