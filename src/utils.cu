#include "include/utils.h"

__host__ size_t adjusted_length(char** hex_string) {
    if (strncmp(*hex_string, "0x", 2) == 0 || strncmp(*hex_string, "0X", 2) == 0) {
        *hex_string += 2;  // Skip the "0x" prefix
        return (strlen(*hex_string) / 2);
    }
    return (strlen(*hex_string) / 2);
}

__host__ const char* adjust_hex_string(const char* hex_string) {
    if (strlen(hex_string) >= 2 && (hex_string[0] == '0' && (hex_string[1] == 'x' || hex_string[1] == 'X')))
                hex_string += 2;  // Skip the "0x" prefix
    if (strlen(hex_string) % 2 != 0) {
        printf("Invalid hex_string length\n");
        return NULL;
    }
    return hex_string;
}
__host__ void hex_to_bytes(const char *hex_string, uint8_t *byte_array, size_t length)
{
    for (size_t idx = 0; idx < length; idx += 2)
    {
        sscanf(&hex_string[idx], "%2hhx", &byte_array[idx / 2]);
    }
}

// support routines
void cuda_check(cudaError_t status, const char *action, const char *file, int32_t line) {
  // check for cuda errors

  if(status!=cudaSuccess) {
    printf("CUDA error occurred: %s\n", cudaGetErrorString(status));
    if(action!=NULL)
      printf("While running %s   (file %s, line %d)\n", action, file, line);
    exit(1);
  }
}

void cgbn_check(cgbn_error_report_t *report, const char *file, int32_t line) {
  // check for cgbn errors

  if(cgbn_error_report_check(report)) {
    printf("\n");
    printf("CGBN error occurred: %s\n", cgbn_error_string(report));

    if(report->_instance!=0xFFFFFFFF) {
      printf("Error reported by instance %d", report->_instance);
      if(report->_blockIdx.x!=0xFFFFFFFF || report->_threadIdx.x!=0xFFFFFFFF)
        printf(", ");
      if(report->_blockIdx.x!=0xFFFFFFFF)
      printf("blockIdx=(%d, %d, %d) ", report->_blockIdx.x, report->_blockIdx.y, report->_blockIdx.z);
      if(report->_threadIdx.x!=0xFFFFFFFF)
        printf("threadIdx=(%d, %d, %d)", report->_threadIdx.x, report->_threadIdx.y, report->_threadIdx.z);
      printf("\n");
    }
    else {
      printf("Error reported by blockIdx=(%d %d %d)", report->_blockIdx.x, report->_blockIdx.y, report->_blockIdx.z);
      printf("threadIdx=(%d %d %d)\n", report->_threadIdx.x, report->_threadIdx.y, report->_threadIdx.z);
    }
    if(file!=NULL)
      printf("file %s, line %d\n", file, line);
    exit(1);
  }
}



__host__ void from_mpz(uint32_t *words, uint32_t count, mpz_t value) {
  size_t written;

  if(mpz_sizeinbase(value, 2)>count*32) {
    fprintf(stdout, "from_mpz failed -- result does not fit\n");
    exit(1);
  }

  mpz_export(words, &written, -1, sizeof(uint32_t), 0, 0, value);
  while(written<count)
    words[written++]=0;
}

__host__ void to_mpz(mpz_t r, uint32_t *x, uint32_t count) {
  mpz_import(r, count, -1, sizeof(uint32_t), 0, 0, x);
}

__host__ cJSON *get_json_from_file(const char *filepath) {
    FILE *fp = fopen(filepath, "r");
    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char *buffer = (char *)malloc(size + 1);
    fread(buffer, 1, size, fp);
    fclose(fp);
    buffer[size] = '\0';
    // parse
    cJSON *root = cJSON_Parse(buffer);
    free(buffer);
    return root;
}
