#include "include/utils.cuh"

__host__ size_t adjusted_length(char** hex_string) {
    if (strncmp(*hex_string, "0x", 2) == 0 || strncmp(*hex_string, "0X", 2) == 0) {
        *hex_string += 2;  // Skip the "0x" prefix
        return (strlen(*hex_string) / 2);
    }
    return (strlen(*hex_string) / 2);
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


namespace cuEVM {
  namespace utils {
    __host__ __device__ int32_t is_hex(const char hex) {
      return hex >= '0' && hex <= '9' ? 1 : (
        hex >= 'a' && hex <= 'f' ? 1 : (
          hex >= 'A' && hex <= 'F' ? 1 : 0
        )
      );
    }

    __host__ __device__ char hex_from_nibble(const uint8_t nibble) {
      return nibble < 10 ? '0' + nibble : 'a' + nibble - 10;
    }
    
    __host__ __device__ uint8_t nibble_from_hex(const char hex) {
      return hex >= '0' && hex <= '9' ? hex - '0' : (
        hex >= 'a' && hex <= 'f' ? hex - 'a' + 10 : (
          hex >= 'A' && hex <= 'F' ? hex - 'A' + 10 : 0
        )
      );
    }

    __host__ __device__ uint8_t byte_from_nibbles(const uint8_t high, const uint8_t low) {
      return (high << 4) | low;
    }
    
    __host__ __device__ void hex_from_byte(char *dst, const uint8_t byte){
      if (dst == NULL)
        return;
      dst[0] = hex_from_nibble(byte >> 4);
      dst[1] = hex_from_nibble(byte & 0x0F);
    }

    __host__ __device__ uint8_t byte_from_two_hex_char(const char high, const char low) {
      return byte_from_nibbles(nibble_from_hex(high), nibble_from_hex(low));
    }
    __host__ __device__ int32_t hex_string_length(
      const char *hex_string)
    {
      int32_t length;
      int32_t error = 0;
      char *current_char;
      current_char = (char *)hex_string;
      if (
        (hex_string[0] == '0') &&
        ((hex_string[1] == 'x') || (hex_string[1] == 'X'))
      ) {
        current_char += 2; // Skip the "0x" prefix
      }
      length = 0;
      do {
        length++;
        error = error | (nibble_from_hex(current_char[length]) == 0);
      } while(current_char[length] != '\0');
      return error ? -1 : length;
    }

    __host__ __device__ int32_t clean_hex_string(
      char **hex_string)
    {
      char *current_char;
      current_char = (char *)*hex_string;
      if (current_char == NULL || current_char[0] == '\0')
      {
        return 1;
      }
      if (
        (current_char[0] == '0') &&
        ((current_char[1] == 'x') || (current_char[1] == 'X'))
      ) {
        current_char += 2; // Skip the "0x" prefix
        *hex_string += 2;
      }
      int32_t length = 0;
      int32_t error = 0;
      do {
        error = error || (is_hex(current_char[length++]) == 0);
      } while(current_char[length] != '\0');
      return error ? -1 : length;
    }

    __host__ __device__ int32_t hex_string_without_leading_zeros(
      char *hex_string) {
      int32_t length;
      char *current_char;
      current_char = (char *)hex_string;
      length = clean_hex_string(&current_char);
      if (length <= 0)
      {
        return 1;
      }
      int32_t prefix = current_char - hex_string;
      int32_t idx;
      for (idx = 0; idx < length; idx++)
      {
        if (current_char++ != '0')
        {
          break;
        }
      }
      if (idx == length)
      {
        hex_string[prefix] = '0';
        hex_string[prefix + 1] = '\0';
      }
      else
      {
        char *dst_char;
        char *src_char;
        dst_char = (char *)hex_string;
        clean_hex_string(&dst_char);
        src_char = dst_char + idx;
        for (int32_t i = 0; i < length - idx; i++)
        {
          *(dst_char++) = *(src_char++);
        }
        *dst_char = '\0';
      }
      return 0;
    }
  } // namespace utils
} // namespace cuEVM