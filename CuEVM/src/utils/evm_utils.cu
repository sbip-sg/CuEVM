
#include <CuCrypto/keccak.cuh>
#include <CuEVM/utils/error_codes.cuh>
#include <CuEVM/utils/evm_utils.cuh>

namespace CuEVM::utils {
__host__ __device__ void get_bit_array(uint8_t *dst_array, uint32_t &array_length, evm_word_t &src_cgbn_mem,
                                       uint32_t limb_count) {
    uint32_t current_limb;
    uint32_t bitIndex = 0;  // Index for each bit in dst_array
    array_length = 0;
    for (uint32_t idx = 0; idx < limb_count; idx++) {
        current_limb = src_cgbn_mem.words[limb_count - 1 - idx];
        for (int bit = 31; bit >= 0; --bit) {  // hardcoded 32 bits per limb
            // Extract each bit from the current limb and store '0' or '1' in
            // dst_array
            dst_array[bitIndex++] = (current_limb & (1U << bit)) ? 1 : 0;
            if (dst_array[bitIndex - 1] == 1 && array_length == 0) {
                array_length = 256 - (bitIndex - 1);
            }
        }
    }
}

__host__ __device__ void byte_array_from_cgbn_memory(uint8_t *dst_array, size_t &array_length, evm_word_t &src_cgbn_mem,
                                                     size_t limb_count) {
    size_t current_limb;
    array_length = limb_count * 4;  // Each limb has 4 bytes

    for (size_t idx = 0; idx < limb_count; idx++) {
        current_limb = src_cgbn_mem.words[limb_count - 1 - idx];
        dst_array[idx * 4] = (current_limb >> 24) & 0xFF;  // Extract the most significant byte
        dst_array[idx * 4 + 1] = (current_limb >> 16) & 0xFF;
        dst_array[idx * 4 + 2] = (current_limb >> 8) & 0xFF;
        dst_array[idx * 4 + 3] = current_limb & 0xFF;  // Extract the least significant byte
    }
}

__host__ __device__ void evm_address_conversion(evm_word_t &address) {
    for (uint32_t idx = 0; idx < 3; idx++) {
        address.words[idx] = 0;
    }
}

__host__ __device__ int32_t get_contract_address_create_word(evm_word_t *contract_address, evm_word_t *sender_address,
                                                             evm_word_t *sender_nonce) {
    CuEVM::byte_array_t sender_address_bytes, sender_nonce_bytes;
    sender_address->to_byte_array_t(sender_address_bytes);
    sender_nonce->to_byte_array_t(sender_nonce_bytes);

    uint32_t nonce_bytes;
    for (nonce_bytes = CuEVM::word_size; nonce_bytes > 0; nonce_bytes--) {
        if (sender_nonce_bytes.data[CuEVM::word_size - nonce_bytes] != 0) {
            break;
        }
    }

    uint8_t rlp_list[1 + 1 + CuEVM::address_size + 1 + CuEVM::word_size];
    rlp_list[1] = 0x80 + CuEVM::address_size;
    for (uint32_t idx = 0; idx < CuEVM::address_size; idx++) {
        rlp_list[2 + idx] = sender_address_bytes.data[CuEVM::word_size - CuEVM::address_size + idx];
    }

    uint32_t rlp_list_length;
    // 21 is from the address the 20 bytes is the length of the address
    // and the 1 byte is the 0x80 + length of the address (20)
    // if (cgbn_compare_ui32(arith.env, sender_nonce, 128) < 0) {
    if (uint256_get_uint32_t(sender_nonce) < 128) {
        rlp_list_length = 1 + CuEVM::address_size + 1;
        if (uint256_get_uint32_t(sender_nonce) == 0) {
            rlp_list[2 + CuEVM::address_size] = 0x80;  // special case for nonce 0
        } else {
            rlp_list[2 + CuEVM::address_size] = sender_nonce_bytes.data[CuEVM::word_size - 1];
        }
    } else {
        // 1 byte for the length of the nonce
        // 0x80 + length of the nonce
        rlp_list_length = 21 + 1 + nonce_bytes;
        rlp_list[2 + CuEVM::address_size] = 0x80 + nonce_bytes;
        for (uint8_t idx = 0; idx < nonce_bytes; idx++) {
            rlp_list[2 + CuEVM::address_size + 1 + idx] = sender_nonce_bytes.data[CuEVM::word_size - nonce_bytes + idx];
        }
    }
    rlp_list[0] = 0xc0 + rlp_list_length;

    CuEVM::byte_array_t hash_address_bytes(CuEVM::hash_size);
    CuCrypto::keccak::sha3(&(rlp_list[0]), rlp_list_length + 1, hash_address_bytes.data, CuEVM::hash_size);

    // cgbn_set_byte_array_t(arith.env, contract_address, hash_address_bytes);
    // cgbn_bitwise_mask_and(arith.env, contract_address, contract_address, CuEVM::address_bits);
    // todo check replacement
    // __ONE_THREAD_PER_INSTANCE(printf("\n\nhash_address_bytes\n"););
    hash_address_bytes.print();
    contract_address->from_byte_array_t(hash_address_bytes, BIG_ENDIAN);
    for (uint32_t idx = uint256_limbs - 3; idx < uint256_limbs; idx++) {
        contract_address->words[idx] = 0;
    }

    return ERROR_SUCCESS;
}

__host__ __device__ int32_t get_contract_address_create(evm_word_t *contract_address, evm_word_t *sender_address,
                                                        evm_word_t *sender_nonce) {
    CuEVM::byte_array_t sender_address_bytes, sender_nonce_bytes;
    sender_address->to_byte_array_t(sender_address_bytes);
    sender_nonce->to_byte_array_t(sender_nonce_bytes);

    uint32_t nonce_bytes;
    for (nonce_bytes = CuEVM::word_size; nonce_bytes > 0; nonce_bytes--) {
        if (sender_nonce_bytes.data[CuEVM::word_size - nonce_bytes] != 0) {
            break;
        }
    }
    // TODO: this might work only for CuEVM::word_size == 32

    // printf("get contract address create\n");
    // sender_address_word[INSTANCE_IDX_PER_BLOCK].print();
    // printf("sender nonce\n");
    // sender_nonce_word[INSTANCE_IDX_PER_BLOCK].print();

    uint8_t rlp_list[1 + 1 + CuEVM::address_size + 1 + CuEVM::word_size];
    rlp_list[1] = 0x80 + CuEVM::address_size;
    for (uint32_t idx = 0; idx < CuEVM::address_size; idx++) {
        rlp_list[2 + idx] = sender_address_bytes.data[CuEVM::word_size - CuEVM::address_size + idx];
    }

    uint32_t rlp_list_length;
    // 21 is from the address the 20 bytes is the length of the address
    // and the 1 byte is the 0x80 + length of the address (20)
    if (uint256_get_uint32_t(sender_nonce) < 128) {
        rlp_list_length = 1 + CuEVM::address_size + 1;
        if (uint256_get_uint32_t(sender_nonce) == 0) {
            rlp_list[2 + CuEVM::address_size] = 0x80;  // special case for nonce 0
        } else {
            rlp_list[2 + CuEVM::address_size] = sender_nonce_bytes.data[CuEVM::word_size - 1];
        }
    } else {
        // 1 byte for the length of the nonce
        // 0x80 + length of the nonce
        rlp_list_length = 21 + 1 + nonce_bytes;
        rlp_list[2 + CuEVM::address_size] = 0x80 + nonce_bytes;
        for (uint8_t idx = 0; idx < nonce_bytes; idx++) {
            rlp_list[2 + CuEVM::address_size + 1 + idx] = sender_nonce_bytes.data[CuEVM::word_size - nonce_bytes + idx];
        }
    }
    rlp_list[0] = 0xc0 + rlp_list_length;

    CuEVM::byte_array_t hash_address_bytes(CuEVM::hash_size);
    CuCrypto::keccak::sha3(&(rlp_list[0]), rlp_list_length + 1, hash_address_bytes.data, CuEVM::hash_size);

    // cgbn_set_byte_array_t(arith.env, contract_address, hash_address_bytes);
    // cgbn_bitwise_mask_and(arith.env, contract_address, contract_address, CuEVM::address_bits);
    // todo check replacement
    // __ONE_THREAD_PER_INSTANCE(printf("\n\nhash_address_bytes\n"););
    // hash_address_bytes.print();
    contract_address->from_byte_array_t(hash_address_bytes, BIG_ENDIAN);
    evm_address_conversion(*contract_address);

    return ERROR_SUCCESS;
}

__host__ __device__ int32_t get_contract_address_create2(evm_word_t *contract_address, evm_word_t *sender_address,
                                                         evm_word_t *salt, const CuEVM::byte_array_t &init_code) {
    CuEVM::byte_array_t sender_address_bytes, salt_bytes;

    uint256_to_bytes(sender_address_bytes.data, sender_address, sender_address_bytes.size);
    uint256_to_bytes(salt_bytes.data, salt, salt_bytes.size);
    uint32_t total_bytes = 1 + CuEVM::address_size + CuEVM::word_size + CuEVM::hash_size;

    CuEVM::byte_array_t hash_code(CuEVM::hash_size);
    CuCrypto::keccak::sha3(init_code.data, init_code.size, hash_code.data, CuEVM::hash_size);

    CuEVM::byte_array_t input_data(total_bytes);
    input_data.data[0] = 0xff;
    for (uint32_t idx = 0; idx < CuEVM::address_size; idx++) {
        input_data.data[1 + idx] = sender_address_bytes.data[CuEVM::word_size - CuEVM::address_size + idx];
    }
    for (uint32_t idx = 0; idx < CuEVM::word_size; idx++) {
        input_data.data[1 + CuEVM::address_size + idx] = salt_bytes.data[CuEVM::word_size - CuEVM::word_size + idx];
    }
    for (uint32_t idx = 0; idx < CuEVM::hash_size; idx++) {
        input_data.data[1 + CuEVM::address_size + CuEVM::word_size + idx] = hash_code.data[idx];
    }

    CuEVM::byte_array_t hash_input_data(CuEVM::hash_size);
    CuCrypto::keccak::sha3(input_data.data, total_bytes, hash_input_data.data, CuEVM::hash_size);

    contract_address->from_byte_array_t(hash_input_data, BIG_ENDIAN);
    evm_address_conversion(*contract_address);
    return ERROR_SUCCESS;
}

__host__ __device__ int32_t is_hex(const char hex) {
    return hex >= '0' && hex <= '9' ? 1 : (hex >= 'a' && hex <= 'f' ? 1 : (hex >= 'A' && hex <= 'F' ? 1 : 0));
}

__host__ __device__ char hex_from_nibble(const uint8_t nibble) {
    return nibble < 10 ? '0' + nibble : 'a' + nibble - 10;
}

__host__ __device__ uint8_t nibble_from_hex(const char hex) {
    return hex >= '0' && hex <= '9'
               ? hex - '0'
               : (hex >= 'a' && hex <= 'f' ? hex - 'a' + 10 : (hex >= 'A' && hex <= 'F' ? hex - 'A' + 10 : 0));
}

__host__ __device__ uint8_t byte_from_nibbles(const uint8_t high, const uint8_t low) { return (high << 4) | low; }

__host__ __device__ void hex_from_byte(char *dst, const uint8_t byte) {
    if (dst == NULL) return;
    dst[0] = hex_from_nibble(byte >> 4);
    dst[1] = hex_from_nibble(byte & 0x0F);
}

__host__ __device__ uint8_t byte_from_two_hex_char(const char high, const char low) {
    return byte_from_nibbles(nibble_from_hex(high), nibble_from_hex(low));
}
__host__ __device__ int32_t hex_string_length(const char *hex_string) {
    int32_t length;
    int32_t error = 0;
    char *current_char;
    current_char = (char *)hex_string;
    if ((hex_string[0] == '0') && ((hex_string[1] == 'x') || (hex_string[1] == 'X'))) {
        current_char += 2;  // Skip the "0x" prefix
    }
    length = 0;
    do {
        length++;
        error = error | (nibble_from_hex(current_char[length]) == 0);
    } while (current_char[length] != '\0');
    return error ? -1 : length;
}

__host__ __device__ int32_t clean_hex_string(char **hex_string) {
    char *current_char;
    current_char = (char *)*hex_string;
    if (current_char == NULL || current_char[0] == '\0') {
        return -1;
    }
    if ((current_char[0] == '0') && ((current_char[1] == 'x') || (current_char[1] == 'X'))) {
        current_char += 2;  // Skip the "0x" prefix
        *hex_string += 2;
    }
    if (current_char[0] == '\0') {
        return 0;
    }
    int32_t length = 0;
    int32_t error = 0;
    do {
        error = error || (is_hex(current_char[length++]) == 0);
    } while (current_char[length] != '\0');
    return error ? -1 : length;
}

__host__ __device__ int32_t hex_string_without_leading_zeros(char *hex_string) {
    int32_t length;
    char *current_char;
    current_char = (char *)hex_string;
    length = clean_hex_string(&current_char);
    if (length <= 0) {
        return 1;
    }
    int32_t prefix = current_char - hex_string;
    int32_t idx;
    for (idx = 0; idx < length; idx++) {
        if (*current_char++ != '0') {
            break;
        }
    }
    if (idx == length) {
        hex_string[prefix] = '0';
        hex_string[prefix + 1] = '\0';
    } else {
        char *dst_char;
        char *src_char;
        dst_char = (char *)hex_string;
        clean_hex_string(&dst_char);
        src_char = dst_char + idx;
        for (int32_t i = 0; i < length - idx; i++) {
            *(dst_char++) = *(src_char++);
        }
        *dst_char = '\0';
    }
    return 0;
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
}  // namespace CuEVM::utils
