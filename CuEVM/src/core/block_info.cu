#include <CuEVM/core/block_info.cuh>
#include <CuEVM/utils/error_codes.cuh>

namespace CuEVM {
__device__ block_info_t *global_block_info;
__device__ block_info_t::block_info_t() {
    coin_base.from_uint32_t(0);
    difficulty.from_uint32_t(0);
    prevrandao.from_uint32_t(0);
    number = 0;
    gas_limit.from_uint32_t(0);
    time_stamp = 0;
    base_fee.from_uint32_t(0);
    chain_id.from_uint32_t(0);
    for (size_t idx = 0; idx < 256; idx++) {
        previous_blocks[idx].number.from_uint32_t(0);
        previous_blocks[idx].hash.from_uint32_t(0);
    }
}

__device__ block_info_t::~block_info_t() {}

__host__ block_info_t::block_info_t(const cJSON *json) { from_json(json); }

__host__ int32_t block_info_t::from_json(const cJSON *json) {
    cJSON *block_json = nullptr;
    cJSON *element_json = nullptr;
    cJSON *previous_blocks_json = nullptr;
    size_t idx = 0;

    block_json = cJSON_GetObjectItemCaseSensitive(json, "env");

    element_json = cJSON_GetObjectItemCaseSensitive(block_json, "currentCoinbase");
    coin_base.from_hex(element_json->valuestring);
    evm_word_t temp_value;
    element_json = cJSON_GetObjectItemCaseSensitive(block_json, "currentTimestamp");
    temp_value.from_hex(element_json->valuestring);
    time_stamp = uint256_get_uint64_t(&temp_value);

    element_json = cJSON_GetObjectItemCaseSensitive(block_json, "currentNumber");
    temp_value.from_hex(element_json->valuestring);
    number = uint256_get_uint64_t(&temp_value);

    element_json = cJSON_GetObjectItemCaseSensitive(block_json, "currentDifficulty");
    difficulty.from_hex(element_json->valuestring);

    element_json = cJSON_GetObjectItemCaseSensitive(block_json, "currentRandom");
    if (element_json != nullptr) {
        prevrandao.from_hex(element_json->valuestring);
    }

    element_json = cJSON_GetObjectItemCaseSensitive(block_json, "currentGasLimit");
    gas_limit.from_hex(element_json->valuestring);

    // element_json=cJSON_GetObjectItemCaseSensitive(block_json, "currentChainId");
    // arith.cgbn_memory_from_hex_string(chain_id, element_json->valuestring);
    chain_id.from_uint32_t(1);

    element_json = cJSON_GetObjectItemCaseSensitive(block_json, "currentBaseFee");
    base_fee.from_hex(element_json->valuestring);

    previous_blocks_json = cJSON_GetObjectItemCaseSensitive(block_json, "previousHashes");
    if ((previous_blocks_json != nullptr) && cJSON_IsArray(previous_blocks_json)) {
        idx = 0;
        cJSON_ArrayForEach(element_json, previous_blocks_json) {
            element_json = cJSON_GetObjectItemCaseSensitive(element_json, "number");
            previous_blocks[idx].number.from_hex(element_json->valuestring);

            element_json = cJSON_GetObjectItemCaseSensitive(element_json, "hash");
            previous_blocks[idx].hash.from_hex(element_json->valuestring);
            idx++;
        }
    } else {
        idx = 0;
        // fill the block with number 0 and the hash if given
        previous_blocks[0].number.from_uint32_t(0);

        element_json = cJSON_GetObjectItemCaseSensitive(block_json, "previousHash");

        if (element_json != nullptr) {
            previous_blocks[0].hash.from_hex(element_json->valuestring);
        } else {
            previous_blocks[0].hash.from_uint32_t(0);
        }

        idx++;
    }

    // fill the remaing parents with 0
    for (size_t jdx = idx; jdx < 256; jdx++) {
        previous_blocks[jdx].number.from_uint32_t(0);
        previous_blocks[jdx].hash.from_uint32_t(0);
    }
    return ERROR_SUCCESS;
}

__device__ int32_t block_info_t::get_previous_hash(evm_word_t &previous_hash, const evm_word_t &previous_number) const {
    uint32_t idx = 0;
    uint32_t number_uint = this->number;
    uint32_t previous_number_uint = previous_number.get_uint32_t();
    // if the rquest number is greater than the current block number
    if (number_uint < previous_number_uint) {
        previous_hash.set_zero();
        return ERROR_BLOCK_INVALID_NUMBER;
    }
    // get the distance from the current block number to the requested block number
    number_uint = number_uint - previous_number_uint;
    idx = number_uint - 1;
    // only the last 256 blocks are stored
    if (idx > 255) {
        previous_hash.set_zero();
        return ERROR_BLOCK_INVALID_NUMBER;
    }
    previous_hash = previous_blocks[idx].hash;
    return ERROR_SUCCESS;
}

__host__ __device__ void block_info_t::print() const {
    uint32_t idx = 0;
    printf("BLOCK: \n");
    printf("COINBASE: ");
    coin_base.print();
    printf("TIMESTAMP: %d\n", time_stamp);
    printf("NUMBER: %d\n", number);
    printf("DIFICULTY: ");
    difficulty.print();
    printf("GASLIMIT: ");
    gas_limit.print();
    printf("CHAINID: ");
    chain_id.print();
    printf("BASE_FEE: ");
    base_fee.print();
    printf("PREVIOUS_BLOCKS: \n");
    for (idx = 0; idx < 256; idx++) {
        printf("NUMBER: ");
        previous_blocks[idx].number.print();
        printf("HASH: ");
        previous_blocks[idx].hash.print();
        printf("\n");
        if (previous_blocks[idx].number == 0) {
            break;
        }
    }
}

__host__ int32_t get_block_info(const cJSON *json) {
    // Create block info on host
    block_info_t *block_info_ptr = new block_info_t(json);

    // Allocate device memory for block info
    block_info_t *d_block_info;
    cudaMalloc(&d_block_info, sizeof(block_info_t));

    // Copy block info to device
    cudaMemcpy(d_block_info, block_info_ptr, sizeof(block_info_t), cudaMemcpyHostToDevice);

    // Copy pointer to symbol
    cudaMemcpyToSymbol(global_block_info, &d_block_info, sizeof(block_info_t *));

    return ERROR_SUCCESS;
}

}  // namespace CuEVM
