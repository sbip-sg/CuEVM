#include <CuCrypto/keccak.cuh>
#include <CuEVM/state/account.cuh>
#include <CuEVM/utils/error_codes.cuh>

namespace CuEVM {
__global__ void account_t_transfer_kernel(account_t *dst_instances, account_t *src_instances, uint32_t count) {
    uint32_t instance = blockIdx.x * blockDim.x + threadIdx.x;

    if (instance >= count) return;
    // TODO Fix this later
    // CuEVM::account_t::transfer_memory(dst_instances[instance], src_instances[instance]);
}

__host__ __device__ account_t::account_t(const account_t &account) {
    address = account.address;
    balance = account.balance;
    nonce = account.nonce;
    byte_code = account.byte_code;
    storage = account.storage;
}

__host__ __device__ account_t::account_t(const account_t &account, const account_flags_t &flags) {
    address = account.address;
    balance = account.balance;
    nonce = account.nonce;
    if (flags.has_byte_code()) {
        byte_code = account.byte_code;
    }
    if (flags.has_storage()) {
        storage = account.storage;
    }
}

__host__ __device__ account_t::account_t(const account_t *account_ptr, const account_flags_t &flags) {
    address = account_ptr->address;
    balance = account_ptr->balance;
    nonce = account_ptr->nonce;
    if (flags.has_byte_code()) {
        byte_code = account_ptr->byte_code;
    }
    if (flags.has_storage()) {
        storage = account_ptr->storage;
    }
}

__host__ __device__ account_t::account_t(const evm_word_t &address) : storage(), byte_code(0U) {
    this->address = address;
    evm_word_t tmp;
    tmp.set_zero();
    this->balance = tmp;
    this->nonce = tmp;
}

__host__ __device__ account_t::account_t(evm_word_t *address) : storage(), byte_code(0U) {
    this->address = *address;
    evm_word_t tmp;
    tmp.set_zero();
    this->balance = tmp;
    this->nonce = tmp;
}

__host__ __device__ account_t::~account_t() { free(); }

__host__ __device__ void account_t::free() {
    byte_code.free();
    // printf("after byte_code.free()\n");
    storage.free();
    // printf("after storage.free()\n");
    clear();
    // printf("after clear()\n");
}

__host__ void account_t::free_managed() {
    byte_code.free_managed();
    storage.free_managed();
    clear();
}

__host__ __device__ void account_t::clear() { empty(); }

__host__ __device__ account_t &account_t::operator=(const account_t &other) {
    if (this != &other) {
        free();
        __ONE_GPU_THREAD_BEGIN__
        memcpy(&address, &other.address, sizeof(evm_word_t));
        memcpy(&balance, &other.balance, sizeof(evm_word_t));
        memcpy(&nonce, &other.nonce, sizeof(evm_word_t));
        // printf("other.bytecode %p\n", other.byte_code.data);
        // printf("other.bytecode.size %d\n", other.byte_code.size);
        // printf("other.storage.size %d\n", other.storage.size);
        // printf("other.storage.capacity %d\n", other.storage.capacity);
        // printf("other.storage.storage %p\n", other.storage.storage);
        __ONE_GPU_THREAD_END__
        byte_code = other.byte_code;
        storage = other.storage;
    }
    return *this;
}

__host__ __device__ int32_t account_t::get_storage_value(const evm_word_t &key, evm_word_t &value) {
    return storage.get_value(key, value);
}

__host__ __device__ int32_t account_t::set_storage_value(const evm_word_t &key, const evm_word_t &value) {
    return storage.set_value(key, value);
}

__host__ __device__ void account_t::get_address(evm_word_t &address) { this->address = address; }

__host__ __device__ void account_t::get_balance(evm_word_t &balance) { this->balance = balance; }

__host__ __device__ void account_t::get_nonce(evm_word_t &nonce) { this->nonce = nonce; }

__host__ __device__ byte_array_t account_t::get_byte_code() const { return byte_code; }

__host__ __device__ void account_t::set_nonce(const evm_word_t &nonce) { this->nonce = nonce; }

__host__ __device__ void account_t::set_balance(const evm_word_t &balance) { this->balance = balance; }

__host__ __device__ void account_t::set_address(const evm_word_t *address) {
    // cgbn_store(arith.env, &this->address, address);
    this->address = *address;
}

__host__ __device__ void account_t::set_byte_code(const byte_array_t &byte_code) { this->byte_code = byte_code; }

__host__ __device__ int32_t account_t::has_address(const evm_word_t *address) { return this->address == *address; }

__host__ __device__ void account_t::update(const account_t &other, const account_flags_t &flags) {
    if (flags.has_address()) {
        address = other.address;
    }
    if (flags.has_balance()) {
        balance = other.balance;
    }
    if (flags.has_nonce()) {
        nonce = other.nonce;
    }
    if (flags.has_byte_code()) {
        byte_code = other.byte_code;
    }
    if (flags.has_storage()) {
        storage.update(other.storage);
    }
}

__host__ __device__ bool account_t::is_empty() {
    return ((balance == 0) && (nonce == 0) && (byte_code.size == 0)) ? true : false;
}

__host__ __device__ bool account_t::is_empty_create() {
    // Goethereum: nonce ==0 && code == 0, can have balance

    return ((nonce == 0) && (byte_code.size == 0)) ? true : false;
}

__host__ __device__ int32_t account_t::is_contract() { return (byte_code.size > 0); }

__host__ __device__ void account_t::empty() {
    memset(&address, 0, sizeof(evm_word_t));
    this->balance.set_zero();
    this->nonce.set_zero();

    byte_code.clear();
    storage.clear();
}

__host__ void account_t::from_json(const cJSON *account_json, int32_t managed) {
    cJSON *balance_json, *nonce_json;

    address.from_hex(account_json->string);

    // set the balance
    balance_json = cJSON_GetObjectItemCaseSensitive(account_json, "balance");
    balance.from_hex(balance_json->valuestring);

    // set the nonce
    nonce_json = cJSON_GetObjectItemCaseSensitive(account_json, "nonce");
    nonce.from_hex(nonce_json->valuestring);

    byte_code.from_hex(cJSON_GetObjectItemCaseSensitive(account_json, "code")->valuestring, LITTLE_ENDIAN, NO_PADDING,
                       managed);

    storage.from_json(cJSON_GetObjectItemCaseSensitive(account_json, "storage"), managed);
#ifdef EIP_3155
// printf("byte_code.size %d\n", byte_code.size);
// printf("byte_code.data %p\n", byte_code.data);
// printf("storage.size %d\n", storage.size);
// printf("storage.capacity %d\n", storage.capacity);
// printf("storage.storage %p\n", storage.storage);
#endif
}

__host__ cJSON *account_t::to_json() const {
    cJSON *account_json = cJSON_CreateObject();
    char *bytes_string = nullptr;
    char *hex_string_ptr = new char[CuEVM::word_size * 2 + 3];
    address.to_hex(hex_string_ptr, 0, 5);
    cJSON_SetValuestring(account_json, hex_string_ptr);
    balance.to_hex(hex_string_ptr);
    cJSON_AddStringToObject(account_json, "balance", hex_string_ptr);
    nonce.to_hex(hex_string_ptr);
    cJSON_AddStringToObject(account_json, "nonce", hex_string_ptr);
    bytes_string = byte_code.to_hex();
    cJSON_AddStringToObject(account_json, "code", bytes_string);
    delete[] bytes_string;
    cJSON_AddItemToObject(account_json, "storage", storage.to_json(1));
    delete[] hex_string_ptr;
    hex_string_ptr = nullptr;
    return account_json;
}

__host__ __device__ void account_t::print() {
    __ONE_GPU_THREAD_WOSYNC_BEGIN__
    printf("Account:\n");
    address.print();
    printf("Balance: ");
    balance.print();
    printf("Nonce: ");
    nonce.print();
    printf("Byte code: ");
    byte_code.print();
    printf("Storage: \n");
    storage.print();
    __ONE_GPU_THREAD_WOSYNC_END__
}

// __host__ __device__ void account_t::transfer_memory(account_t &dst, account_t &src) {
//     CuEVM::byte_array_t::transfer_memory(dst.byte_code, src.byte_code);

//     CuEVM::contract_storage_t::transfer_memory(dst.storage, src.storage);
//     // copy the others not necesary
//     memcpy(&dst.nonce, &src.nonce, sizeof(evm_word_t));
//     memcpy(&dst.balance, &src, sizeof(evm_word_t));
//     memcpy(&dst.address, &src, sizeof(evm_word_t));
// }

__host__ cJSON *account_t::merge_json(const account_t *&account1_ptr, const account_t *&account2_ptr,
                                      const account_flags_t &flags) {
    cJSON *account_json = cJSON_CreateObject();
    char *hex_string_ptr = new char[CuEVM::word_size * 2 + 3];
    account1_ptr->address.to_hex(hex_string_ptr, 0, 5);
    cJSON_AddStringToObject(account_json, "address", hex_string_ptr);

    if (flags.has_balance()) {
        account2_ptr->balance.to_hex(hex_string_ptr, 1);
    } else {
        account1_ptr->balance.to_hex(hex_string_ptr, 1);
    }
    cJSON_AddStringToObject(account_json, "balance", hex_string_ptr);

    if (flags.has_nonce()) {
        account2_ptr->nonce.to_hex(hex_string_ptr, 1);
    } else {
        account1_ptr->nonce.to_hex(hex_string_ptr, 1);
    }
    cJSON_AddStringToObject(account_json, "nonce", hex_string_ptr);

    char *code_hash_hex_string_ptr = nullptr;
    char *code_hex_string_ptr = nullptr;
    CuEVM::byte_array_t *hash;
    hash = new CuEVM::byte_array_t(CuEVM::hash_size);
    if (flags.has_byte_code()) {
        CuCrypto::keccak::sha3(account2_ptr->byte_code.data, account2_ptr->byte_code.size, hash->data, hash->size);
        code_hex_string_ptr = account2_ptr->byte_code.to_hex();
    } else {
        CuCrypto::keccak::sha3(account1_ptr->byte_code.data, account1_ptr->byte_code.size, hash->data, hash->size);
        code_hex_string_ptr = account1_ptr->byte_code.to_hex();
    }
    cJSON_AddStringToObject(account_json, "code", code_hex_string_ptr);
    delete[] code_hex_string_ptr;
    code_hash_hex_string_ptr = hash->to_hex();
    cJSON_AddStringToObject(account_json, "codeHash", code_hash_hex_string_ptr);
    delete[] code_hash_hex_string_ptr;
    delete hash;

    cJSON *storage_json = nullptr;

    if (flags.has_storage()) {
        storage_json = CuEVM::contract_storage_t::merge_json(account1_ptr->storage, account2_ptr->storage, 1);
    } else {
        storage_json = CuEVM::contract_storage_t::merge_json(account1_ptr->storage, account1_ptr->storage, 1);
    }

    cJSON_AddItemToObject(account_json, "storage", storage_json);

    delete[] hex_string_ptr;
    return account_json;
}

__host__ account_t *account_t::get_cpu(uint32_t count) {
    account_t *cpu_instances = new account_t[count];
    for (uint32_t index = 0; index < count; index++) {
        cpu_instances[index].empty();
    }
    return cpu_instances;
}

__host__ void account_t::free_cpu(account_t *cpu_instances, uint32_t count) { delete[] cpu_instances; }

}  // namespace CuEVM