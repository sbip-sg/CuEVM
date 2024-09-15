// CuEVM: CUDA Ethereum Virtual Machine implementation
// Copyright 2023 Stefan-Dan Ciocirlan (SBIP - Singapore Blockchain Innovation
// Programme) Author: Stefan-Dan Ciocirlan Date: 2024-09-15
// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>

#include <CuEVM/utils/cuda_utils.cuh>

// #include <bitset>

namespace CuEVM {
constexpr CONSTANT uint32_t ACCOUNT_NONE_FLAG = 0;
constexpr CONSTANT uint32_t ACCOUNT_ADDRESS_FLAG = (1 << 0);
constexpr CONSTANT uint32_t ACCOUNT_BALANCE_FLAG = (1 << 1);
constexpr CONSTANT uint32_t ACCOUNT_NONCE_FLAG = (1 << 2);
constexpr CONSTANT uint32_t ACCOUNT_BYTE_CODE_FLAG = (1 << 3);
constexpr CONSTANT uint32_t ACCOUNT_STORAGE_FLAG = (1 << 4);
constexpr CONSTANT uint32_t ACCOUNT_DELETED_FLAG = (1 << 5);
constexpr CONSTANT uint32_t ACCOUNT_NON_STORAGE_FLAG =
    (ACCOUNT_ADDRESS_FLAG | ACCOUNT_BALANCE_FLAG | ACCOUNT_NONCE_FLAG |
     ACCOUNT_BYTE_CODE_FLAG);
constexpr CONSTANT uint32_t ACCOUNT_ALL_FLAG =
    (ACCOUNT_ADDRESS_FLAG | ACCOUNT_BALANCE_FLAG | ACCOUNT_NONCE_FLAG |
     ACCOUNT_BYTE_CODE_FLAG | ACCOUNT_STORAGE_FLAG);
/**
 * The account flags.
 * The flags are used to indicate which fields are used.
 */
struct account_flags_t {
    // TODO: in some sollution we can use std::bitset<5> flags;
    uint32_t flags; /**< The flags */
    /**
     * The default constructor for the account flags.
     */
    __host__ __device__ account_flags_t() : flags(CuEVM::ACCOUNT_NONE_FLAG) {};

    /**
     * The constructor for the account flags.
     * @param[in] flags The flags
     */
    __host__ __device__ __forceinline__ account_flags_t(uint32_t flags)
        : flags(flags) {}
    /**
     * The copy constructor for the account flags.
     * @param[in] account_flags The account flags
     */
    __host__ __device__ __forceinline__
    account_flags_t(const account_flags_t &account_flags)
        : flags(account_flags.flags) {}

    /**
     * The assignment operator for the account flags.
     * @param[in] account_flags The account flags
     */
    __host__ __device__ __forceinline__ account_flags_t &operator=(
        const account_flags_t &account_flags) {
        flags = account_flags.flags;
        return *this;
    }
    /**
     * The assignment operator for the account flags.
     * @param[in] account_flags The account flags
     */
    __host__ __device__ __forceinline__ account_flags_t &operator=(
        const uint32_t &other_flags) {
        flags = other_flags;
        return *this;
    }
    /**
     * If the flag for the address is set.
     * @return If unset 0, otherwise 1
     */
    __host__ __device__ __forceinline__ uint32_t has_address() const {
        return flags & CuEVM::ACCOUNT_ADDRESS_FLAG;
    }

    /**
     * If the flag for the balance is set.
     * @return If unset 0, otherwise 1
     */
    __host__ __device__ __forceinline__ uint32_t has_balance() const {
        return flags & CuEVM::ACCOUNT_BALANCE_FLAG;
    }

    /**
     * If the flag for the nonce is set.
     * @return If unset 0, otherwise 1
     */
    __host__ __device__ __forceinline__ uint32_t has_nonce() const {
        return flags & CuEVM::ACCOUNT_NONCE_FLAG;
    }

    /**
     * If the flag for the byte code is set.
     * @return If unset 0, otherwise 1
     */
    __host__ __device__ __forceinline__ uint32_t has_byte_code() const {
        return flags & CuEVM::ACCOUNT_BYTE_CODE_FLAG;
    }

    /**
     * If the flag for the storage is set.
     * @return If unset 0, otherwise 1
     */
    __host__ __device__ __forceinline__ uint32_t has_storage() const {
        return flags & CuEVM::ACCOUNT_STORAGE_FLAG;
    }

    /**
     * If the flag for the deleted is set.
     * @return If unset 0, otherwise 1
     */
    __host__ __device__ __forceinline__ uint32_t has_deleted() const {
        return flags & CuEVM::ACCOUNT_DELETED_FLAG;
    }

    /**
     * Set the flag for the address.
     */
    __host__ __device__ __forceinline__ void set_address() {
        flags |= CuEVM::ACCOUNT_ADDRESS_FLAG;
    }

    /**
     * Unset the flag for the address.
     */
    __host__ __device__ __forceinline__ void unset_address() {
        flags &= ~CuEVM::ACCOUNT_ADDRESS_FLAG;
    }

    /**
     * Set the flag for the balance.
     */
    __host__ __device__ __forceinline__ void set_balance() {
        flags |= CuEVM::ACCOUNT_BALANCE_FLAG;
    }

    /**
     * Unset the flag for the balance.
     */
    __host__ __device__ __forceinline__ void unset_balance() {
        flags &= ~CuEVM::ACCOUNT_BALANCE_FLAG;
    }

    /**
     * Set the flag for the nonce.
     */
    __host__ __device__ __forceinline__ void set_nonce() {
        flags |= CuEVM::ACCOUNT_NONCE_FLAG;
    }

    /**
     * Unset the flag for the nonce.
     */
    __host__ __device__ __forceinline__ void unset_nonce() {
        flags &= ~CuEVM::ACCOUNT_NONCE_FLAG;
    }

    /**
     * Set the flag for the byte code.
     */
    __host__ __device__ __forceinline__ void set_byte_code() {
        flags |= CuEVM::ACCOUNT_BYTE_CODE_FLAG;
    }

    /**
     * Unset the flag for the byte code.
     */
    __host__ __device__ __forceinline__ void unset_byte_code() {
        flags &= ~CuEVM::ACCOUNT_BYTE_CODE_FLAG;
    }

    /**
     * Set the flag for the storage.
     */
    __host__ __device__ __forceinline__ void set_storage() {
        flags |= CuEVM::ACCOUNT_STORAGE_FLAG;
    }

    /**
     * Unset the flag for the storage.
     */
    __host__ __device__ __forceinline__ void unset_storage() {
        flags &= ~CuEVM::ACCOUNT_STORAGE_FLAG;
    }

    /**
     * Set the flag for the deleted.
     */
    __host__ __device__ __forceinline__ void set_deleted() {
        flags = CuEVM::ACCOUNT_DELETED_FLAG;
    }

    /**
     * Unset the flag for the deleted.
     */
    __host__ __device__ __forceinline__ void unset_deleted() {
        flags &= ~CuEVM::ACCOUNT_DELETED_FLAG;
    }

    /**
     * Update the flags with the given flags.
     * @param[in] other_flags The other flags
     */
    __host__ __device__ __forceinline__ void update(
        const account_flags_t &other_flags) {
        flags = (has_deleted() ? other_flags.flags : flags | other_flags.flags);
    }
    /**
     * Reset all flags
     */
    __host__ __device__ __forceinline__ void reset() {
        flags = CuEVM::ACCOUNT_NONE_FLAG;
    }

    /**
     * Set all the flags.
     */
    __host__ __device__ __forceinline__ void set_all() {
        flags = CuEVM::ACCOUNT_ALL_FLAG;
    }

    /**
     * Print the account flags.
     */
    __host__ __device__ void print() const;
    /**
     * Get the hextring of the account flags.
     * @param[inout] hex The hex string
     * @return The hex string
     */
    __host__ char *to_hex(char *hex) const;
};

}  // namespace CuEVM