// Eliptic curve utilities using CGBN
#ifndef _ECC_H_
#define _ECC_H_
#include "include/utils.h"
#include "keccak.cuh"
/// The secp256k1 field prime number (P) and order
inline constexpr char *secp256k1_FieldPrime = "0xfffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f";
inline constexpr char *secp256k1_Order = "0xfffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141";

inline constexpr char *secp256k1_GX = "0x79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798";
inline constexpr char *secp256k1_GY = "0x483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8";

/// The alt_BN128 field prime number (P) and order
inline constexpr char *alt_BN128_FieldPrime = "0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47";
inline constexpr char *alt_BN128_Order = "0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001";

namespace ecc {

    typedef struct {
        evm_word_t r;
        evm_word_t s;
        uint32_t v;
        evm_word_t msg_hash;
        evm_word_t res;
        evm_word_t signer;
    } signature_t;

    typedef struct {
        evm_word_t FP;
        evm_word_t Order;
        evm_word_t GX;
        evm_word_t GY;
        // A!=0 is not supported for simplicity
        uint32_t B;
        // y^2 = x^3 + 7 // y^2 = x^3 + 3
        // y^2 = x^3 + Ax + B
    } Curve;

    // Add two point on the curve P and Q
    void ec_add(arith_t arith, Curve curve, bn_t &ResX, bn_t &ResY, bn_t &Px, bn_t &Py, bn_t &Qx, bn_t &Qy) {
        bn_t mod_fp;
        bn_t lambda, numerator, denominator, temp, x_r, y_r;
        evm_word_t scratch_pad;
        cgbn_load(arith._env, mod_fp, &curve.FP);

        if (cgbn_equals(arith._env, Px, Qx) && cgbn_equals(arith._env, Py, Qy)) {
            // Special case for doubling P == Q
            // printf("Doubling\n");
            // lambda = (3*Px^2) / (2*Py)
            arith.cgbn_mul_mod(arith._env, temp, Px, Px, mod_fp);  // temp = Px^2
            cgbn_set_ui32(arith._env, numerator, 3);
            arith.cgbn_mul_mod(arith._env, numerator, numerator, temp, mod_fp);  // numerator = 3*Px^2

            cgbn_set_ui32(arith._env, denominator, 2);
            arith.cgbn_mul_mod(arith._env, denominator, denominator, Py, mod_fp);  // denominator = 2*Py
            cgbn_modular_inverse(arith._env, denominator, denominator, mod_fp);

            arith.cgbn_mul_mod(arith._env, lambda, numerator, denominator, mod_fp);  // lambda = (3*Px^2) / (2*Py)
            // print lambda
            cgbn_store(arith._env, &scratch_pad, lambda);

        } else {
            // printf("Adding\n");
            // General case for P != Q
            // lambda = (Qy - Py) / (Qx - Px)
            arith.cgbn_sub_mod(arith._env, temp, Qy, Py, mod_fp);       // temp = Qy - Py
            arith.cgbn_sub_mod(arith._env, numerator, Qx, Px, mod_fp);  // numerator = Qx - Px
            cgbn_modular_inverse(arith._env, numerator, numerator, mod_fp);
            arith.cgbn_mul_mod(arith._env, lambda, temp, numerator, mod_fp);  // lambda = (Qy - Py) / (Qx - Px)
        }

        arith.cgbn_mul_mod(arith._env, x_r, lambda, lambda, mod_fp);  // x_r = lambda^2
        arith.cgbn_add_mod(arith._env, temp, Px, Qx, mod_fp);         // temp = Px + Qx
        arith.cgbn_sub_mod(arith._env, x_r, x_r, temp, mod_fp);       // x_r = lambda^2 - (Px + Qx)
        // y_r = lambda * (Px - x_r) - Py
        arith.cgbn_sub_mod(arith._env, temp, Px, x_r, mod_fp);      // temp = Px - x_r
        arith.cgbn_mul_mod(arith._env, y_r, lambda, temp, mod_fp);  // y_r = lambda * (Px - x_r)
        arith.cgbn_sub_mod(arith._env, y_r, y_r, Py, mod_fp);       // y_r = lambda * (Px - x_r) - Py
        // Set the result
        cgbn_set(arith._env, ResX, x_r);
        cgbn_set(arith._env, ResY, y_r);
    }
    // Multiply a point on the curve G by a scalar n, store result in Res
    void ec_mul(arith_t arith, Curve curve, bn_t &ResX, bn_t &ResY, bn_t &Gx, bn_t &Gy, bn_t &n) {
        bn_t mod_fp;
        evm_word_t scratch_pad;
        cgbn_load(arith._env, mod_fp, &curve.FP);
        uint8_t bitArray[evm_params::BITS];
        uint32_t bit_array_length = 0;

        cgbn_store(arith._env, &scratch_pad, n);
        arith.bit_array_from_cgbn_memory(bitArray, bit_array_length, scratch_pad);
        // // there is a bug if calling The result RES == G, need to copy to temps
        bn_t temp_ResX, temp_ResY;

        // Double-and-add algorithm
        cgbn_set(arith._env, temp_ResX, Gx);
        cgbn_set(arith._env, temp_ResY, Gy);

        for (int i = bit_array_length - 2; i >= 0; --i) {
            // Gz = 2 * Gz
            ec_add(arith, curve, temp_ResX, temp_ResY, temp_ResX, temp_ResY, temp_ResX, temp_ResY);

            if (bitArray[evm_params::BITS - 1 - i]) {
                ec_add(arith, curve, temp_ResX, temp_ResY, temp_ResX, temp_ResY, Gx, Gy);
            }
        }
        cgbn_set(arith._env, ResX, temp_ResX);
        cgbn_set(arith._env, ResY, temp_ResY);
    }

    void convert_point_to_address(arith_t arith, keccak::keccak_t keccak, bn_t& address, bn_t X, bn_t Y){

        evm_word_t scratch_pad;
        uint8_t input[64];
        uint8_t temp_array[32];
        size_t array_length = 0;
        cgbn_store(arith._env, &scratch_pad, X);
        arith.byte_array_from_cgbn_memory(temp_array, array_length, scratch_pad);
        for (int i = 0; i < 32; i++){
            input[i] = temp_array[i];
        }
        cgbn_store(arith._env, &scratch_pad, Y);
        arith.byte_array_from_cgbn_memory(temp_array, array_length, scratch_pad);
        for (int i = 0; i < 32; i++){
            input[i+32] = temp_array[i];
        }
        // print the entire byte array
        // print_byte_array_as_hex(input, 64);
        uint32_t in_length=64, out_length=32;
        keccak.sha3(input, in_length, (uint8_t*)temp_array, out_length);

        arith.cgbn_from_memory( address, temp_array);
        cgbn_bitwise_mask_and(arith._env, address, address, 160);

    }


    /**
     * Recover signer from the signature with the above structure. return the signer address in cgbn type
     *
     * @param arith
     * @param keccak
     * @param sig
     * @param signer
     */
    void ec_recover(arith_t arith, keccak::keccak_t keccak, signature_t sig,  bn_t &signer){

        Curve  curve;
        arith.cgbn_memory_from_hex_string(curve.FP, secp256k1_FieldPrime);
        arith.cgbn_memory_from_hex_string(curve.Order, secp256k1_Order);
        arith.cgbn_memory_from_hex_string(curve.GX, secp256k1_GX);
        arith.cgbn_memory_from_hex_string(curve.GY, secp256k1_GY);
        curve.B = 7;

        bn_t  r, r_y, r_inv, temp_cgbn, mod_order, mod_fp;
        evm_word_t scratch_pad;
        bn_t Gx, Gy, ResX, ResY, XY_x, XY_y; // for the point multiplication

        // calculate R_invert
        cgbn_load(arith._env, r, &sig.r);
        cgbn_load(arith._env, mod_order, &curve.Order);
        cgbn_load(arith._env, mod_fp, &curve.FP);

        // calculate r_y
        arith.cgbn_mul_mod(arith._env, temp_cgbn, r, r, mod_fp);
        arith.cgbn_mul_mod(arith._env,temp_cgbn, temp_cgbn, r, mod_fp);
        cgbn_add_ui32(arith._env, r_y, temp_cgbn, curve.B);
        cgbn_rem(arith._env, r_y, r_y, mod_fp);

        // find r_y using Tonelli–Shanks algorithm
        // beta = pow(xcubedaxb, (P+1)//4, P)
        cgbn_load(arith._env, temp_cgbn, &curve.FP);
        cgbn_add_ui32(arith._env, temp_cgbn, temp_cgbn, 1);
        cgbn_div_ui32(arith._env, temp_cgbn, temp_cgbn, 4);
        cgbn_modular_power(arith._env, r_y, r_y, temp_cgbn, mod_fp);

        // y = beta if v % 2 ^ beta % 2 else (P - beta)
        uint32_t beta_mod2 = cgbn_extract_bits_ui32(arith._env, r_y, 0, 1);
        uint32_t v_mod2 = sig.v % 2;
        if (beta_mod2 == v_mod2)
            cgbn_sub(arith._env, r_y, mod_fp, r_y);

        // calculate n_z = (N-msg_hash) mod N
        cgbn_load(arith._env, temp_cgbn, &sig.msg_hash);
        cgbn_sub(arith._env, temp_cgbn, mod_order, temp_cgbn);

        cgbn_load(arith._env, Gx, &curve.GX);
        cgbn_load(arith._env, Gy, &curve.GY);

        ec_mul(arith, curve, ResX, ResY, Gx, Gy, temp_cgbn);

        // calculate XY = (r, r_y) * s
        cgbn_load(arith._env, temp_cgbn , &sig.s);
        ec_mul(arith, curve, XY_x, XY_y, r, r_y, temp_cgbn);

        // calculate QR = (ResX, ResY) + (XY_x, XY_y)
        ec_add(arith, curve, ResX, ResY, ResX, ResY, XY_x, XY_y);
        cgbn_modular_inverse(arith._env, r_inv, r, mod_order);

        // env_t::cgbn_t Q_x, Q_y;
        //calculate Q = Qr * r_inv %N
        ec_mul(arith, curve, ResX, ResY, ResX, ResY, r_inv);

        convert_point_to_address(arith, keccak, signer, ResX, ResY);

    }
}  // namespace ecc

#endif