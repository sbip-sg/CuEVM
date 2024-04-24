// Eliptic curve utilities using CGBN
#ifndef _ECC_H_
#define _ECC_H_
#include "include/utils.h"
#include "keccak.cuh"
/// The secp256k1 field prime number (P) and order
inline constexpr const char *secp256k1_FieldPrime = "0xfffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f";
inline constexpr const char *secp256k1_Order = "0xfffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141";

inline constexpr const char *secp256k1_GX = "0x79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798";
inline constexpr const char *secp256k1_GY = "0x483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8";

/// The alt_BN128 field prime number (P) and order
inline constexpr const char* alt_BN128_FieldPrime = "0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47";
inline constexpr const char* alt_BN128_Order = "0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001";

inline constexpr const char* alt_BN128_G2_B_0 = "0x2b149d40ceb8aaae81be18991be06ac3b5b4c5e559dbefa33267e6dc24a138e5";
inline constexpr const char* alt_BN128_G2_B_1 = "0x9713b03af0fed4cd2cafadeed8fdf4a74fa084e52d1852e4a2bd0685c315d2";

inline constexpr const char* alt_BN128_GX = "0x01";
inline constexpr const char* alt_BN128_GY = "0x02";

inline constexpr const char* alt_BN128_G2X1 = "0x1800deef121f1e76426a00665e5c4479674322d4f75edadd46debd5cd992f6ed";
inline constexpr const char* alt_BN128_G2X2 = "0x198e9393920d483a7260bfb731fb5d25f1aa493335a9e71297e485b7aef312c2";
inline constexpr const char* alt_BN128_G2Y1 = "0x12c85ea5db8c6deb4aab71808dcb408fe3d1e7690c43d37b4ce6cc0166fa7daa";
inline constexpr const char* alt_BN128_G2Y2 = "0x90689d0585ff075ec9e99ad690c3395bc4b313370b38ef355acdadcd122975b";
inline constexpr const char* ate_loop_count_hex = "0x19d797039be763ba8";
constexpr size_t log_ate_loop_count = 63;

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
        // Constructor that accepts a curve name
    } Curve;

    __host__ __device__ static const int32_t FQ2_mod_coeffs[] = {1,0};
    __host__ __device__ static const int32_t FQ1_mod_coeffs[] = {1};
    __host__ __device__ static const int32_t FQ12_mod_coeffs[] = {82, 0, 0, 0, 0, 0, -18, 0, 0, 0, 0, 0};

    template <size_t Degree>
    struct FQ {
        bn_t coeffs[Degree];
    };

    // Constexpr function to return the appropriate modulus coefficients array based on the template parameter Degree
    template<size_t Degree>
    __host__ __device__ __forceinline__ constexpr const int32_t* get_modulus_coeffs();

    template<>
    __host__ __device__ __forceinline__ constexpr const int32_t* get_modulus_coeffs<2>() {
        return FQ2_mod_coeffs;
    }
    template<>
    __host__ __device__ __forceinline__ constexpr const int32_t* get_modulus_coeffs<1>() {
        return FQ1_mod_coeffs;
    }
    template<>
    __host__ __device__ __forceinline__ constexpr const int32_t* get_modulus_coeffs<12>() {
        return FQ12_mod_coeffs;
    }

    /**
     * @brief Get the curve object
     *
     * @param arith
     * @param curve_id 256 for secp256k1, 128 for alt_BN128
     * @return Curve
     */
    __host__ __device__ __forceinline__ Curve get_curve(arith_t &arith, int curve_id) {
        Curve curve;
        if (curve_id == 256) {
            arith.cgbn_memory_from_hex_string(curve.FP, secp256k1_FieldPrime);
            arith.cgbn_memory_from_hex_string(curve.Order, secp256k1_Order);
            arith.cgbn_memory_from_hex_string(curve.GX, secp256k1_GX);
            arith.cgbn_memory_from_hex_string(curve.GY, secp256k1_GY);
            curve.B = 7;
        } else if (curve_id == 128) {
            arith.cgbn_memory_from_hex_string(curve.FP, alt_BN128_FieldPrime);
            arith.cgbn_memory_from_hex_string(curve.Order, alt_BN128_Order);
            arith.cgbn_memory_from_hex_string(curve.GX, "0x1");
            arith.cgbn_memory_from_hex_string(curve.GY, "0x2");
            curve.B = 3;
        }
        return curve;
    }



    __host__ __device__ __forceinline__ void cgbn_mul_mod(env_t env, bn_t &res, bn_t &a, bn_t &b, bn_t &mod) {
        env_t::cgbn_wide_t temp;
        cgbn_mul_wide(env, temp, a, b);
        cgbn_rem_wide(env, res, temp, mod);
    }
    __host__ __device__ __forceinline__ void cgbn_add_mod(env_t env, bn_t &res, bn_t &a, bn_t &b, bn_t &mod) {
        int32_t carry = cgbn_add(env, res, a, b);
        env_t::cgbn_wide_t d;
        if (carry == 1)
        {
            cgbn_set_ui32(env, d._high, 1);
            cgbn_set(env, d._low, res);
            cgbn_rem_wide(env, res, d, mod);
        }
        else
        {
            cgbn_rem(env, res, res, mod);
        }
    }
    __host__ __device__ __forceinline__ void cgbn_sub_mod(env_t env, bn_t &res, bn_t &a, bn_t &b, bn_t &mod) {
        // if b > a then a - b + mod
        if (cgbn_compare(env, a, b) < 0) {
            env_t::cgbn_accumulator_t acc;
            cgbn_set(env, acc, a);
            cgbn_add(env, acc, mod);
            cgbn_sub(env, acc, b);
            cgbn_resolve(env, res, acc);
            cgbn_rem(env, res, res, mod);
        } else{
            cgbn_sub(env, res, a, b);
        }
    }

    /**
     * @brief helper div_mod for bn_t
     *
     * @param env
     * @param res
     * @param a
     * @param b
     * @param mod
     */
    __host__ __device__ __forceinline__ void cgbn_div_mod(env_t env, bn_t &res, bn_t &a, bn_t &b, bn_t &mod) {
        cgbn_modular_inverse(env, res, b, mod);
        cgbn_mul_mod(env, res, a, res, mod);
    }

    /**
     * @brief Check if a basic point P(x, y) is on the curve y^2 = x^3 + B
     *
     * @param env
     * @param Px
     * @param Py
     * @param mod
     * @param B
     * @return true if on curve
     */
    __host__ __device__ __forceinline__ bool is_on_cuve_simple(env_t env, bn_t& Px, bn_t &Py, bn_t& mod, uint32_t B){
        if (cgbn_equals_ui32(env, Px, 0) && cgbn_equals_ui32(env, Py, 0))
            return true;
        bn_t temp, temp2;
        cgbn_mul_mod(env, temp, Px, Px, mod); // temp = Px^2
        cgbn_mul_mod(env, temp, temp, Px, mod); // temp = Px^3
        cgbn_add_ui32(env, temp, temp, B);
        cgbn_rem(env, temp, temp, mod);
        cgbn_mul_mod(env, temp2, Py, Py, mod); // temp2 = Py^2
        return cgbn_equals(env, temp, temp2);
        }

    template <size_t Degree>
    __host__ __device__ __forceinline__ bool FQP_equals(arith_t& arith, FQ<Degree> &P1, FQ<Degree> &P2){
        for (size_t i = 0; i < Degree; i++) {
            if (!cgbn_equals(arith._env, P1.coeffs[i], P2.coeffs[i])){
            return false;
            }
        }
        return true;
    }

    // Add two point on the curve P and Q
    __host__ __device__ __forceinline__ int ec_add(arith_t& arith, Curve curve, bn_t &ResX, bn_t &ResY, bn_t &Px, bn_t &Py, bn_t &Qx, bn_t &Qy) {
        bn_t mod_fp;
        bn_t lambda, numerator, denominator, temp, x_r, y_r;
        evm_word_t scratch_pad;
        cgbn_load(arith._env, mod_fp, &curve.FP);
        // point at infinity
        if (cgbn_equals_ui32(arith._env, Px, 0) && cgbn_equals_ui32(arith._env, Py, 0)) {
            if (is_on_cuve_simple(arith._env, Qx, Qy, mod_fp, curve.B) ){
                cgbn_set(arith._env, ResX, Qx);
                cgbn_set(arith._env, ResY, Qy);
                return 0;
            } else
                return -1;
        } else if (cgbn_equals_ui32(arith._env, Qx, 0) && cgbn_equals_ui32(arith._env, Qy, 0)) {
            if (is_on_cuve_simple(arith._env, Px, Py, mod_fp, curve.B) ){
                cgbn_set(arith._env, ResX, Px);
                cgbn_set(arith._env, ResY, Py);
                return 0;
            } else
                return -1;
        }
        if (!is_on_cuve_simple(arith._env, Px, Py, mod_fp, curve.B) || !is_on_cuve_simple(arith._env, Qx, Qy, mod_fp, curve.B)) {
            return -1;
        }
        if (cgbn_equals(arith._env, Px, Qx) && cgbn_equals(arith._env, Py, Qy)) {
            // Special case for doubling P == Q
            // printf("Doubling\n");
            // lambda = (3*Px^2) / (2*Py)
            cgbn_mul_mod(arith._env, temp, Px, Px, mod_fp);  // temp = Px^2
            cgbn_set_ui32(arith._env, numerator, 3);
            cgbn_mul_mod(arith._env, numerator, numerator, temp, mod_fp);  // numerator = 3*Px^2

            cgbn_set_ui32(arith._env, denominator, 2);
            cgbn_mul_mod(arith._env, denominator, denominator, Py, mod_fp);  // denominator = 2*Py
            cgbn_modular_inverse(arith._env, denominator, denominator, mod_fp);

            cgbn_mul_mod(arith._env, lambda, numerator, denominator, mod_fp);  // lambda = (3*Px^2) / (2*Py)
            // print lambda
            cgbn_store(arith._env, &scratch_pad, lambda);

        } else if (cgbn_equals(arith._env, Px, Qx) ){
            // printf("Doubling\n");
            // Special case for P != Q and Px == Qx
            // The result is the point at infinity
            cgbn_set_ui32(arith._env, ResX, 0);
            cgbn_set_ui32(arith._env, ResY, 0);
            return 0;
        } else {
            // printf("Adding\n");
            // General case for P != Q
            // lambda = (Qy - Py) / (Qx - Px)
            cgbn_sub_mod(arith._env, temp, Qy, Py, mod_fp);       // temp = Qy - Py
            cgbn_sub_mod(arith._env, numerator, Qx, Px, mod_fp);  // numerator = Qx - Px
            cgbn_modular_inverse(arith._env, numerator, numerator, mod_fp);
            cgbn_mul_mod(arith._env, lambda, temp, numerator, mod_fp);  // lambda = (Qy - Py) / (Qx - Px)
        }

        cgbn_mul_mod(arith._env, x_r, lambda, lambda, mod_fp);  // x_r = lambda^2
        cgbn_add_mod(arith._env, temp, Px, Qx, mod_fp);         // temp = Px + Qx
        cgbn_sub_mod(arith._env, x_r, x_r, temp, mod_fp);       // x_r = lambda^2 - (Px + Qx)
        // y_r = lambda * (Px - x_r) - Py
        cgbn_sub_mod(arith._env, temp, Px, x_r, mod_fp);      // temp = Px - x_r
        cgbn_mul_mod(arith._env, y_r, lambda, temp, mod_fp);  // y_r = lambda * (Px - x_r)
        cgbn_sub_mod(arith._env, y_r, y_r, Py, mod_fp);       // y_r = lambda * (Px - x_r) - Py
        // Set the result
        cgbn_set(arith._env, ResX, x_r);
        cgbn_set(arith._env, ResY, y_r);
        return 0;
    }
    // Multiply a point on the curve G by a scalar n, store result in Res
    __host__ __device__ __forceinline__ int ec_mul(arith_t &arith, Curve curve, bn_t &ResX, bn_t &ResY, bn_t &Gx, bn_t &Gy, bn_t &n) {
        bn_t mod_fp;
        evm_word_t scratch_pad;
        cgbn_load(arith._env, mod_fp, &curve.FP);

        if(!is_on_cuve_simple(arith._env, Gx, Gy, mod_fp, curve.B)){
            printf("Point not on curve\n");
            return -1;
        }
        // check point at infinity
        if (cgbn_equals_ui32(arith._env, Gx, 0) && cgbn_equals_ui32(arith._env, Gy, 0) || cgbn_equals_ui32(arith._env, n, 0)) {
            cgbn_set_ui32(arith._env, ResX, 0);
            cgbn_set_ui32(arith._env, ResY, 0);
            return 0;
        }


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
        return 0;
    }

    __host__ __device__ __forceinline__ void convert_point_to_address(arith_t &arith, keccak::keccak_t &keccak, bn_t& address, bn_t &X, bn_t &Y){

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
        //cgbn_bitwise_mask_and(arith._env, address, address, 160);
        arith.address_conversion(address);
    }


    /**
     * Recover signer from the signature with the above structure. return the signer address in cgbn type
     *
     * @param arith
     * @param keccak
     * @param sig
     * @param signer
     */
    __host__ __device__ __forceinline__ int ec_recover(arith_t &arith, keccak::keccak_t &keccak, signature_t sig,  bn_t &signer){

        Curve  curve;
        arith.cgbn_memory_from_hex_string(curve.FP, secp256k1_FieldPrime);
        arith.cgbn_memory_from_hex_string(curve.Order, secp256k1_Order);
        arith.cgbn_memory_from_hex_string(curve.GX, secp256k1_GX);
        arith.cgbn_memory_from_hex_string(curve.GY, secp256k1_GY);
        curve.B = 7;
        if (sig.v < 27 || sig.v > 28) {
            return -1;
        }

        bn_t  r, r_y, r_inv, temp_cgbn, mod_order, mod_fp, temp_compare;
        bn_t Gx, Gy, ResX, ResY, XY_x, XY_y; // for the point multiplication

        // calculate R_invert
        cgbn_load(arith._env, r, &sig.r);
        cgbn_load(arith._env, mod_order, &curve.Order);
        cgbn_load(arith._env, mod_fp, &curve.FP);

        cgbn_rem(arith._env, temp_compare, r, mod_order);
        if (cgbn_equals_ui32(arith._env, temp_compare, 0) || cgbn_compare(arith._env, r, mod_order) >= 0)
            return -1;

        // calculate r_y
        cgbn_mul_mod(arith._env, temp_cgbn, r, r, mod_fp);
        cgbn_mul_mod(arith._env,temp_cgbn, temp_cgbn, r, mod_fp);
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

        // invalid point check
        if (!is_on_cuve_simple(arith._env, r, r_y, mod_fp, curve.B))
            return -1;

        // calculate n_z = (N-msg_hash) mod N
        cgbn_load(arith._env, temp_cgbn, &sig.msg_hash);
        cgbn_sub(arith._env, temp_cgbn, mod_order, temp_cgbn);

        cgbn_load(arith._env, Gx, &curve.GX);
        cgbn_load(arith._env, Gy, &curve.GY);

        ec_mul(arith, curve, ResX, ResY, Gx, Gy, temp_cgbn);

        // calculate XY = (r, r_y) * s
        cgbn_load(arith._env, temp_cgbn , &sig.s);
        // check invalid s
        cgbn_rem(arith._env, temp_compare, temp_cgbn, mod_order);
        if (cgbn_equals_ui32(arith._env, temp_compare, 0) || cgbn_compare(arith._env, temp_cgbn, mod_order) >= 0)
            return -1;

        ec_mul(arith, curve, XY_x, XY_y, r, r_y, temp_cgbn);

        // calculate QR = (ResX, ResY) + (XY_x, XY_y)
        ec_add(arith, curve, ResX, ResY, ResX, ResY, XY_x, XY_y);
        cgbn_modular_inverse(arith._env, r_inv, r, mod_order);

        // env_t::cgbn_t Q_x, Q_y;
        //calculate Q = Qr * r_inv %N
        ec_mul(arith, curve, ResX, ResY, ResX, ResY, r_inv);

        convert_point_to_address(arith, keccak, signer, ResX, ResY);
        return 0;
    }

    /**
     * @brief Get the FQ12 from bn_t
     *
     * @tparam Degree
     * @param env
     * @param res
     */
    template <size_t Degree>
    __host__ __device__ __forceinline__ void getFQ12_from_cgbn_t(arith_t &arith, FQ<Degree> &res, bn_t (&coeffs)[Degree]){
    for (uint32_t i = 0; i < Degree; i++) {
        cgbn_set(arith._env, res.coeffs[i], coeffs[i]);
    }
    }

    /**
     * @brief Addition in FQP
     *
     * @tparam Degree
     * @param env
     * @param Res
     * @param P1
     * @param P2
     * @param mod
     */
    template <size_t Degree>
    __host__ __device__ __forceinline__ void FQP_add(arith_t &arith, FQ<Degree>& Res, FQ<Degree>& P1, FQ<Degree>& P2, bn_t& mod) {
        for (size_t i = 0; i < Degree; i++) {
            cgbn_add_mod(arith._env, Res.coeffs[i], P1.coeffs[i], P2.coeffs[i], mod);
        }
    }

    /**
     * @brief Substraction in FQP
     *
     * @tparam Degree
     * @param env
     * @param Res
     * @param P1
     * @param P2
     * @param mod
     */
    template <size_t Degree>
    __host__ __device__ __forceinline__ void FQP_sub(arith_t &arith, FQ<Degree>& Res, FQ<Degree>& P1, FQ<Degree>& P2, bn_t& mod) {
        for (size_t i = 0; i < Degree; i++) {
            // printf("P1[%d]: %s\n", i, bnt_to_string(env, P1.coeffs[i]));
            // printf("P2[%d]: %s\n", i, bnt_to_string(env, P2.coeffs[i]));
            // printf("mod: %s\n", bnt_to_string(env, mod));
            // printf("compare %d\n", cgbn_compare(env, P1.coeffs[i], P2.coeffs[i]));
            cgbn_sub_mod(arith._env, Res.coeffs[i], P1.coeffs[i], P2.coeffs[i], mod);
        }
    }

    /**
     * @brief Degree of a polynomial
     *
     * @tparam Degree
     * @param env
     * @param P
     * @return uint
     */
    template <size_t Degree>
    __host__ __device__ __forceinline__ uint deg(arith_t &arith, const FQ<Degree>& P){
        uint res = Degree - 1;
        while (cgbn_equals_ui32(arith._env, P.coeffs[res],0) && res)
            res -= 1;
        return res;
    }

    /**
     * @brief Get the 1 element in FQ
     *
     * @tparam Degree
     * @param env
     * @return FQ<Degree>
     */
    template <size_t Degree>
    __host__ __device__ __forceinline__ FQ<Degree> get_one(arith_t &arith){
        FQ<Degree> res;
        cgbn_set_ui32(arith._env, res.coeffs[0], 1);
        return res;
    }

    /**
     * @brief Helper function poly_rounded_div for inverse
     *
     * @tparam Degree
     * @param env
     * @param Res
     * @param A
     * @param B
     * @param mod
     */
    template <size_t Degree>
    __host__ __device__ __forceinline__ void poly_rounded_div(arith_t &arith, FQ<Degree>& Res, FQ<Degree>& A, FQ<Degree>& B, bn_t& mod) {
        uint dega = deg(arith, A);
        uint degb = deg(arith, B);
        FQ<Degree> temp;
        for (size_t i = 0; i < Degree; i++) {
            cgbn_set(arith._env, temp.coeffs[i], A.coeffs[i]);
            cgbn_set_ui32(arith._env, Res.coeffs[i], 0);
        }

        FQ<Degree> o;
        bn_t temp_res;
        for (int32_t i = dega - degb; i >= 0; i--) {
            // o[i] += temp[degb + i] / b[degb]
            cgbn_div_mod(arith._env, temp_res, temp.coeffs[degb + i], B.coeffs[degb], mod);
            cgbn_add_mod(arith._env, o.coeffs[i], o.coeffs[i], temp_res, mod);
            for (size_t c = 0; c <= degb; c++) {
                cgbn_sub_mod(arith._env, temp.coeffs[c + i], temp.coeffs[c + i], o.coeffs[c], mod);
            }
        }
        // return o[:deg(o)+1]
        for (size_t i = 0; i < Degree; i++) {
            cgbn_set(arith._env, Res.coeffs[i], o.coeffs[i]);
        }
    }

    /**
     * @brief Helper FQP copy
     *
     * @tparam Degree
     * @param env
     * @param Res
     * @param P
     */
    template <size_t Degree>
    __host__ __device__ __forceinline__ void FQP_copy(arith_t &arith, FQ<Degree>& Res, FQ<Degree>& P){
        for (size_t i = 0; i < Degree; i++) {
            cgbn_set(arith._env, Res.coeffs[i], P.coeffs[i]);
        }
    }


    /**
     * @brief Multiplication in FQP Res = P1*P2
     *
     * @tparam Degree
     * @param env
     * @param Res
     * @param P1
     * @param P2
     * @param mod
     */
    template <size_t Degree>
    __host__ __device__ __forceinline__ void FQP_mul(arith_t &arith, FQ<Degree> &Res, FQ<Degree> &P1, FQ<Degree> &P2, bn_t& mod) {
        if (Degree == 1) {
            cgbn_mul_mod(arith._env, Res.coeffs[0], P1.coeffs[0], P2.coeffs[0], mod);
            return;
        }
        bn_t b[2 * Degree - 1];

        bn_t temp; // Temporary variable for intermediate results

        // set val
        // Polynomial multiplication
        for (size_t i = 0; i < Degree; i++) {
            for (size_t j = 0; j < Degree; j++) {
                cgbn_mul_mod(arith._env, temp, P1.coeffs[i], P2.coeffs[j], mod); // Multiply coefficients
                cgbn_add_mod(arith._env, b[i + j], b[i + j], temp, mod); // Add to the corresponding position and mod
            }
        }

        bn_t mod_coeffs[Degree];
        const int32_t* mod_coeffs_array = get_modulus_coeffs<Degree>();
        for (size_t i = 0; i < Degree; i++) {
            if (mod_coeffs_array[i] < 0) {
                // printf("negative mod coeffs\n");
                cgbn_sub_ui32(arith._env, mod_coeffs[i], mod, 0 - mod_coeffs_array[i]);
            } else {
                cgbn_set_ui32(arith._env, mod_coeffs[i], mod_coeffs_array[i]);
            }
        }
        for (size_t len_b = 2 * Degree - 1; len_b > Degree; len_b--) {
            size_t exp = len_b - Degree - 1;
            for (size_t i = 0; i < Degree; i++) {
                // Assuming FQ is a function that takes an int and returns a bn_t type and modulus_coeffs is accessible
                // b[exp + i] -= top * FQ(modulus_coeffs[i]);
                cgbn_mul_mod(arith._env, temp, b[len_b - 1], mod_coeffs[i], mod);
                cgbn_sub_mod(arith._env, b[exp + i], b[exp + i], temp, mod);
            }
        }
        // Copy the result back to Res, adjusting to the actual degree
        for (size_t i = 0; i < Degree; i++) {
            cgbn_set(arith._env, Res.coeffs[i], b[i]);
        }
    }

    /**
     * @brief Inverse in Field Extension
     *
     * @tparam Degree
     * @param env
     * @param Res
     * @param P
     * @param mod
     */
    template <size_t Degree>
    __host__ __device__ __forceinline__ void FQP_inv(arith_t &arith, FQ<Degree>& Res, FQ<Degree>& P, bn_t& mod) {
        if (Degree == 1){
            cgbn_modular_inverse(arith._env, Res.coeffs[0], P.coeffs[0], mod);
            return;
        }
        FQ<Degree+1> lm, hm, low, high;
        bn_t temp;
        // lm[0] = 1;
        cgbn_set_ui32(arith._env, lm.coeffs[0], 1);
        // set low,high
        // Initialize high with modulus coefficients + [1]
        const int32_t* mod_coeffs = get_modulus_coeffs<Degree>();

        for (size_t i = 0; i < Degree; i++) {
            cgbn_set(arith._env, low.coeffs[i], P.coeffs[i]);
            if (mod_coeffs[i] < 0) {
                cgbn_sub_ui32(arith._env, high.coeffs[i], mod, 0 - mod_coeffs[i]);
            } else {
            cgbn_set_ui32(arith._env, high.coeffs[i], mod_coeffs[i]);
            }
        }
        cgbn_set_ui32(arith._env, high.coeffs[Degree], 1);

        while (deg(arith, low) > 0) { // Assuming a function `deg` to find the polynomial's degree
            FQ<Degree+1> r;
            poly_rounded_div(arith, r, high, low, mod); // Needs to return FQ<Degree+1> or similar

            FQ<Degree+1> nm, new_h;
            FQP_copy(arith, nm, hm);
            FQP_copy(arith, new_h, high);
            for (size_t i = 0; i <= Degree; ++i) {
                for (size_t j = 0; j <= Degree - i; ++j) {

                    // nm[i+j] -= lm[i] * r[j]
                    cgbn_mul_mod(arith._env, temp, lm.coeffs[i], r.coeffs[j], mod);
                    cgbn_sub_mod(arith._env, nm.coeffs[i+j], nm.coeffs[i+j], temp, mod);

                    // new[i+j] -= low[i] * r[j]
                    cgbn_mul_mod(arith._env, temp, low.coeffs[i], r.coeffs[j], mod);
                    cgbn_sub_mod(arith._env, new_h.coeffs[i+j], new_h.coeffs[i+j], temp, mod);

                }
            }
            // lm, low, hm, high = nm, new, lm, low
            FQP_copy(arith, high, low);
            FQP_copy(arith, hm, lm);
            FQP_copy(arith, low, new_h);
            FQP_copy(arith, lm, nm);

        }

        // // self.__class__(lm[:self.degree]) / low[0]
        for (size_t i = 0; i < Degree; i++) {
            cgbn_set(arith._env, Res.coeffs[i], lm.coeffs[i]);
        }
        cgbn_modular_inverse(arith._env, temp, low.coeffs[0], mod);
        for (size_t i = 0; i < Degree; i++) {
            cgbn_mul_mod(arith._env, Res.coeffs[i], Res.coeffs[i], temp, mod);
        }
    }

    /**
     * @brief Division in field extension
     *
     * @tparam Degree
     * @param env
     * @param Res
     * @param P1
     * @param P2
     * @param mod
     */
    template <size_t Degree>
    __host__ __device__ __forceinline__ void FQP_div(arith_t &arith, FQ<Degree> &Res, FQ<Degree> &P1, FQ<Degree> &P2, bn_t& mod){ // P1/P2
        FQP_inv(arith, Res, P2, mod);
        FQP_mul(arith, Res, P1, Res, mod);
    }

    /**
     * @brief Calcualte negation of a point in field extension
     *
     * @tparam Degree
     * @param env
     * @param Res
     * @param P
     * @param mod
     */
    template <size_t Degree>
    __host__ __device__ __forceinline__ void FQP_neg(arith_t &arith, FQ<Degree> &Res, FQ<Degree> &P, bn_t& mod){
        for (size_t i = 0; i < Degree; i++) {
            cgbn_sub_mod(arith._env, Res.coeffs[i], mod, P.coeffs[i], mod);
        }
    }

    /**
     * @brief Calculate Res = P^n in field extension
     *
     * @tparam Degree
     * @param env
     * @param Res
     * @param P
     * @param n
     * @param mod
     */
    template <size_t Degree>
    __host__ __device__ __forceinline__ void FQP_pow(arith_t &arith, FQ<Degree> &Res, FQ<Degree> &P, bn_t &n, bn_t &mod) {
        if (Degree == 1){
            cgbn_modular_power(arith._env, Res.coeffs[0], P.coeffs[0], n, mod);
            return;
        }
        bn_t temp_n;
        FQ<Degree> temp;
        FQP_copy(arith, temp, P);
        cgbn_set(arith._env, temp_n, n);
        // TODO, may not be necessary
        for (size_t i = 0; i < Degree; i++) {
            cgbn_set_ui32(arith._env, Res.coeffs[i], 0);
        }
        cgbn_set_ui32(arith._env, Res.coeffs[0], 1);
        while(cgbn_get_ui32(arith._env, temp_n) != 0) {
            // printf("%d \n", cgbn_extract_bits_ui32(env, temp_n, 0, 1));
            if (cgbn_extract_bits_ui32(arith._env, temp_n, 0, 1) == 1) {
                FQP_mul(arith, Res, Res, temp, mod);
            }
            FQP_mul(arith, temp, temp, temp, mod);
            cgbn_shift_right(arith._env, temp_n, temp_n, 1);
        }
    }

    /**
     * @brief Helper function: Mul FQP with a scalar, store result in Res
     *
     * @tparam Degree
     * @param env
     * @param Res
     * @param P
     * @param n
     * @param mod
     */
    template <size_t Degree>
    __host__ __device__ __forceinline__ void FQP_mul_scalar(arith_t &arith, FQ<Degree> &Res, FQ<Degree> &P, bn_t &n, bn_t &mod) {
        for (size_t i = 0; i < Degree; i++) {
            cgbn_mul_mod(arith._env, Res.coeffs[i], P.coeffs[i], n, mod);
        }
    }

    /**
     * @brief Check P(x, y) is on the curve y^2 = x^3 + B
     * with cordinate x, y and B are in field extension with Degree
     *
     * @tparam Degree
     * @param env
     * @param Px
     * @param Py
     * @param mod
     * @param B
     * @return true
     * @return false
     */
    template <size_t Degree>
    __host__ __device__ __forceinline__ bool FQP_is_on_curve(arith_t &arith, FQ<Degree> &Px, FQ<Degree> &Py, bn_t& mod, FQ<Degree> &B){
        // y^2 = x^3 + B
        FQ<Degree> temp, temp2;
        FQP_mul(arith, temp, Px, Px, mod);
        FQP_mul(arith, temp, temp, Px, mod);
        FQP_add(arith, temp, temp, B, mod);
        FQP_mul(arith, temp2, Py, Py, mod);

        return FQP_equals(arith, temp, temp2);
    }

    template <size_t Degree>
    __host__ __device__ __forceinline__ bool FQP_is_inf(arith_t &arith, FQ<Degree> &Px, FQ<Degree> &Py){
        bool res = true;
        for (size_t i = 0; i < Degree; i++) {
            res = res && cgbn_equals_ui32(arith._env, Px.coeffs[i], 0) && cgbn_equals_ui32(arith._env, Py.coeffs[i], 0);
        }
        return res;
    }
    /**
     * @brief Point Addition in field extension with Degree
     *  Res = P + Q
     * @tparam Degree
     * @param env
     * @param ResX
     * @param ResY
     * @param Px
     * @param Py
     * @param Qx
     * @param Qy
     * @param mod_fp
     */
    template <size_t Degree>
    __host__ __device__ __forceinline__ void FQP_ec_add(arith_t &arith, FQ<Degree> &ResX, FQ<Degree> &ResY, FQ<Degree> &Px, FQ<Degree> &Py, FQ<Degree> &Qx, FQ<Degree> &Qy, bn_t &mod_fp) {
        FQ<Degree> lambda, numerator, denominator, temp, x_r, y_r;
        bn_t two, three;
        cgbn_set_ui32(arith._env, two, 2);
        cgbn_set_ui32(arith._env, three, 3);
        // check infinity
        if (FQP_is_inf(arith, Px, Py)){
            FQP_copy(arith, ResX, Qx);
            FQP_copy(arith, ResY, Qy);
            return;
        } else if (FQP_is_inf(arith, Qx, Qy)){
            FQP_copy(arith, ResX, Px);
            FQP_copy(arith, ResY, Py);
            return;
        }

        if (FQP_equals(arith, Px, Qx) && FQP_equals(arith, Py, Qy)) {
            FQP_mul(arith, temp, Px, Px, mod_fp); // temp = Px^2
            FQP_mul_scalar(arith, numerator, temp, three, mod_fp); // numerator = 3*Px^2
            FQP_mul_scalar(arith, denominator, Py, two, mod_fp); // denominator = 2*Py
            FQP_div(arith, lambda, numerator, denominator, mod_fp); // lambda = (3*Px^2) / (2*Py)

        } else if (FQP_equals(arith, Px, Qx)){
            // special case, return inf
            for (size_t i = 0; i < Degree; i++) {
                cgbn_set_ui32(arith._env, ResX.coeffs[i], 0);
                cgbn_set_ui32(arith._env, ResY.coeffs[i], 0);
            }
        } else{

            FQP_sub(arith, numerator, Qy, Py, mod_fp); // temp = Qy - Py
            FQP_sub(arith, denominator, Qx, Px, mod_fp); // numerator = Qx - Px
            FQP_div(arith, lambda, numerator, denominator, mod_fp); // lambda = (Qy - Py) / (Qx - Px)
        }

        FQP_mul(arith, x_r, lambda, lambda, mod_fp); // x_r = lambda^2
        FQP_add(arith, temp, Px, Qx, mod_fp); // temp = Px + Qx
        FQP_sub(arith, x_r, x_r, temp, mod_fp); // x_r = lambda^2 - (Px + Qx)

        FQP_sub(arith, temp, Px, x_r, mod_fp); // temp = Px - x_r
        FQP_mul(arith, y_r, lambda, temp, mod_fp); // y_r = lambda * (Px - x_r)
        FQP_sub(arith, y_r, y_r, Py, mod_fp); // y_r = lambda * (Px - x_r) - Py

        // Set the result
        FQP_copy(arith, ResX, x_r);
        FQP_copy(arith, ResY, y_r);
    }

    /**
     * @brief ec_mul in field extension with Degree
     * Res = G*n
     * @tparam Degree
     * @param env
     * @param ResX
     * @param ResY
     * @param Gx Point to Mul
     * @param Gy Point to Mul
     * @param n  Mul scalar
     * @param mod_fp
     */
    template <size_t Degree>
    __host__ __device__ __forceinline__ void FQP_ec_mul(arith_t &arith, FQ<Degree> &ResX, FQ<Degree> &ResY, FQ<Degree> &Gx, FQ<Degree> &Gy, bn_t &n, bn_t &mod_fp) {
        uint8_t bitArray[evm_params::BITS];
        uint32_t bit_array_length = 0;

        evm_word_t scratch_pad;
        cgbn_store(arith._env, &scratch_pad, n);
        arith.bit_array_from_cgbn_memory(bitArray, bit_array_length, scratch_pad);

        // check point at infinity
        if (FQP_is_inf(arith, Gx, Gy)){
            FQP_copy(arith, ResX, Gx);
            FQP_copy(arith, ResY, Gy);
            return;
        }
        // mul zero
        if (cgbn_equals_ui32(arith._env, n, 0)){
            // return inf
            for (size_t i = 0; i < Degree; i++) {
                cgbn_set_ui32(arith._env, ResX.coeffs[i], 0);
                cgbn_set_ui32(arith._env, ResY.coeffs[i], 0);
            }
            return;
        }

        FQ<Degree> temp_ResX, temp_ResY;
        // Double-and-add algorithm
        FQP_copy(arith, temp_ResX, Gx);
        FQP_copy(arith, temp_ResY, Gy);

        for (int i = bit_array_length - 2; i >= 0; --i)
        {
            // Gz = 2 * Gz
            FQP_ec_add(arith, temp_ResX, temp_ResY, temp_ResX, temp_ResY, temp_ResX, temp_ResY, mod_fp);
            if (bitArray[evm_params::BITS-1-i])
            {
            // printf("bit1 add i %d\n", i);
            // Gz = Gz + P
            FQP_ec_add(arith, temp_ResX, temp_ResY, temp_ResX, temp_ResY, Gx, Gy, mod_fp);
            }
        }
        FQP_copy(arith, ResX, temp_ResX);
        FQP_copy(arith, ResY, temp_ResY);
    }


    /**
     * @brief Linefunc
     *
     * @tparam Degree
     * @param env
     * @param Res Result in FQ<Degree>
     * @param P1x Point 1 x
     * @param P1y Point 1 y
     * @param P2x Point 2 x
     * @param P2y Point 2 y
     * @param Tx  Point T x
     * @param Ty  Point T y
     * @param mod
     */
    template <size_t Degree>
    __host__ __device__ __forceinline__ void FQP_linefunc(arith_t &arith, FQ<Degree> &Res, FQ<Degree> &P1x, FQ<Degree> &P1y, FQ<Degree> &P2x, FQ<Degree> &P2y, FQ<Degree> &Tx, FQ<Degree> &Ty, bn_t& mod){
        FQ<Degree> m, temp, temp2;
        bn_t three, two;
        cgbn_set_ui32(arith._env, three, 3);
        cgbn_set_ui32(arith._env, two, 2);

        if (!FQP_equals(arith, P1x, P2x)){
            FQP_sub(arith, temp, P2y, P1y, mod);
            FQP_sub(arith, temp2, P2x, P1x, mod);
            FQP_div(arith, m, temp, temp2, mod);

            FQP_sub(arith, temp, Tx, P1x, mod);
            FQP_mul(arith, Res, m, temp, mod);
            FQP_sub(arith, temp, Ty, P1y, mod);
            FQP_sub(arith, Res, Res, temp, mod);
            return;
        }
        else if (FQP_equals(arith, P1y, P2y)){

            FQP_mul(arith, temp, P1x, P1x, mod);
            FQP_mul_scalar(arith, temp, temp, three, mod);
            FQP_mul_scalar(arith, temp2, P1y, two, mod);

            FQP_div(arith, m, temp, temp2, mod);

            FQP_sub(arith, temp, Tx, P1x, mod);
            FQP_mul(arith, Res, m, temp, mod);
            FQP_sub(arith, temp, Ty, P1y, mod);
            FQP_sub(arith, Res, Res, temp, mod);
            return;
        }
        else {
            FQP_sub(arith, Res, Tx, P1x, mod);
            return;
        }

    }

    /**
     * @brief Point Twist: # "Twist" a point in E(FQ2) into a point in E(FQ12)
     *
     * @param env
     * @param Rx FQ<12> type (12 FQ elements)
     * @param Ry FQ<12> type (12 FQ elements)
     * @param Px FQ<2> type (2 FQ elements)
     * @param Py FQ<2> type (2 FQ elements)
     * @param mod_fp
     */
    __host__ __device__ __forceinline__ void FQP_twist(arith_t &arith, FQ<12> &Rx, FQ<12> &Ry, FQ<2> &Px, FQ<2> &Py, bn_t &mod_fp){
        // # "Twist" a point in E(FQ2) into a point in E(FQ12)
        FQ<12> w;
        cgbn_set_ui32(arith._env, w.coeffs[1], 1);

        bn_t X[2];
        bn_t Y[2];
        bn_t temp;
        // xcoeffs = [_x.coeffs[0] - _x.coeffs[1] * 9, _x.coeffs[1]]
        cgbn_set_ui32(arith._env, temp, 9);
        cgbn_mul_mod(arith._env, X[0], Px.coeffs[1], temp, mod_fp);
        cgbn_sub_mod(arith._env, X[0], Px.coeffs[0], X[0], mod_fp);
        cgbn_set(arith._env, X[1], Px.coeffs[1]);

        // ycoeffs = [_y.coeffs[0] - _y.coeffs[1] * 9, _y.coeffs[1]]
        cgbn_mul_mod(arith._env, Y[0], Py.coeffs[1], temp, mod_fp);
        cgbn_sub_mod(arith._env, Y[0], Py.coeffs[0], Y[0], mod_fp);
        cgbn_set(arith._env, Y[1], Py.coeffs[1]);

        FQ<12> nx, ny, temp_fqp;
        //     nx = FQ12([xcoeffs[0]] + [0] * 5 + [xcoeffs[1]] + [0] * 5)
        //     ny = FQ12([ycoeffs[0]] + [0] * 5 + [ycoeffs[1]] + [0] * 5)
        cgbn_set(arith._env, nx.coeffs[0], X[0]);
        cgbn_set(arith._env, nx.coeffs[6], X[1]);
        cgbn_set(arith._env, ny.coeffs[0], Y[0]);
        cgbn_set(arith._env, ny.coeffs[6], Y[1]);

        //    return (nx * w **2, ny * w**3)
        // w**2
        FQP_mul(arith, temp_fqp, w, w, mod_fp);
        FQP_mul(arith, Rx, nx, temp_fqp, mod_fp);
        // w**3
        FQP_mul(arith, temp_fqp, temp_fqp, w, mod_fp);
        FQP_mul(arith, Ry, ny, temp_fqp, mod_fp);
    }

    /**
    * @brief Final Expoentiation step for Pairing
    * @param env
    * @param res FQ<12> type (12 FQ elements) result of the pairing
    * @param p FQ<12> type (12 FQ elements)
    * @param mod
    */
    template <size_t Degree>
    __host__ __device__ __forceinline__ void FQP_final_exponentiation(arith_t &arith, FQ<Degree> &res, FQ<Degree> &p, bn_t &mod) {
        evm_word_t temp_mem;
        size_t final_exp_len = 11;
        const char* final_exp[11] = {
            "0000002f4b6dc97020fddadf107d20bc842d43bf6369b1ff6a1c71015f3f7be2",
            "e1e30a73bb94fec0daf15466b2383a5d3ec3d15ad524d8f70c54efee1bd8c3b2",
            "1377e563a09a1b705887e72eceaddea3790364a61f676baaf977870e88d5c6c8",
            "fef0781361e443ae77f5b63a2a2264487f2940a8b1ddb3d15062cd0fb2015dfc",
            "6668449aed3cc48a82d0d602d268c7daab6a41294c0cc4ebe5664568dfc50e16",
            "48a45a4a1e3a5195846a3ed011a337a02088ec80e0ebae8755cfe107acf3aafb",
            "40494e406f804216bb10cf430b0f37856b42db8dc5514724ee93dfb10826f0dd",
            "4a0364b9580291d2cd65664814fde37ca80bb4ea44eacc5e641bbadf423f9a2c",
            "bf813b8d145da90029baee7ddadda71c7f3811c4105262945bba1668c3be69a3",
            "c230974d83561841d766f9c9d570bb7fbe04c7e8a6c3c760c0de81def35692da",
            "361102b6b9b2b918837fa97896e84abb40a4efb7e54523a486964b64ca86f120"};

        cgbn_set_ui32(arith._env, res.coeffs[0], 1);
        for (int i = 1 ; i < Degree; i++){
            cgbn_set_ui32(arith._env, res.coeffs[i], 0);
        }
        FQ<Degree> temp;

        FQP_copy(arith, temp, p);

        for (int i = final_exp_len - 1; i >= 0; i--) {
            bn_t temp_n;

            arith.cgbn_memory_from_hex_string(temp_mem, final_exp[i]);
            cgbn_load(arith._env, temp_n, &temp_mem);

            for (int j =0 ; j<256; j++){

                if (cgbn_extract_bits_ui32(arith._env, temp_n, 0, 1) == 1) {
                    // cgbn_mul_mod(env, res, res, temp, mod);
                    FQP_mul(arith, res, res, temp, mod);
                }
                // cgbn_mul_mod(env, temp, temp, temp, mod);
                FQP_mul(arith, temp, temp, temp, mod);
                cgbn_shift_right(arith._env, temp_n, temp_n, 1);
            }

        }
    }

    /**
    * @brief MillerLoop algorithm
    * @param env
    * @param Result FQ<12> type (12 FQ elements) result of the pairing
    * @params P, Q are in FQ<12>
    * @param mod_fp
    * @param curve_order
    * @param final_exp if true, do the final exponentiation
    */
    template<size_t Degree>
    __host__ __device__ __forceinline__ void miller_loop(arith_t &arith, FQ<Degree> &Result, FQ<Degree> &Qx, FQ<Degree> &Qy, FQ<Degree> &Px, FQ<Degree> &Py, bn_t &mod_fp, bn_t& curve_order, bool final_exp = true) {
        //  if Q is None or P is None: return FQ12.one()
        FQ<Degree> R_x, R_y, temp1, temp2;
        evm_word_t scratch_pad;
        bn_t ate_loop_count;
        arith.cgbn_memory_from_hex_string(scratch_pad, ate_loop_count_hex);
        cgbn_load(arith._env, ate_loop_count, &scratch_pad);
        // f = FQ12.one()
        FQ<Degree> f = get_one<Degree>(arith);
        // R = Q
        FQP_copy(arith, R_x, Qx);
        FQP_copy(arith, R_y, Qy);

        for (int i = log_ate_loop_count; i >= 0; i--) {
            // f = f * f * linefunc(R, R, P)
            FQP_mul(arith, temp1, f, f, mod_fp);
            // temp 2 = linefunc(R, R, P)
            FQP_linefunc(arith, temp2, R_x, R_y, R_x, R_y, Px, Py, mod_fp);
            // print_fqp(env, temp2, "linefunc");
            FQP_mul(arith, f, temp1, temp2, mod_fp);
            // // R = double(R)
            FQP_ec_add(arith, R_x, R_y, R_x, R_y, R_x, R_y, mod_fp);

            // // if ate_loop_count & (2**i):
            if (cgbn_extract_bits_ui32(arith._env, ate_loop_count, i, 1)) {
                // f = f * linefunc(R, Q, P)
                FQP_linefunc(arith, temp2, R_x, R_y, Qx, Qy, Px, Py, mod_fp);
                FQP_mul(arith, f, f, temp2, mod_fp);
                // R = add(R, Q)
                FQP_ec_add(arith, R_x, R_y, R_x, R_y, Qx, Qy, mod_fp);
            }

        }

        // Compute Q1 and nQ2, adapt for your representation of points
        FQ<Degree> Q1_x, Q1_y, nQ2_x, nQ2_y;
        // Q1 = (Q[0] ** field_modulus, Q[1] ** field_modulus)
        FQP_pow(arith, Q1_x, Qx, mod_fp, mod_fp);
        FQP_pow(arith, Q1_y, Qy, mod_fp, mod_fp);

        // nQ2 = (Q1[0] ** field_modulus, -Q1[1] ** field_modulus)
        FQP_pow(arith, nQ2_x, Q1_x, mod_fp, mod_fp);
        FQP_neg(arith, nQ2_y, Q1_y, mod_fp);
        FQP_pow(arith, nQ2_y, nQ2_y, mod_fp, mod_fp);

        // f = f * linefunc(R, Q1, P)
        FQP_linefunc(arith, temp1, R_x, R_y, Q1_x, Q1_y, Px, Py, mod_fp);
        FQP_mul(arith, f, f, temp1, mod_fp);

        // R = add(R, Q1)
        FQP_ec_add(arith, R_x, R_y, R_x, R_y, Q1_x, Q1_y, mod_fp);

        // f = f * linefunc(R, nQ2, P)
        FQP_linefunc(arith, temp1, R_x, R_y, nQ2_x, nQ2_y, Px, Py, mod_fp);
        FQP_mul(arith, f, f, temp1, mod_fp);

        if (final_exp)
            FQP_final_exponentiation(arith, Result, f, mod_fp);
        else
            FQP_copy(arith, Result, f);

    }

    /**
     * @brief Paring of P in G1 and Q in G2.
     * @param env
     * @param Res FQ<12> type (12 FQ elements) result of the pairing
     * @param Qx FQ<2> type (two FQ elements)
     * @param Qy FQ<2> type (two FQ elements)
     * @param Px FQ<1> type (one FQ element)
     * @param Py FQ<1> type (one FQ element)
     * @param mod_fp
     * @param curve_order
     * @param final_exp if true, do the final exponentiation
    */
    __host__ __device__ __forceinline__ void pairing(arith_t &arith, FQ<12> &Res, FQ<2> &Qx, FQ<2> &Qy, FQ<1> &Px, FQ<1> &Py, bn_t &mod_fp, bn_t &curve_order, bool final_exp = true){
        // assert is_on_curve(Q, b2) assert is_on_curve(P, b)
        // return miller_loop(twist(Q), cast_point_to_fq12(P))
        FQ<12> Qx_tw, Qy_tw, Px_fp12, Py_fp12;
        FQP_twist(arith, Qx_tw, Qy_tw, Qx, Qy, mod_fp);
        cgbn_set(arith._env,Px_fp12.coeffs[0], Px.coeffs[0]);
        cgbn_set(arith._env,Py_fp12.coeffs[0], Py.coeffs[0]);
        miller_loop(arith, Res, Qx_tw, Qy_tw, Px_fp12, Py_fp12, mod_fp, curve_order, final_exp);
    }

    /*
    * @brief Pairing function for BN128 curve
    * @param points_data points data in the format of Px, Py, Qx, Qy (2 + 4 evm_words)
    * @param data_len length of the points_data in bytes
    * @return 1 if the pairing result is 1, -1 if invalid inputs, 0 if failed.
    *
    */
    __host__ __device__ __forceinline__ int pairing_multiple(
        arith_t &arith,
        uint8_t* points_data,
        size_t data_len
    ) {
        bn_t curve_order, mod_fp;
        evm_word_t scratch_pad;
        // load curve order and mod_fp
        arith.cgbn_memory_from_hex_string(scratch_pad, alt_BN128_Order);
        cgbn_load(arith._env, curve_order, &scratch_pad);
        arith.cgbn_memory_from_hex_string(scratch_pad, alt_BN128_FieldPrime);
        cgbn_load(arith._env, mod_fp, &scratch_pad);
        FQ<2> Qx, Qy, B2;
        FQ<1> Px, Py, B1;

        // setup B1, B2
        cgbn_set_ui32(arith._env, B1.coeffs[0], 3);
        arith.cgbn_memory_from_hex_string(scratch_pad, alt_BN128_G2_B_0);
        cgbn_load(arith._env, B2.coeffs[0],&scratch_pad);
        arith.cgbn_memory_from_hex_string(scratch_pad, alt_BN128_G2_B_1);
        cgbn_load(arith._env, B2.coeffs[1],&scratch_pad);

        FQ<12> final_res = get_one<12>(arith);

        size_t num_pairs = data_len / 192; // 2 for G1, 4 for G2

        for (int i = 0; i<num_pairs; i++){
            FQ<12> temp_res;
            points_data += i*192;

            arith.cgbn_from_memory(Px.coeffs[0], points_data );
            arith.cgbn_from_memory(Py.coeffs[0], points_data + 32);
            // Important!!! X2 first then X1 for G2
            arith.cgbn_from_memory(Qx.coeffs[1], points_data + 64);
            arith.cgbn_from_memory(Qx.coeffs[0], points_data + 96);
            arith.cgbn_from_memory(Qy.coeffs[1], points_data + 128);
            arith.cgbn_from_memory(Qy.coeffs[0], points_data + 160);
            //print point for debugging
            //print_fqp(env, Px, "Px");
            //print_fqp(env, Py, "Py");
            //print_fqp(env, Qx, "Qx");
            //print_fqp(env, Qy, "Qy");
            bool on_curve = FQP_is_on_curve(arith, Px, Py, mod_fp, B1)  && FQP_is_on_curve(arith, Qx, Qy, mod_fp, B2);
            if (!on_curve){
                if (FQP_is_inf(arith, Qx, Qy) || FQP_is_inf(arith, Px, Py)){
                    FQ<12> one_fq12 = get_one<12>(arith);
                    FQP_copy(arith, temp_res, one_fq12);
                } else {
                    return -1;
                }
            } else {
                pairing(arith, temp_res, Qx, Qy, Px, Py, mod_fp, curve_order, false);
            }
                FQP_mul(arith, final_res, final_res, temp_res, mod_fp);

        }
        // final exp
        FQP_final_exponentiation(arith, final_res, final_res, mod_fp);
        printf("Final result: \n");
        //print_fqp(arith._env, final_res, "Final");
        FQ<12> one_fq12 = get_one<12>(arith);
        return FQP_equals(arith, final_res, one_fq12) ? 1 : 0;
    }
}  // namespace ecc

#endif