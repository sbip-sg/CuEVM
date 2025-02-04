// Eliptic curve utilities using CGBN
#include <CuEVM/utils/ecc.cuh>
/// The secp256k1 field prime number (P) and order

namespace ecc {
// only works for prime modulus
__device__ void uint256_modular_inverse(uint256 *res, uint256 *a, uint256 *mod) {
    // TODO : check this implementation
    // Optional: check that 'a' is not zero.
    // assume mod is prime without checking
    if (uint256_is_zero(a)) {
        // Inverse does not exist; here we choose to set the result to 0.
        uint256_set_zero(res);
        return;
    }
    uint256 exponent;
    // Copy mod into exponent, then subtract 2,
    // so that exponent = mod - 2.
    uint256_sub_word(&exponent, mod, 2);
    // Using Fermat's little theorem: a^(mod-2) mod mod is the
    // modular inverse of a (provided mod is prime).
    uint256_powmod(res, a, &exponent, mod);
}
__device__ void uint256_pow_mod(uint256 *res, uint256 *a, uint256 *b, uint256 *mod) {
    // printf("powmod\n");
    // print_uint256(a);
    // print_uint256(b);
    // print_uint256(mod);
    uint256_powmod(res, a, b, mod);
    // printf("powmod result\n");
    //  print_uint256(res);
}
__device__ void uint256_mul_mod(uint256 *res, uint256 *a, uint256 *b, uint256 *mod) { uint256_mulmod(res, a, b, mod); }
__device__ void uint256_add_mod(uint256 *res, uint256 *a, uint256 *b, uint256 *mod) { uint256_addmod(res, a, b, mod); }
__device__ void uint256_sub_mod(uint256 *res, uint256 *a, uint256 *b, uint256 *mod) { uint256_submod(res, a, b, mod); }

/**
 * @brief helper div_mod for bn_t
 *
 * @param env
 * @param res
 * @param a
 * @param b
 * @param mod
 */
// __device__ void uint256_div_mod(uint256 *res, uint256 *a, uint256 *b, uint256 *mod) {
//     uint256 temp;
//     uint256_div(res, a, b);  // modular inverse
//     uint256_mod(res, res, mod);
//     // cgbn_modular_inverse(env, res, b, mod);
//     // cgbn_mul_mod(env, res, a, res, mod);
// }

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
__device__ bool is_on_cuve_simple(uint256 *Px, uint256 *Py, uint256 *mod, uint32_t B) {
    // printf("is_on_cuve_simple\n");
    // printf("Px\n");
    // print_uint256(Px);
    // printf("Py\n");
    // print_uint256(Py);
    // printf("mod\n");
    // print_uint256(mod);
    // printf("B %d\n", B);
    if (uint256_is_zero(Px) && uint256_is_zero(Py)) return true;
    uint256 temp, temp2;
    uint256_mul_mod(&temp, Px, Px, mod);     // temp = Px^2
    uint256_mul_mod(&temp, &temp, Px, mod);  // temp = Px^3
    uint256_add_word(&temp, &temp, B);
    uint256_mod(&temp, &temp, mod);
    uint256_mul_mod(&temp2, Py, Py, mod);  // temp2 = Py^2
    // printf("temp\n");
    // print_uint256(&temp);
    // printf("temp2\n");
    // print_uint256(&temp2);
    return uint256_cmp(&temp, &temp2) == 0;
}

// Add two point on the curve P and Q
__device__ int ec_add(Curve curve, evm_word_t *ResX, evm_word_t *ResY, evm_word_t *Px, evm_word_t *Py, evm_word_t *Qx,
                      evm_word_t *Qy, bool check_curve) {
    evm_word_t mod_fp;
    evm_word_t lambda, numerator, denominator, temp, x_r, y_r;
    mod_fp = curve.FieldPrime;
    // point at infinity
    if (uint256_is_zero(Px) && uint256_is_zero(Py)) {
        if (is_on_cuve_simple(Qx, Qy, &mod_fp, curve.B)) {
            *ResX = *Qx;
            *ResY = *Qy;
            return 0;
        } else
            return -1;
    } else if (uint256_is_zero(Qx) && uint256_is_zero(Qy)) {
        if (is_on_cuve_simple(Px, Py, &mod_fp, curve.B)) {
            *ResX = *Px;
            *ResY = *Py;
            return 0;
        } else
            return -1;
    }
    if (!is_on_cuve_simple(Px, Py, &mod_fp, curve.B) || !is_on_cuve_simple(Qx, Qy, &mod_fp, curve.B)) {
        return -1;
    }
    if (uint256_cmp(Px, Qx) == 0 && uint256_cmp(Py, Qy) == 0) {
        // Special case for doubling P == Q
        // printf("Doubling\n");
        // lambda = (3*Px^2) / (2*Py)
        uint256_mul_mod(&temp, Px, Px, &mod_fp);  // temp = Px^2
        numerator = 3;
        uint256_mul_mod(&numerator, &numerator, &temp, &mod_fp);  // numerator = 3*Px^2
        // printf("numerator\n");
        // numerator.print();

        denominator = 2;
        uint256_mul_mod(&denominator, &denominator, Py, &mod_fp);  // denominator = 2*Py

        uint256_modular_inverse(&lambda, &denominator, &mod_fp);
        // printf("lambda\n");
        // lambda.print();
        uint256_mul_mod(&lambda, &lambda, &numerator, &mod_fp);  // lambda = (3*Px^2) / (2*Py)
        // printf("lambda * numerator\n");
        // lambda.print();

    } else if (uint256_cmp(Px, Qx) == 0) {
        // printf("Doubling\n");
        // Special case for P != Q and Px == Qx
        // The result is the point at infinity
        *ResX = 0;
        *ResY = 0;
        return 0;
    } else {
        // printf("Adding\n");
        // General case for P != Q
        // lambda = (Qy - Py) / (Qx - Px)
        uint256_sub_mod(&temp, Qy, Py, &mod_fp);                // temp = Qy - Py
        uint256_sub_mod(&numerator, Qx, Px, &mod_fp);           // numerator = Qx - Px
        uint256_modular_inverse(&lambda, &numerator, &mod_fp);  // lambda = (Qy - Py) / (Qx - Px)
        uint256_mul_mod(&lambda, &lambda, &temp, &mod_fp);      // lambda = (Qy - Py) / (Qx - Px)
    }

    uint256_mul_mod(&x_r, &lambda, &lambda, &mod_fp);  // x_r = lambda^2
    // printf("x_r\n");
    // x_r.print();
    uint256_add_mod(&temp, Px, Qx, &mod_fp);  // temp = Px + Qx
    // printf("2x temp\n");
    // temp.print();
    uint256_sub_mod(&x_r, &x_r, &temp, &mod_fp);  // x_r = lambda^2 - (Px + Qx)
    // printf("x_r\n");
    // x_r.print();
    // y_r = lambda * (Px - x_r) - Py
    uint256_sub_mod(&temp, Px, &x_r, &mod_fp);       // temp = Px - x_r
    uint256_mul_mod(&y_r, &lambda, &temp, &mod_fp);  // y_r = lambda * (Px - x_r)
    uint256_sub_mod(&y_r, &y_r, Py, &mod_fp);        // y_r = lambda * (Px - x_r) - Py
    // Set the result
    *ResX = x_r;
    *ResY = y_r;
    return 0;
}
// Multiply a point on the curve G by a scalar n, store result in Res
__device__ int ec_mul(Curve curve, evm_word_t *ResX, evm_word_t *ResY, evm_word_t *Gx, evm_word_t *Gy, evm_word_t *n) {
    evm_word_t mod_fp;

    mod_fp = curve.FieldPrime;
    printf("EC_MUL\n");
    printf("Gx\n");
    Gx->print();
    printf("Gy\n");
    Gy->print();
    printf("n\n");
    n->print();
    if (!is_on_cuve_simple(Gx, Gy, &mod_fp, curve.B)) {
        printf("Point not on curve\n");
        return -1;
    }
    // check point at infinity
    if (uint256_is_zero(Gx) && uint256_is_zero(Gy) || uint256_is_zero(n)) {
        *ResX = 0;
        *ResY = 0;
        return 0;
    }

    uint8_t bitArray[CuEVM::word_bits];
    uint32_t bit_array_length = 0;

    n->to_bit_array_t(bitArray, bit_array_length);
    // printf("bitArray, length %d\n", bit_array_length);
    // for (int i = 0; i < bit_array_length; i++) {
    //     printf("%d", bitArray[i]);
    // }
    // printf("\n");
    // // there is a bug if calling The result RES == G, need to copy to temps
    evm_word_t temp_ResX, temp_ResY;

    // Double-and-add algorithm
    temp_ResX = *Gx;
    temp_ResY = *Gy;
    // printf("temp_ResX\n");
    // temp_ResX.print();
    // printf("temp_ResY\n");
    // temp_ResY.print();
    for (int i = bit_array_length - 2; i >= 0; --i) {
        // Gz = 2 * Gz

        ec_add(curve, &temp_ResX, &temp_ResY, &temp_ResX, &temp_ResY, &temp_ResX, &temp_ResY, false);
        // printf("point doubling\n");
        // temp_ResX.print();
        // temp_ResY.print();
        if (bitArray[i]) {
            // printf("Bit 1 Adding %d \n", i);
            ec_add(curve, &temp_ResX, &temp_ResY, &temp_ResX, &temp_ResY, Gx, Gy, false);
        }
        // printf("i %d\n", i);
        // temp_ResX.print();
        // temp_ResY.print();
    }
    *ResX = temp_ResX;
    *ResY = temp_ResY;
    return 0;
}

__host__ __device__ void convert_point_to_address(evm_word_t *address, evm_word_t *X, evm_word_t *Y) {
    uint8_t input[64];

    uint32_t array_length = 0;
    X->to_byte_array_t(input, array_length);

    Y->to_byte_array_t(input + 32, array_length);

    // print the entire byte array
    // print_byte_array_as_hex(input, 64);
    uint32_t in_length = 64, out_length = 32;
    uint8_t output[32];
    CuCrypto::keccak::sha3(input, in_length, output, out_length);
    // cgbn_set_memory(arith.env, address, temp_array, 32);
    // cgbn_bitwise_mask_and(arith.env, address, address, 160);
    uint256_from_bytes(address, output, 32);
    CuEVM::utils::evm_address_conversion(*address);
}

/**
 * Recover signer from the signature with the above structure. return the signer
 * address in cgbn type
 *
 * @param arith
 * @param sig
 * @param signer
 */
__device__ int ec_recover(CuEVM::EccConstants *ecc_constants_ptr, signature_t *sig, evm_word_t *signer) {
    // curve = ecc_constants_ptr->secp256k1;

    if (sig->v < 27 || sig->v > 28) {
        return -1;
    }

    evm_word_t r, r_y, r_inv, temp_1, temp_2;
    evm_word_t Gx, Gy, ResX, ResY, XY_x, XY_y;  // for the point multiplication

    // calculate R_invert
    r = sig->r;
    // cgbn_load(arith.env, mod_order, &curve.Order);
    evm_word_t *mod_order = &ecc_constants_ptr->secp256k1.Order;
    // cgbn_load(arith.env, mod_fp, &curve.FP);
    evm_word_t *mod_fp = &ecc_constants_ptr->secp256k1.FieldPrime;
    printf("r\n");
    r.print();

    uint256_mod(&r, &r, mod_order);
    printf("r\n");
    r.print();
    if (uint256_is_zero(&r) || uint256_cmp(&r, mod_order) >= 0) return -1;

    // calculate r_y

    uint256_mul_mod(&temp_1, &r, &r, mod_fp);

    uint256_mul_mod(&temp_1, &temp_1, &r, mod_fp);

    temp_2 = ecc_constants_ptr->secp256k1.B;
    uint256_addmod(&r_y, &temp_1, &temp_2, mod_fp);

    // find r_y using Tonelli–Shanks algorithm
    // beta = pow(xcubedaxb, (P+1)//4, P)

    evm_word_t four = 4;
    temp_2 = ecc_constants_ptr->secp256k1.FieldPrime;
    uint256_add_word(&temp_2, &temp_2, 1);
    uint256_div(&temp_2, &temp_2, &four);
    // printf(" (p+1)/4 \n");
    // temp_2.print();
    uint256_pow_mod(&r_y, &r_y, &temp_2, mod_fp);
    // printf(" r_y \n");
    // r_y.print();
    // y = beta if v % 2 ^ beta % 2 else (P - beta)
    uint32_t beta_mod2 = r_y.words[0] % 2;
    uint32_t v_mod2 = sig->v % 2;
    if (beta_mod2 == v_mod2) uint256_sub(&r_y, mod_fp, &r_y);

    // invalid point check
    // if (!is_on_cuve_simple(arith.env, r, r_y, mod_fp, curve.B)) return -1;
    if (!is_on_cuve_simple(&r, &r_y, mod_fp, ecc_constants_ptr->secp256k1.B)) return -1;

    // calculate n_z = (N-msg_hash) mod N
    uint256_sub_mod(&temp_1, mod_order, &sig->msg_hash, mod_order);

    // cgbn_load(arith.env, Gx, &curve.GX);
    Gx = ecc_constants_ptr->secp256k1.GX;

    // cgbn_load(arith.env, Gy, &curve.GY);
    Gy = ecc_constants_ptr->secp256k1.GY;

    ec_mul(ecc_constants_ptr->secp256k1, &ResX, &ResY, &Gx, &Gy, &temp_1);
    // printf("ResX\n");
    // ResX.print();
    // printf("ResY\n");
    // ResY.print();

    // calculate XY = (r, r_y) * s

    // check invalid s

    if (uint256_is_zero(&sig->s) || uint256_cmp(&sig->s, mod_order) >= 0) return -1;

    ec_mul(ecc_constants_ptr->secp256k1, &XY_x, &XY_y, &r, &r_y, &sig->s);
    // printf("XY_x\n");
    // XY_x.print();
    // printf("XY_y\n");
    // XY_y.print();
    // calculate QR = (ResX, ResY) + (XY_x, XY_y)
    ec_add(ecc_constants_ptr->secp256k1, &ResX, &ResY, &ResX, &ResY, &XY_x, &XY_y);
    // printf("QR_x\n");
    // ResX.print();
    // printf("QR_y\n");
    // ResY.print();

    uint256_modular_inverse(&r_inv, &r, mod_order);

    // env_t::cgbn_t Q_x, Q_y;
    // calculate Q = Qr * r_inv %N
    ec_mul(ecc_constants_ptr->secp256k1, &ResX, &ResY, &ResX, &ResY, &r_inv);
    if (uint256_is_zero(&ResX) && uint256_is_zero(&ResY))
        return -1;
    else {
        convert_point_to_address(signer, &ResX, &ResY);
        return 0;
    }
}
///

#ifdef ENABLE_PAIRING_CODE
template <size_t Degree>
__host__ __device__ constexpr const int32_t *get_modulus_coeffs();

template <>
__host__ __device__ constexpr const int32_t *get_modulus_coeffs<2>() {
    const int32_t FQ2_mod_coeffs[] = {1, 0};
    return FQ2_mod_coeffs;
}

template <>
__host__ __device__ constexpr const int32_t *get_modulus_coeffs<1>() {
    const int32_t FQ1_mod_coeffs[] = {1};
    return FQ1_mod_coeffs;
}

template <>
__host__ __device__ constexpr const int32_t *get_modulus_coeffs<12>() {
    const int32_t FQ12_mod_coeffs[] = {82, 0, 0, 0, 0, 0, -18, 0, 0, 0, 0, 0};
    return FQ12_mod_coeffs;
}

/***
 * @brief helper function to print FQP in hex
 */
template <size_t Degree>
__host__ __device__ void print_fqp(env_t env, FQ<Degree> &P, const char *name) {
    __SHARED_MEMORY__ evm_word_t scratch_pad;
    __ONE_GPU_THREAD_WOSYNC_BEGIN__
    printf("FQP %s: \n", name);
    __ONE_GPU_THREAD_END__
    for (size_t i = 0; i < Degree; i++) {
        cgbn_store(env, &scratch_pad, P.coeffs[i]);
        // pretty_hex_string_from_cgbn_memory(temp_str, scratch_pad);
        scratch_pad.print();
    }
}

/**
 * @brief Get the curve object
 *
 * @param arith
 * @param curve_id 256 for secp256k1, 128 for alt_BN128
 * @return Curve
 */
// __host__ __device__  Curve get_curve(ArithEnv &arith, int curve_id) {
//     Curve curve;
//     if (curve_id == 256) {
//         curve.FP.from_hex(secp256k1_FieldPrime);
//         curve.Order.from_hex(secp256k1_Order);
//         curve.GX.from_hex(secp256k1_GX);
//         curve.GY.from_hex(secp256k1_GY);
//         curve.B = 7;
//     } else if (curve_id == 128) {
//         curve.FP.from_hex(alt_BN128_FieldPrime);
//         curve.Order.from_hex(alt_BN128_Order);
//         curve.GX.from_hex(alt_BN128_GX);
//         curve.GY.from_hex(alt_BN128_GY);
//         curve.B = 3;
//     }
//     return curve;
// }

// /***
//  * @brief helper function to print FQP in hex
// */
// template <size_t Degree>
// void print_fqp(ArithEnv &arith, FQ<Degree> &P, const char *name) {
//     evm_word_t scratch_pad;
//     char *temp_str = new char[CuEVM::word_bits/8 * 2 + 3];
//     printf("%s: \n", name);
//     for (size_t i = 0; i < Degree; i++) {
//         cgbn_store(arith.env, &scratch_pad, P.coeffs[i]);
//         arith.pretty_hex_string_from_cgbn_memory(temp_str, scratch_pad);
//         printf("%s[%d] : %s\n", name, i, temp_str);
//     }

// }

template <size_t Degree>
__host__ __device__ bool FQP_equals(ArithEnv &arith, FQ<Degree> &P1, FQ<Degree> &P2) {
    for (size_t i = 0; i < Degree; i++) {
        if (!cgbn_equals(arith.env, P1.coeffs[i], P2.coeffs[i])) {
            return false;
        }
    }
    return true;
}

/**
 * @brief Get the FQ12 from bn_t
 *
 * @tparam Degree
 * @param env
 * @param res
 */
template <size_t Degree>
__host__ __device__ void getFQ12_from_cgbn_t(ArithEnv &arith, FQ<Degree> &res, bn_t (&coeffs)[Degree]) {
    for (uint32_t i = 0; i < Degree; i++) {
        cgbn_set(arith.env, res.coeffs[i], coeffs[i]);
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
__host__ __device__ void FQP_add(ArithEnv &arith, FQ<Degree> &Res, FQ<Degree> &P1, FQ<Degree> &P2, bn_t &mod) {
    for (size_t i = 0; i < Degree; i++) {
        cgbn_add_mod(arith.env, Res.coeffs[i], P1.coeffs[i], P2.coeffs[i], mod);
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
__host__ __device__ void FQP_sub(ArithEnv &arith, FQ<Degree> &Res, FQ<Degree> &P1, FQ<Degree> &P2, bn_t &mod) {
    for (size_t i = 0; i < Degree; i++) {
        // printf("P1[%d]: %s\n", i, bnt_to_string(env, P1.coeffs[i]));
        // printf("P2[%d]: %s\n", i, bnt_to_string(env, P2.coeffs[i]));
        // printf("mod: %s\n", bnt_to_string(env, mod));
        // printf("compare %d\n", cgbn_compare(env, P1.coeffs[i],
        // P2.coeffs[i]));
        cgbn_sub_mod(arith.env, Res.coeffs[i], P1.coeffs[i], P2.coeffs[i], mod);
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
__host__ __device__ uint deg(ArithEnv &arith, const FQ<Degree> &P) {
    uint res = Degree - 1;
    while (cgbn_equals_ui32(arith.env, P.coeffs[res], 0) && res) res -= 1;
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
__host__ __device__ FQ<Degree> get_one(ArithEnv &arith) {
    FQ<Degree> res;
    cgbn_set_ui32(arith.env, res.coeffs[0], 1);
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
__host__ __device__ void poly_rounded_div(ArithEnv &arith, FQ<Degree> &Res, FQ<Degree> &A, FQ<Degree> &B, bn_t &mod) {
    uint dega = deg(arith, A);
    uint degb = deg(arith, B);
    FQ<Degree> temp;
    for (size_t i = 0; i < Degree; i++) {
        cgbn_set(arith.env, temp.coeffs[i], A.coeffs[i]);
        cgbn_set_ui32(arith.env, Res.coeffs[i], 0);
    }

    FQ<Degree> o;
    bn_t temp_res;
    for (int32_t i = dega - degb; i >= 0; i--) {
        // o[i] += temp[degb + i] / b[degb]
        cgbn_div_mod(arith.env, temp_res, temp.coeffs[degb + i], B.coeffs[degb], mod);
        cgbn_add_mod(arith.env, o.coeffs[i], o.coeffs[i], temp_res, mod);
        for (size_t c = 0; c <= degb; c++) {
            cgbn_sub_mod(arith.env, temp.coeffs[c + i], temp.coeffs[c + i], o.coeffs[c], mod);
        }
    }
    // return o[:deg(o)+1]
    for (size_t i = 0; i < Degree; i++) {
        cgbn_set(arith.env, Res.coeffs[i], o.coeffs[i]);
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
__host__ __device__ void FQP_copy(ArithEnv &arith, FQ<Degree> &Res, FQ<Degree> &P) {
    for (size_t i = 0; i < Degree; i++) {
        cgbn_set(arith.env, Res.coeffs[i], P.coeffs[i]);
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
__host__ __device__ void FQP_mul(ArithEnv &arith, FQ<Degree> &Res, FQ<Degree> &P1, FQ<Degree> &P2, bn_t &mod) {
    if (Degree == 1) {
        cgbn_mul_mod(arith.env, Res.coeffs[0], P1.coeffs[0], P2.coeffs[0], mod);
        return;
    } else {
        bn_t b[2 * Degree - 1];

        bn_t temp;  // Temporary variable for intermediate results

        // set val
        // Polynomial multiplication
        for (size_t i = 0; i < Degree; i++) {
            for (size_t j = 0; j < Degree; j++) {
                cgbn_mul_mod(arith.env, temp, P1.coeffs[i], P2.coeffs[j],
                             mod);  // Multiply coefficients
                cgbn_add_mod(arith.env, b[i + j], b[i + j], temp,
                             mod);  // Add to the corresponding position and mod
            }
        }

        bn_t mod_coeffs[Degree];
        const int32_t *mod_coeffs_array = get_modulus_coeffs<Degree>();
        for (size_t i = 0; i < Degree; i++) {
            if (mod_coeffs_array[i] < 0) {
                // printf("negative mod coeffs\n");
                cgbn_sub_ui32(arith.env, mod_coeffs[i], mod, 0 - mod_coeffs_array[i]);
            } else {
                cgbn_set_ui32(arith.env, mod_coeffs[i], mod_coeffs_array[i]);
            }
        }
        for (size_t len_b = 2 * Degree - 1; len_b > Degree; len_b--) {
            size_t exp = len_b - Degree - 1;
            for (size_t i = 0; i < Degree; i++) {
                // Assuming FQ is a function that takes an int and returns a bn_t
                // type and modulus_coeffs is accessible b[exp + i] -= top *
                // FQ(modulus_coeffs[i]);
                cgbn_mul_mod(arith.env, temp, b[len_b - 1], mod_coeffs[i], mod);
                cgbn_sub_mod(arith.env, b[exp + i], b[exp + i], temp, mod);
            }
        }
        // Copy the result back to Res, adjusting to the actual degree
        for (size_t i = 0; i < Degree; i++) {
            cgbn_set(arith.env, Res.coeffs[i], b[i]);
        }
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
__host__ __device__ void FQP_inv(ArithEnv &arith, FQ<Degree> &Res, FQ<Degree> &P, bn_t &mod) {
    if (Degree == 1) {
        cgbn_modular_inverse(arith.env, Res.coeffs[0], P.coeffs[0], mod);
        return;
    }
    FQ<Degree + 1> lm, hm, low, high;
    bn_t temp;
    // lm[0] = 1;
    cgbn_set_ui32(arith.env, lm.coeffs[0], 1);
    // set low,high
    // Initialize high with modulus coefficients + [1]
    const int32_t *mod_coeffs = get_modulus_coeffs<Degree>();

    for (size_t i = 0; i < Degree; i++) {
        cgbn_set(arith.env, low.coeffs[i], P.coeffs[i]);
        if (mod_coeffs[i] < 0) {
            cgbn_sub_ui32(arith.env, high.coeffs[i], mod, 0 - mod_coeffs[i]);
        } else {
            cgbn_set_ui32(arith.env, high.coeffs[i], mod_coeffs[i]);
        }
    }
    cgbn_set_ui32(arith.env, high.coeffs[Degree], 1);

    while (deg(arith, low) > 0) {  // Assuming a function `deg` to find the polynomial's degree
        FQ<Degree + 1> r;
        poly_rounded_div(arith, r, high, low,
                         mod);  // Needs to return FQ<Degree+1> or similar

        FQ<Degree + 1> nm, new_h;
        FQP_copy(arith, nm, hm);
        FQP_copy(arith, new_h, high);
        for (size_t i = 0; i <= Degree; ++i) {
            for (size_t j = 0; j <= Degree - i; ++j) {
                // nm[i+j] -= lm[i] * r[j]
                cgbn_mul_mod(arith.env, temp, lm.coeffs[i], r.coeffs[j], mod);
                cgbn_sub_mod(arith.env, nm.coeffs[i + j], nm.coeffs[i + j], temp, mod);

                // new[i+j] -= low[i] * r[j]
                cgbn_mul_mod(arith.env, temp, low.coeffs[i], r.coeffs[j], mod);
                cgbn_sub_mod(arith.env, new_h.coeffs[i + j], new_h.coeffs[i + j], temp, mod);
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
        cgbn_set(arith.env, Res.coeffs[i], lm.coeffs[i]);
    }
    cgbn_modular_inverse(arith.env, temp, low.coeffs[0], mod);
    for (size_t i = 0; i < Degree; i++) {
        cgbn_mul_mod(arith.env, Res.coeffs[i], Res.coeffs[i], temp, mod);
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
__host__ __device__ void FQP_div(ArithEnv &arith, FQ<Degree> &Res, FQ<Degree> &P1, FQ<Degree> &P2,
                                 bn_t &mod) {  // P1/P2
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
__host__ __device__ void FQP_neg(ArithEnv &arith, FQ<Degree> &Res, FQ<Degree> &P, bn_t &mod) {
    for (size_t i = 0; i < Degree; i++) {
        cgbn_sub_mod(arith.env, Res.coeffs[i], mod, P.coeffs[i], mod);
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
// template <size_t Degree>
__host__ __device__ void FQP_pow(ArithEnv &arith, FQ<12> &Res, FQ<12> &P, bn_t &n, bn_t &mod) {
    // if (Degree == 1) {
    //     cgbn_modular_power(arith.env, Res.coeffs[0], P.coeffs[0], n, mod);
    //     return;
    // }
    bn_t temp_n;
    FQ<12> temp;
    FQP_copy(arith, temp, P);
    cgbn_set(arith.env, temp_n, n);
    // TODO, may not be necessary
    for (size_t i = 0; i < 12; i++) {
        cgbn_set_ui32(arith.env, Res.coeffs[i], 0);
    }
    cgbn_set_ui32(arith.env, Res.coeffs[0], 1);
    while (cgbn_get_ui32(arith.env, temp_n) != 0) {
        // printf("%d \n", cgbn_extract_bits_ui32(env, temp_n, 0, 1));
        if (cgbn_extract_bits_ui32(arith.env, temp_n, 0, 1) == 1) {
            FQP_mul(arith, Res, Res, temp, mod);
        }
        FQP_mul(arith, temp, temp, temp, mod);
        cgbn_shift_right(arith.env, temp_n, temp_n, 1);
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
__host__ __device__ void FQP_mul_scalar(ArithEnv &arith, FQ<Degree> &Res, FQ<Degree> &P, bn_t &n, bn_t &mod) {
    for (size_t i = 0; i < Degree; i++) {
        cgbn_mul_mod(arith.env, Res.coeffs[i], P.coeffs[i], n, mod);
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
__host__ __device__ bool FQP_is_on_curve(ArithEnv &arith, FQ<Degree> &Px, FQ<Degree> &Py, bn_t &mod, FQ<Degree> &B) {
    // y^2 = x^3 + B
    if (FQP_is_inf(arith, Px, Py)) return true;
    FQ<Degree> temp, temp2;
    FQP_mul(arith, temp, Px, Px, mod);
    FQP_mul(arith, temp, temp, Px, mod);
    FQP_add(arith, temp, temp, B, mod);
    FQP_mul(arith, temp2, Py, Py, mod);

    return FQP_equals(arith, temp, temp2);
}

// check if coordinates are valid (smaller than mod_fp)
template <size_t Degree>
__host__ __device__ bool FQP_is_valid(ArithEnv &arith, FQ<Degree> &P, bn_t &mod) {
    for (size_t i = 0; i < Degree; i++) {
        if (cgbn_compare(arith.env, P.coeffs[i], mod) >= 0) return false;
    }
    return true;
}

template <size_t Degree>
__host__ __device__ bool FQP_is_inf(ArithEnv &arith, FQ<Degree> &Px, FQ<Degree> &Py) {
    bool res = true;
    for (size_t i = 0; i < Degree; i++) {
        res = res && cgbn_equals_ui32(arith.env, Px.coeffs[i], 0) && cgbn_equals_ui32(arith.env, Py.coeffs[i], 0);
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
// template <size_t Degree>
__host__ __device__ void FQP_ec_add(ArithEnv &arith, FQ<12> &ResX, FQ<12> &ResY, FQ<12> &Px, FQ<12> &Py, FQ<12> &Qx,
                                    FQ<12> &Qy, bn_t &mod_fp) {
    FQ<12> lambda, numerator, denominator, temp, x_r, y_r;
    bn_t two, three;
    cgbn_set_ui32(arith.env, two, 2);
    cgbn_set_ui32(arith.env, three, 3);
    // check infinity
    if (FQP_is_inf(arith, Px, Py)) {
        FQP_copy(arith, ResX, Qx);
        FQP_copy(arith, ResY, Qy);
        return;
    } else if (FQP_is_inf(arith, Qx, Qy)) {
        FQP_copy(arith, ResX, Px);
        FQP_copy(arith, ResY, Py);
        return;
    }

    if (FQP_equals(arith, Px, Qx) && FQP_equals(arith, Py, Qy)) {
        FQP_mul(arith, temp, Px, Px, mod_fp);  // temp = Px^2
        FQP_mul_scalar(arith, numerator, temp, three,
                       mod_fp);  // numerator = 3*Px^2
        FQP_mul_scalar(arith, denominator, Py, two,
                       mod_fp);  // denominator = 2*Py
        FQP_div(arith, lambda, numerator, denominator,
                mod_fp);  // lambda = (3*Px^2) / (2*Py)

    } else if (FQP_equals(arith, Px, Qx)) {
        // special case, return inf
        for (size_t i = 0; i < 12; i++) {
            cgbn_set_ui32(arith.env, ResX.coeffs[i], 0);
            cgbn_set_ui32(arith.env, ResY.coeffs[i], 0);
        }
    } else {
        FQP_sub(arith, numerator, Qy, Py, mod_fp);    // temp = Qy - Py
        FQP_sub(arith, denominator, Qx, Px, mod_fp);  // numerator = Qx - Px
        FQP_div(arith, lambda, numerator, denominator,
                mod_fp);  // lambda = (Qy - Py) / (Qx - Px)
    }

    FQP_mul(arith, x_r, lambda, lambda, mod_fp);  // x_r = lambda^2
    FQP_add(arith, temp, Px, Qx, mod_fp);         // temp = Px + Qx
    FQP_sub(arith, x_r, x_r, temp, mod_fp);       // x_r = lambda^2 - (Px + Qx)

    FQP_sub(arith, temp, Px, x_r, mod_fp);      // temp = Px - x_r
    FQP_mul(arith, y_r, lambda, temp, mod_fp);  // y_r = lambda * (Px - x_r)
    FQP_sub(arith, y_r, y_r, Py, mod_fp);       // y_r = lambda * (Px - x_r) - Py

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
/*
template <size_t Degree>
__host__ __device__ void FQP_ec_mul(ArithEnv &arith, FQ<Degree> &ResX, FQ<Degree> &ResY, FQ<Degree> &Gx, FQ<Degree> &Gy,
                                    bn_t &n, bn_t &mod_fp) {
    uint8_t bitArray[CuEVM::word_bits];
    uint32_t bit_array_length = 0;

    __SHARED_MEMORY__ evm_word_t scratch_pad;
    cgbn_store(arith.env, &scratch_pad, n);
    get_bit_array(bitArray, bit_array_length, scratch_pad);

    // check point at infinity
    if (FQP_is_inf(arith, Gx, Gy)) {
        FQP_copy(arith, ResX, Gx);
        FQP_copy(arith, ResY, Gy);
        return;
    }
    // mul zero
    if (cgbn_equals_ui32(arith.env, n, 0)) {
        // return inf
        for (size_t i = 0; i < Degree; i++) {
            cgbn_set_ui32(arith.env, ResX.coeffs[i], 0);
            cgbn_set_ui32(arith.env, ResY.coeffs[i], 0);
        }
        return;
    }

    FQ<Degree> temp_ResX, temp_ResY;
    // Double-and-add algorithm
    FQP_copy(arith, temp_ResX, Gx);
    FQP_copy(arith, temp_ResY, Gy);

    for (int i = bit_array_length - 2; i >= 0; --i) {
        // Gz = 2 * Gz
        FQP_ec_add(arith, temp_ResX, temp_ResY, temp_ResX, temp_ResY, temp_ResX, temp_ResY, mod_fp);
        if (bitArray[CuEVM::word_bits - 1 - i]) {
            // printf("bit1 add i %d\n", i);
            // Gz = Gz + P
            FQP_ec_add(arith, temp_ResX, temp_ResY, temp_ResX, temp_ResY, Gx, Gy, mod_fp);
        }
    }
    FQP_copy(arith, ResX, temp_ResX);
    FQP_copy(arith, ResY, temp_ResY);
}
*/
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
// template <size_t Degree>
__host__ __device__ void FQP_linefunc(ArithEnv &arith, FQ<12> &Res, FQ<12> &P1x, FQ<12> &P1y, FQ<12> &P2x, FQ<12> &P2y,
                                      FQ<12> &Tx, FQ<12> &Ty, bn_t &mod) {
    FQ<12> m, temp, temp2;
    bn_t three, two;
    cgbn_set_ui32(arith.env, three, 3);
    cgbn_set_ui32(arith.env, two, 2);

    if (!FQP_equals(arith, P1x, P2x)) {
        FQP_sub(arith, temp, P2y, P1y, mod);
        FQP_sub(arith, temp2, P2x, P1x, mod);
        FQP_div(arith, m, temp, temp2, mod);

        FQP_sub(arith, temp, Tx, P1x, mod);
        FQP_mul(arith, Res, m, temp, mod);
        FQP_sub(arith, temp, Ty, P1y, mod);
        FQP_sub(arith, Res, Res, temp, mod);
        return;
    } else if (FQP_equals(arith, P1y, P2y)) {
        FQP_mul(arith, temp, P1x, P1x, mod);
        FQP_mul_scalar(arith, temp, temp, three, mod);
        FQP_mul_scalar(arith, temp2, P1y, two, mod);

        FQP_div(arith, m, temp, temp2, mod);

        FQP_sub(arith, temp, Tx, P1x, mod);
        FQP_mul(arith, Res, m, temp, mod);
        FQP_sub(arith, temp, Ty, P1y, mod);
        FQP_sub(arith, Res, Res, temp, mod);
        return;
    } else {
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
__host__ __device__ void FQP_twist(ArithEnv &arith, FQ<12> &Rx, FQ<12> &Ry, FQ<2> &Px, FQ<2> &Py, bn_t &mod_fp) {
    // # "Twist" a point in E(FQ2) into a point in E(FQ12)
    FQ<12> w;
    cgbn_set_ui32(arith.env, w.coeffs[1], 1);

    bn_t X[2];
    bn_t Y[2];
    bn_t temp;
    // xcoeffs = [_x.coeffs[0] - _x.coeffs[1] * 9, _x.coeffs[1]]
    cgbn_set_ui32(arith.env, temp, 9);
    cgbn_mul_mod(arith.env, X[0], Px.coeffs[1], temp, mod_fp);
    cgbn_sub_mod(arith.env, X[0], Px.coeffs[0], X[0], mod_fp);
    cgbn_set(arith.env, X[1], Px.coeffs[1]);

    // ycoeffs = [_y.coeffs[0] - _y.coeffs[1] * 9, _y.coeffs[1]]
    cgbn_mul_mod(arith.env, Y[0], Py.coeffs[1], temp, mod_fp);
    cgbn_sub_mod(arith.env, Y[0], Py.coeffs[0], Y[0], mod_fp);
    cgbn_set(arith.env, Y[1], Py.coeffs[1]);

    FQ<12> nx, ny, temp_fqp;
    //     nx = FQ12([xcoeffs[0]] + [0] * 5 + [xcoeffs[1]] + [0] * 5)
    //     ny = FQ12([ycoeffs[0]] + [0] * 5 + [ycoeffs[1]] + [0] * 5)
    cgbn_set(arith.env, nx.coeffs[0], X[0]);
    cgbn_set(arith.env, nx.coeffs[6], X[1]);
    cgbn_set(arith.env, ny.coeffs[0], Y[0]);
    cgbn_set(arith.env, ny.coeffs[6], Y[1]);

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
// template <size_t Degree>
__host__ __device__ void FQP_final_exponentiation(ArithEnv &arith, CuEVM::EccConstants *constants, FQ<12> &res,
                                                  FQ<12> &p, bn_t &mod) {
    // __SHARED_MEMORY__ evm_word_t temp_mem;
    size_t final_exp_len = 11;
    // const char *final_exp[11] = {"0000002f4b6dc97020fddadf107d20bc842d43bf6369b1ff6a1c71015f3f7be2",
    //                              "e1e30a73bb94fec0daf15466b2383a5d3ec3d15ad524d8f70c54efee1bd8c3b2",
    //                              "1377e563a09a1b705887e72eceaddea3790364a61f676baaf977870e88d5c6c8",
    //                              "fef0781361e443ae77f5b63a2a2264487f2940a8b1ddb3d15062cd0fb2015dfc",
    //                              "6668449aed3cc48a82d0d602d268c7daab6a41294c0cc4ebe5664568dfc50e16",
    //                              "48a45a4a1e3a5195846a3ed011a337a02088ec80e0ebae8755cfe107acf3aafb",
    //                              "40494e406f804216bb10cf430b0f37856b42db8dc5514724ee93dfb10826f0dd",
    //                              "4a0364b9580291d2cd65664814fde37ca80bb4ea44eacc5e641bbadf423f9a2c",
    //                              "bf813b8d145da90029baee7ddadda71c7f3811c4105262945bba1668c3be69a3",
    //                              "c230974d83561841d766f9c9d570bb7fbe04c7e8a6c3c760c0de81def35692da",
    //                              "361102b6b9b2b918837fa97896e84abb40a4efb7e54523a486964b64ca86f120"};

    FQ<12> temp, temp_res;
    cgbn_set_ui32(arith.env, temp_res.coeffs[0], 1);
    for (int i = 1; i < 12; i++) {
        cgbn_set_ui32(arith.env, temp_res.coeffs[i], 0);
    }

    FQP_copy(arith, temp, p);

    for (int i = final_exp_len - 1; i >= 0; i--) {
        bn_t temp_n;

        // temp_mem.from_hex(final_exp[i]);
        cgbn_load(arith.env, temp_n, &constants->final_exp[i]);

        for (int j = 0; j < 256; j++) {
            if (cgbn_extract_bits_ui32(arith.env, temp_n, 0, 1) == 1) {
                // cgbn_mul_mod(env, res, res, temp, mod);
                FQP_mul(arith, temp_res, temp_res, temp, mod);
            }
            // cgbn_mul_mod(env, temp, temp, temp, mod);
            FQP_mul(arith, temp, temp, temp, mod);
            cgbn_shift_right(arith.env, temp_n, temp_n, 1);
        }
    }
    FQP_copy(arith, res, temp_res);
}

__host__ __device__ void miller_loop_inner(int i, bn_t &ate_loop_count, ArithEnv &arith, FQ<12> &temp1, FQ<12> &temp2,
                                           FQ<12> &Px, FQ<12> &Py, FQ<12> &Qx, FQ<12> &Qy, FQ<12> &R_x, FQ<12> &R_y,
                                           FQ<12> &f, bn_t &mod_fp) {
    // f = f * f * linefunc(R, R, P)
    FQP_mul(arith, temp1, f, f, mod_fp);
    // temp 2 = linefunc(R, R, P)
    FQP_linefunc(arith, temp2, R_x, R_y, R_x, R_y, Px, Py, mod_fp);
    // print_fqp(env, temp2, "linefunc");
    FQP_mul(arith, f, temp1, temp2, mod_fp);
    // // R = double(R)
    FQP_ec_add(arith, R_x, R_y, R_x, R_y, R_x, R_y, mod_fp);

    // // if ate_loop_count & (2**i):
    if (cgbn_extract_bits_ui32(arith.env, ate_loop_count, i, 1)) {
        // f = f * linefunc(R, Q, P)
        FQP_linefunc(arith, temp2, R_x, R_y, Qx, Qy, Px, Py, mod_fp);
        FQP_mul(arith, f, f, temp2, mod_fp);
        // R = add(R, Q)
        FQP_ec_add(arith, R_x, R_y, R_x, R_y, Qx, Qy, mod_fp);
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
// template <size_t Degree>
__host__ __device__ void miller_loop(ArithEnv &arith, FQ<12> &Result, FQ<12> &Qx, FQ<12> &Qy, FQ<12> &Px, FQ<12> &Py,
                                     bn_t &mod_fp, bn_t &curve_order, bn_t &ate_loop_count, bool final_exp) {
    //  if Q is None or P is None: return FQ12.one()
    FQ<12> R_x, R_y, temp1, temp2;
    __SHARED_MEMORY__ evm_word_t scratch_pad;
    // bn_t ate_loop_count;
    // scratch_pad.from_hex(ate_loop_count_hex);
    // cgbn_load(arith.env, ate_loop_count, &scratch_pad);

    // f = FQ12.one()
    FQ<12> f = get_one<12>(arith);
    // R = Q
    FQP_copy(arith, R_x, Qx);
    FQP_copy(arith, R_y, Qy);

    for (int i = log_ate_loop_count; i >= 0; i--) {
        miller_loop_inner(i, ate_loop_count, arith, temp1, temp2, Qx, Qy, Px, Py, R_x, R_y, f, mod_fp);
    }

    // Compute Q1 and nQ2, adapt for your representation of points
    FQ<12> Q1_x, Q1_y, nQ2_x, nQ2_y;
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

    // if (final_exp) //not happens in the current implementation
    //     FQP_final_exponentiation(arith, constants, Result, f, mod_fp);
    // else
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
__host__ __device__ void pairing(ArithEnv &arith, FQ<12> &Res, FQ<2> &Qx, FQ<2> &Qy, FQ<1> &Px, FQ<1> &Py, bn_t &mod_fp,
                                 bn_t &curve_order, bn_t &ate_loop_count, bool final_exp) {
    // assert is_on_curve(Q, b2) assert is_on_curve(P, b)
    // return miller_loop(twist(Q), cast_point_to_fq12(P))
    FQ<12> Qx_tw, Qy_tw, Px_fp12, Py_fp12;
    FQP_twist(arith, Qx_tw, Qy_tw, Qx, Qy, mod_fp);
    cgbn_set(arith.env, Px_fp12.coeffs[0], Px.coeffs[0]);
    cgbn_set(arith.env, Py_fp12.coeffs[0], Py.coeffs[0]);
    miller_loop(arith, Res, Qx_tw, Qy_tw, Px_fp12, Py_fp12, mod_fp, curve_order, ate_loop_count, final_exp);
}

/*
 * @brief Pairing function for BN128 curve
 * @param points_data points data in the format of Px, Py, Qx, Qy (2 + 4
 * evm_words)
 * @param data_len length of the points_data in bytes
 * @return 1 if the pairing result is 1, -1 if invalid inputs, 0 if failed.
 *
 */
__host__ __device__ int pairing_multiple(ArithEnv &arith, EccConstants *ecc_constants_ptr, uint8_t *points_data,
                                         size_t data_len) {
    bn_t curve_order, mod_fp, ate_loop_count;
    __SHARED_MEMORY__ evm_word_t scratch_pad;
    // load curve order and mod_fp
    // scratch_pad.from_hex(alt_BN128_Order);
    // cgbn_load(arith.env, curve_order, &scratch_pad);
    cgbn_load(arith.env, curve_order, &ecc_constants_ptr->alt_BN128.Order);
    // scratch_pad.from_hex(alt_BN128_FieldPrime);
    // cgbn_load(arith.env, mod_fp, &scratch_pad);
    cgbn_load(arith.env, mod_fp, &ecc_constants_ptr->alt_BN128.FieldPrime);
    FQ<2> Qx, Qy, B2;
    FQ<1> Px, Py, B1;

    cgbn_load(arith.env, ate_loop_count, &ecc_constants_ptr->ate_loop_count);
    // setup B1, B2
    cgbn_set_ui32(arith.env, B1.coeffs[0], 3);
    // scratch_pad.from_hex(alt_BN128_G2_B_0);
    // cgbn_load(arith.env, B2.coeffs[0], &scratch_pad);
    cgbn_load(arith.env, B2.coeffs[0], &ecc_constants_ptr->alt_BN128_G2_B_0);
    // scratch_pad.from_hex(alt_BN128_G2_B_1);
    // cgbn_load(arith.env, B2.coeffs[1], &scratch_pad);
    cgbn_load(arith.env, B2.coeffs[1], &ecc_constants_ptr->alt_BN128_G2_B_1);

    FQ<12> final_res = get_one<12>(arith);

    size_t num_pairs = data_len / 192;  // 2 for G1, 4 for G2
    printf("Pairing Multiple, numpairs %d \n", num_pairs);
    for (int i = 0; i < num_pairs; i++) {
        FQ<12> temp_res;

        cgbn_set_memory(arith.env, Px.coeffs[0], points_data);
        cgbn_set_memory(arith.env, Py.coeffs[0], points_data + 32);
        // Important!!! X2 first then X1 for G2
        cgbn_set_memory(arith.env, Qx.coeffs[1], points_data + 64);
        cgbn_set_memory(arith.env, Qx.coeffs[0], points_data + 96);
        cgbn_set_memory(arith.env, Qy.coeffs[1], points_data + 128);
        cgbn_set_memory(arith.env, Qy.coeffs[0], points_data + 160);
        points_data += 192;
        // print point for debugging
        printf("before print_ fqp \n");
        print_fqp(arith.env, Px, "Px");
        print_fqp(arith.env, Py, "Py");
        print_fqp(arith.env, Qx, "Qx");
        print_fqp(arith.env, Qy, "Qy");
        bool on_curve = FQP_is_on_curve(arith, Px, Py, mod_fp, B1) && FQP_is_on_curve(arith, Qx, Qy, mod_fp, B2);

        bool valid = FQP_is_valid(arith, Px, mod_fp) && FQP_is_valid(arith, Py, mod_fp) &&
                     FQP_is_valid(arith, Qx, mod_fp) && FQP_is_valid(arith, Qy, mod_fp);

        if (!on_curve || !valid) {
            return -1;
        } else {
            if (FQP_is_inf(arith, Qx, Qy) || FQP_is_inf(arith, Px, Py)) {
                FQ<12> one_fq12 = get_one<12>(arith);
                FQP_copy(arith, temp_res, one_fq12);
            } else
                pairing(arith, temp_res, Qx, Qy, Px, Py, mod_fp, curve_order, ate_loop_count, false);
            // always false
        }

        FQP_mul(arith, final_res, final_res, temp_res, mod_fp);
        // print_fqp(arith, temp_res, "Temp Res");
        // print_fqp(arith, final_res, "Acc Res");
    }
    // final exp
    FQP_final_exponentiation(arith, ecc_constants_ptr, final_res, final_res, mod_fp);

    // print_fqp(arith, final_res, "Final");
    FQ<12> one_fq12 = get_one<12>(arith);
    return FQP_equals(arith, final_res, one_fq12) ? 1 : 0;
}
#else  // dummy delc to avoid compilation error
  __host__ __device__ int pairing_multiple(EccConstants *ecc_constants_ptr, uint8_t *points_data, size_t data_len) { return -1;}
#endif
}  // namespace ecc
