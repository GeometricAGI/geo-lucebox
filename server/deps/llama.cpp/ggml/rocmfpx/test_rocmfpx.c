#include "rocmfpx.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

static void fill_row(float * x, int n) {
    for (int i = 0; i < n; ++i) {
        const float wave = 0.75f*sinf((float) i * 0.37f) + 0.25f*cosf((float) i * 0.13f);
        const float ramp = ((float) (i % 11) - 5.0f) * 0.035f;
        x[i] = wave + ramp;
    }

    x[7]  =  3.25f;
    x[19] = -2.75f;
    x[43] =  1.875f;
}

static float mse(const float * a, const float * b, int n) {
    float err = 0.0f;

    for (int i = 0; i < n; ++i) {
        const float d = a[i] - b[i];
        err += d*d;
    }

    return err / (float) n;
}

static float weighted_mse(const float * a, const float * b, const float * w, int n) {
    float err = 0.0f;
    float sum_w = 0.0f;

    for (int i = 0; i < n; ++i) {
        const float d = a[i] - b[i];
        err += w[i]*d*d;
        sum_w += w[i];
    }

    return sum_w > 0.0f ? err / sum_w : 0.0f;
}

// ── Kernel-decode correctness ────────────────────────────────────────────
// The GPU MMVQ kernels (vec_dot_rocmfpx_fp3_q8_1 / _fp2_q8_1) don't dequantize
// to float — they decode each packed code to an int8 kvalue, integer-dp4a it
// against the q8 activations per half-block, then scale by the half-block's
// ue4m3 e[]. These host tests pin the three things that decode contract depends
// on — the code->value table, the packed-bit unpacking, and the half-block scale
// assignment — against the independent library dequantizer. The device perm-LUT
// decode is in turn tied to this same code->value mapping by a compile-time
// static_assert in vecdotq.cuh, so a divergence fails the GPU build too.

// fp3 code -> signed kvalue, mirroring rocmfpx_decode_fp3_code_vec_cuda:
// mag = (code&3)==3 ? 4 : code&3;  value = (code&4) ? -mag : mag.  => {0,1,2,4,0,-1,-2,-4}
static int fp3_code_value(unsigned code) {
    const unsigned mag_code = code & 3u;
    const int mag = (mag_code == 3u) ? 4 : (int) mag_code;
    return (code & 4u) ? -mag : mag;
}

static int fp2_code_value(unsigned code) {
    static const int kv[4] = {
        ROCMFP2_KVALUE_0_I8, ROCMFP2_KVALUE_1_I8, ROCMFP2_KVALUE_2_I8, ROCMFP2_KVALUE_3_I8
    };
    return kv[code & 3u];
}

// Unpack fp3 code i from the little-endian 3-bit stream (matches the kernel's
// dword-splice extraction: (stream >> 3i) & 7). 16-bit window handles a code
// straddling a byte boundary; the guard avoids reading past the 12-byte qs.
static unsigned fp3_get_code(const uint8_t * qs, int i) {
    const int bit = 3 * i;
    const int byte = bit >> 3;
    unsigned window = qs[byte];
    if (byte + 1 < QS_ROCMFP3) {
        window |= (unsigned) qs[byte + 1] << 8;
    }
    return (window >> (bit & 7)) & 7u;
}

// fp2 code i: byte i/4, 2 bits at 2*(i%4) (matches rocmfpx_pack4_fp2_vec_cuda).
static unsigned fp2_get_code(const uint8_t * qs, int i) {
    return (qs[i >> 2] >> (2 * (i & 3))) & 3u;
}

// Table pinned exhaustively — this is the contract the device perm-LUT encodes.
static void check_decode_tables(void) {
    const int fp3_expect[8] = { 0, 1, 2, 4, 0, -1, -2, -4 };
    for (unsigned c = 0; c < 8; ++c) {
        assert(fp3_code_value(c) == fp3_expect[c]);
    }
    const int fp2_expect[4] = { -1, 0, 1, 2 };
    for (unsigned c = 0; c < 4; ++c) {
        assert(fp2_code_value(c) == fp2_expect[c]);
    }
    printf("decode tables: fp3 {0,1,2,4,0,-1,-2,-4} fp2 {-1,0,1,2} OK\n");
}

// Reconstruct each weight from code->value * half-block scale using the kernel's
// unpack, and assert it equals the library dequantizer bit-for-bit. This pins
// the bit order, the code table, AND the e[i>=QK/2] half-block scale split.
static void check_decode_contract_fp3(void) {
    enum { N = 2 * QK_ROCMFP3 };
    float src[N], ref[N];
    block_rocmfp3 q[N / QK_ROCMFP3];
    fill_row(src, N);
    rocmfpx_quantize_fp3(src, q, 1, N, NULL);
    rocmfpx_dequantize_row_fp3(q, ref, N);
    for (int b = 0; b < N / QK_ROCMFP3; ++b) {
        const block_rocmfp3 * blk = &q[b];
        const float e0 = rocmfpx_ue4m3_to_fp32(blk->e[0]);
        const float e1 = rocmfpx_ue4m3_to_fp32(blk->e[1]);
        for (int i = 0; i < QK_ROCMFP3; ++i) {
            const float e = (i >= QK_ROCMFP3 / 2) ? e1 : e0;
            const float recon = (float) fp3_code_value(fp3_get_code(blk->qs, i)) * e;
            assert(recon == ref[b * QK_ROCMFP3 + i]);
        }
    }
    printf("fp3 decode contract: unpack + table + half-block scale == dequantize_row (exact) OK\n");
}

static void check_decode_contract_fp2(void) {
    enum { N = 2 * QK_ROCMFP2 };
    float src[N], ref[N];
    block_rocmfp2 q[N / QK_ROCMFP2];
    fill_row(src, N);
    rocmfpx_quantize_fp2(src, q, 1, N, NULL);
    rocmfpx_dequantize_row_fp2(q, ref, N);
    for (int b = 0; b < N / QK_ROCMFP2; ++b) {
        const block_rocmfp2 * blk = &q[b];
        const float e0 = rocmfpx_ue4m3_to_fp32(blk->e[0]);
        const float e1 = rocmfpx_ue4m3_to_fp32(blk->e[1]);
        for (int i = 0; i < QK_ROCMFP2; ++i) {
            const float e = (i >= QK_ROCMFP2 / 2) ? e1 : e0;
            const float recon = (float) fp2_code_value(fp2_get_code(blk->qs, i)) * e;
            assert(recon == ref[b * QK_ROCMFP2 + i]);
        }
    }
    printf("fp2 decode contract: unpack + table + half-block scale == dequantize_row (exact) OK\n");
}

// The kernel's dot is db * sum_half( e_half * sum_i code_value[i]*q8[i] ) with an
// exact integer inner sum per half-block (the VDR=4 half-block structure for fp3
// / the two-chain fp2 form). Assert it matches the float reference dot to tol.
static void check_dot_structure_fp3(void) {
    enum { N = QK_ROCMFP3 };
    float src[N], w[N];
    block_rocmfp3 q[1];
    fill_row(src, N);
    rocmfpx_quantize_fp3(src, q, 1, N, NULL);
    rocmfpx_dequantize_row_fp3(q, w, N);

    int8_t a8[N];
    for (int i = 0; i < N; ++i) { a8[i] = (int8_t) ((i * 37) % 251 - 125); }
    const float db = 0.031f;

    double ref = 0.0;
    for (int i = 0; i < N; ++i) { ref += (double) w[i] * (double) (db * (float) a8[i]); }

    const block_rocmfp3 * blk = &q[0];
    long s0 = 0, s1 = 0;
    for (int i = 0; i < N; ++i) {
        const int c = fp3_code_value(fp3_get_code(blk->qs, i));
        if (i < N / 2) { s0 += (long) c * a8[i]; } else { s1 += (long) c * a8[i]; }
    }
    const double e0 = rocmfpx_ue4m3_to_fp32(blk->e[0]);
    const double e1 = rocmfpx_ue4m3_to_fp32(blk->e[1]);
    const double kern = (double) db * (e0 * (double) s0 + e1 * (double) s1);
    const double rel = fabs(kern - ref) / (fabs(ref) + 1e-9);
    printf("fp3 dot structure: ref=%g kernel=%g rel=%g\n", ref, kern, rel);
    assert(rel < 1e-5);
}

static void check_dot_structure_fp2(void) {
    enum { N = QK_ROCMFP2 };
    float src[N], w[N];
    block_rocmfp2 q[1];
    fill_row(src, N);
    rocmfpx_quantize_fp2(src, q, 1, N, NULL);
    rocmfpx_dequantize_row_fp2(q, w, N);

    int8_t a8[N];
    for (int i = 0; i < N; ++i) { a8[i] = (int8_t) ((i * 37) % 251 - 125); }
    const float db = 0.031f;

    double ref = 0.0;
    for (int i = 0; i < N; ++i) { ref += (double) w[i] * (double) (db * (float) a8[i]); }

    const block_rocmfp2 * blk = &q[0];
    long s0 = 0, s1 = 0;
    for (int i = 0; i < N; ++i) {
        const int c = fp2_code_value(fp2_get_code(blk->qs, i));
        if (i < N / 2) { s0 += (long) c * a8[i]; } else { s1 += (long) c * a8[i]; }
    }
    const double e0 = rocmfpx_ue4m3_to_fp32(blk->e[0]);
    const double e1 = rocmfpx_ue4m3_to_fp32(blk->e[1]);
    const double kern = (double) db * (e0 * (double) s0 + e1 * (double) s1);
    const double rel = fabs(kern - ref) / (fabs(ref) + 1e-9);
    printf("fp2 dot structure: ref=%g kernel=%g rel=%g\n", ref, kern, rel);
    assert(rel < 1e-5);
}

static void check_weighted_imatrix_fp3(void) {
    enum { N = QK_ROCMFP3 };

    float src[N];
    float imatrix[N];
    float plain[N];
    float weighted[N];
    block_rocmfp3 q_plain[N / QK_ROCMFP3];
    block_rocmfp3 q_weighted[N / QK_ROCMFP3];

    for (int i = 0; i < N; ++i) {
        src[i] = (i % 2) ? 0.21f : -0.21f;
        imatrix[i] = 100.0f;
    }

    src[0] = 9.0f;
    imatrix[0] = 0.0f;

    rocmfpx_quantize_fp3(src, q_plain,    1, N, NULL);
    rocmfpx_quantize_fp3(src, q_weighted, 1, N, imatrix);
    rocmfpx_dequantize_row_fp3(q_plain,    plain,    N);
    rocmfpx_dequantize_row_fp3(q_weighted, weighted, N);

    const float plain_err = weighted_mse(src, plain, imatrix, N);
    const float weighted_err = weighted_mse(src, weighted, imatrix, N);

    printf("ROCmFP3 imatrix weighted_mse: plain=%g weighted=%g\n", plain_err, weighted_err);
    assert(weighted_err < plain_err);
}

static void check_weighted_imatrix_fp2(void) {
    enum { N = QK_ROCMFP2 };

    float src[N];
    float imatrix[N];
    float plain[N];
    float weighted[N];
    block_rocmfp2 q_plain[N / QK_ROCMFP2];
    block_rocmfp2 q_weighted[N / QK_ROCMFP2];

    for (int i = 0; i < N; ++i) {
        src[i] = (i % 2) ? 0.21f : -0.21f;
        imatrix[i] = 100.0f;
    }

    src[0] = 9.0f;
    imatrix[0] = 0.0f;

    rocmfpx_quantize_fp2(src, q_plain,    1, N, NULL);
    rocmfpx_quantize_fp2(src, q_weighted, 1, N, imatrix);
    rocmfpx_dequantize_row_fp2(q_plain,    plain,    N);
    rocmfpx_dequantize_row_fp2(q_weighted, weighted, N);

    const float plain_err = weighted_mse(src, plain, imatrix, N);
    const float weighted_err = weighted_mse(src, weighted, imatrix, N);

    printf("ROCmFP2 imatrix weighted_mse: plain=%g weighted=%g\n", plain_err, weighted_err);
    assert(weighted_err < plain_err);
}

int main(void) {
    enum { N = 64 };

    float src[N];
    float fp2[N];
    float fp3[N];
    float fp6[N];
    float fp8[N];

    block_rocmfp2 q2[N / QK_ROCMFP2];
    block_rocmfp3 q3[N / QK_ROCMFP3];
    block_rocmfp6 q6[N / QK_ROCMFP6];
    block_rocmfp8 q8[N / QK_ROCMFP8];

    fill_row(src, N);

    rocmfpx_quantize_row_fp2_ref(src, q2, N);
    rocmfpx_quantize_row_fp3_ref(src, q3, N);
    rocmfpx_quantize_row_fp6_ref(src, q6, N);
    rocmfpx_quantize_row_fp8_ref(src, q8, N);

    assert(rocmfpx_validate_row_data_fp2(q2, sizeof(q2)));
    assert(rocmfpx_validate_row_data_fp3(q3, sizeof(q3)));
    assert(rocmfpx_validate_row_data_fp6(q6, sizeof(q6)));
    assert(rocmfpx_validate_row_data_fp8(q8, sizeof(q8)));

    rocmfpx_dequantize_row_fp2(q2, fp2, N);
    rocmfpx_dequantize_row_fp3(q3, fp3, N);
    rocmfpx_dequantize_row_fp6(q6, fp6, N);
    rocmfpx_dequantize_row_fp8(q8, fp8, N);

    const float mse2 = mse(src, fp2, N);
    const float mse3 = mse(src, fp3, N);
    const float mse6 = mse(src, fp6, N);
    const float mse8 = mse(src, fp8, N);

    printf("ROCmFP2: block=%zu row=%zu bpw=%.2f mse=%g\n",
            sizeof(block_rocmfp2), rocmfpx_row_size_fp2(N),
            8.0f*(float) sizeof(block_rocmfp2)/(float) QK_ROCMFP2, mse2);
    printf("ROCmFP3: block=%zu row=%zu bpw=%.2f mse=%g\n",
            sizeof(block_rocmfp3), rocmfpx_row_size_fp3(N),
            8.0f*(float) sizeof(block_rocmfp3)/(float) QK_ROCMFP3, mse3);
    printf("ROCmFP6: block=%zu row=%zu bpw=%.2f mse=%g\n",
            sizeof(block_rocmfp6), rocmfpx_row_size_fp6(N),
            8.0f*(float) sizeof(block_rocmfp6)/(float) QK_ROCMFP6, mse6);
    printf("ROCmFP8: block=%zu row=%zu bpw=%.2f mse=%g\n",
            sizeof(block_rocmfp8), rocmfpx_row_size_fp8(N),
            8.0f*(float) sizeof(block_rocmfp8)/(float) QK_ROCMFP8, mse8);

    assert(sizeof(block_rocmfp2) == 10);
    assert(isfinite(mse2));
    assert(isfinite(mse3));
    assert(isfinite(mse6));
    assert(isfinite(mse8));
    assert(mse8 < mse6);
    assert(mse6 < mse3);

    check_weighted_imatrix_fp2();
    check_weighted_imatrix_fp3();

    // Kernel-decode correctness (MMVQ fp3/fp2 vec-dot contract).
    check_decode_tables();
    check_decode_contract_fp3();
    check_decode_contract_fp2();
    check_dot_structure_fp3();
    check_dot_structure_fp2();

    return 0;
}
