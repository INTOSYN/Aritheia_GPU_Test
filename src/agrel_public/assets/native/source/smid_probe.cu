// Aritheia Community — native SM-coverage probes (contract native-signed-unit-smid-v1)
// Copyright (C) 2026 The Aritheia authors — GNU GPL v3 or later (see LICENSE).
//
// Two tiny kernels, precompiled by the maintainer into a fatbinary that the client
// loads through the CUDA *driver* API (libcuda, shipped with every NVIDIA driver).
// The user installs nothing: no nvcc, no toolkit, no extra Python package.
//
//   smid_mma_bf16 : one warp per block; `chains` chains of `depth` chained
//                   mma.sync.m16n8k16 BF16->FP32 tensor-core steps. Operands are
//                   ±1 (bf16 0x3F80 / 0xBF80) from a shared integer hash, so every
//                   product is ±1 and, with depth <= 8, every partial sum lies in
//                   [-128, 128]: exactly representable, order-independent.
//   smid_ffma_fp32: 128 threads per block; each thread accumulates depth*16
//                   ±1 products with fmaf (FP32 pipe), same bound.
//
// Every block records %smid at entry and exit (thread 0). The host reconstructs
// the operands from the same hash and compares bit-exactly, grouping results by
// the logical SM that executed the block. SMID is a logical identifier; nothing
// here maps to a physical core, and a start==end SMID does not prove the block
// was never preempted.
#include <cstdint>

__device__ __forceinline__ unsigned mix32(unsigned x) {
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16; return x;
}
// One 32-bit word of sign bits for (seed, block, chain, step, word_index).
__device__ __forceinline__ unsigned sign_word(unsigned seed, unsigned block, unsigned chain, unsigned step, unsigned word) {
    return mix32(seed ^ mix32(block * 0x9E3779B1U + 1U) ^ mix32(chain * 0x85EBCA6BU + 2U)
                 ^ mix32(step * 0xC2B2AE35U + 3U) ^ mix32(word * 0x27D4EB2FU + 4U));
}
__device__ __forceinline__ unsigned bf16_pm1(unsigned bit) { return bit ? 0xBF80U : 0x3F80U; }  // 1 -> -1.0, 0 -> +1.0
__device__ __forceinline__ unsigned pack2(unsigned lo, unsigned hi) { return (lo & 0xFFFFU) | (hi << 16); }
__device__ __forceinline__ unsigned smid() { unsigned s; asm volatile("mov.u32 %0, %%smid;" : "=r"(s)); return s; }

// Element numbering shared with the host: A[r][c] -> elem r*16+c (0..255), B[k][n] -> elem 256+k*8+n (256..383).
__device__ __forceinline__ unsigned sign_of(unsigned seed, unsigned block, unsigned chain, unsigned step, unsigned elem) {
    return (sign_word(seed, block, chain, step, elem >> 5) >> (elem & 31U)) & 1U;
}

extern "C" __global__ void smid_mma_bf16(float* __restrict__ out, unsigned* __restrict__ sm_start,
                                         unsigned* __restrict__ sm_end, unsigned seed, unsigned chains, unsigned depth) {
    const unsigned block = blockIdx.x, lane = threadIdx.x & 31U;
    if (threadIdx.x == 0) sm_start[block] = smid();
    const unsigned g = lane >> 2, t = lane & 3U;
    for (unsigned chain = 0; chain < chains; ++chain) {
        float d[4] = {0.f, 0.f, 0.f, 0.f};
        for (unsigned step = 0; step < depth; ++step) {
            // mma.m16n8k16 .bf16 fragment layout (PTX ISA): A row-major 16x16, B "col" 16x8.
            unsigned a[4], b[2];
            #define SA(r, c) bf16_pm1(sign_of(seed, block, chain, step, (r) * 16U + (c)))
            #define SB(k, n) bf16_pm1(sign_of(seed, block, chain, step, 256U + (k) * 8U + (n)))
            a[0] = pack2(SA(g, 2U * t), SA(g, 2U * t + 1U));
            a[1] = pack2(SA(g + 8U, 2U * t), SA(g + 8U, 2U * t + 1U));
            a[2] = pack2(SA(g, 2U * t + 8U), SA(g, 2U * t + 9U));
            a[3] = pack2(SA(g + 8U, 2U * t + 8U), SA(g + 8U, 2U * t + 9U));
            b[0] = pack2(SB(2U * t, g), SB(2U * t + 1U, g));
            b[1] = pack2(SB(2U * t + 8U, g), SB(2U * t + 9U, g));
            #undef SA
            #undef SB
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
        }
        float* o = out + ((size_t)block * chains + chain) * 128U;   // D[r][c] at r*8+c
        o[g * 8U + 2U * t] = d[0];
        o[g * 8U + 2U * t + 1U] = d[1];
        o[(g + 8U) * 8U + 2U * t] = d[2];
        o[(g + 8U) * 8U + 2U * t + 1U] = d[3];
    }
    __syncthreads();
    if (threadIdx.x == 0) sm_end[block] = smid();
}

extern "C" __global__ void smid_ffma_fp32(float* __restrict__ out, unsigned* __restrict__ sm_start,
                                          unsigned* __restrict__ sm_end, unsigned seed, unsigned chains, unsigned depth) {
    const unsigned block = blockIdx.x, thread = threadIdx.x;
    if (thread == 0) sm_start[block] = smid();
    const unsigned terms = depth * 16U;
    for (unsigned chain = 0; chain < chains; ++chain) {
        float acc = 0.f;
        for (unsigned i = 0; i < terms; ++i) {
            // a-terms: words 0.., b-terms: words 4096.. ; the thread index selects the word group.
            unsigned wa = (thread * 2U) * 64U + (i >> 5), wb = (thread * 2U + 1U) * 64U + (i >> 5);
            float a = (sign_word(seed, block, chain, 0U, wa) >> (i & 31U)) & 1U ? -1.f : 1.f;
            float b = (sign_word(seed, block, chain, 0U, wb) >> (i & 31U)) & 1U ? -1.f : 1.f;
            acc = fmaf(a, b, acc);
        }
        out[((size_t)block * blockDim.x + thread) * chains + chain] = acc;
    }
    __syncthreads();
    if (thread == 0) sm_end[block] = smid();
}
