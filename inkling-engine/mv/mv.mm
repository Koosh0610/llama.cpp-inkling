// Prototype: Q4_0-style matvec with a repacked SoA layout (16-byte nibble blocks + separate half scales).
// y[r] = sum_k W[r,k] * x[k]; W is M x K, 32-weight blocks: w = d * (q - 8).
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <initializer_list>
#include <string>

static std::string src(int NR, int NSG) {
    char buf[8192];
    snprintf(buf, sizeof buf, R"(
#include <metal_stdlib>
using namespace metal;
#define NR %d
#define NSG %d
// qs: [M][K/32] uint4 (32 nibbles), d: [M][K/32] half, x: [K] float
kernel void mv(device const uint4 * qs [[buffer(0)]], device const half * d [[buffer(1)]],
               device const float4 * x [[buffer(2)]], device float * y [[buffer(3)]],
               constant uint & M [[buffer(4)]], constant uint & K [[buffer(5)]],
               uint tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
               ushort tid [[thread_index_in_simdgroup]]) {
    const uint nb = K / 32;
    const uint r0 = (tg * NSG + sg) * NR;
    float acc[NR];
    for (int r = 0; r < NR; r++) acc[r] = 0.f;
    for (uint b = tid; b < nb; b += 32) {
        // x block: 32 floats = 8 float4
        float4 xv[8];
        for (int j = 0; j < 8; j++) xv[j] = x[b * 8 + j];
        for (int r = 0; r < NR; r++) {
            const uint row = r0 + r;
            const uint4 q = qs[row * nb + b];
            const float s = (float) d[row * nb + b];
            // nibble j of word w: low nibbles are weights 0..15, high are 16..31 (like q4_0)
            float sum = 0.f;
            const uint w[4] = { q.x, q.y, q.z, q.w };
            for (int i = 0; i < 4; i++) {
                const uint v = w[i];
                const float4 lo = float4((v      ) & 0xF, (v >>  8) & 0xF, (v >> 16) & 0xF, (v >> 24) & 0xF) - 8.f;
                const float4 hi = float4((v >>  4) & 0xF, (v >> 12) & 0xF, (v >> 20) & 0xF, (v >> 28) & 0xF) - 8.f;
                sum += dot(lo, xv[i]) + dot(hi, xv[4 + i]);
            }
            acc[r] += s * sum;
        }
    }
    for (int r = 0; r < NR; r++) {
        const float t = simd_sum(acc[r]);
        if (tid == 0 && r0 + r < M) y[r0 + r] = t;
    }
}

struct block_q4_0 { half d; uchar qs[16]; };
kernel void mv_aos(device const block_q4_0 * W [[buffer(0)]], device const half * unused [[buffer(1)]],
               device const float4 * x [[buffer(2)]], device float * y [[buffer(3)]],
               constant uint & M [[buffer(4)]], constant uint & K [[buffer(5)]],
               uint tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
               ushort tid [[thread_index_in_simdgroup]]) {
    const uint nb = K / 32;
    const uint r0 = (tg * NSG + sg) * NR;
    float acc[NR];
    for (int r = 0; r < NR; r++) acc[r] = 0.f;
    for (uint b = tid; b < nb; b += 32) {
        float4 xv[8];
        for (int j = 0; j < 8; j++) xv[j] = x[b * 8 + j];
        for (int r = 0; r < NR; r++) {
            device const block_q4_0 & blk = W[(r0 + r) * nb + b];
            device const ushort * q16 = (device const ushort *) blk.qs;
            const float s = (float) blk.d;
            float sum = 0.f;
            for (int i = 0; i < 4; i++) {
                const uint v = uint(q16[2*i]) | (uint(q16[2*i+1]) << 16);
                const float4 lo = float4((v      ) & 0xF, (v >>  8) & 0xF, (v >> 16) & 0xF, (v >> 24) & 0xF) - 8.f;
                const float4 hi = float4((v >>  4) & 0xF, (v >> 12) & 0xF, (v >> 20) & 0xF, (v >> 28) & 0xF) - 8.f;
                sum += dot(lo, xv[i]) + dot(hi, xv[4 + i]);
            }
            acc[r] += s * sum;
        }
    }
    for (int r = 0; r < NR; r++) {
        const float t = simd_sum(acc[r]);
        if (tid == 0 && r0 + r < M) y[r0 + r] = t;
    }
}
)", NR, NSG);
    return buf;
}

static const char * KNAME = getenv("AOS") ? "mv_aos" : "mv";
int main() {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> q = [dev newCommandQueue];
        const uint M = 18432, K = 2304, nb = K / 32;
        // 64 copies of the matrix so every run streams from DRAM, not the SLC
        const int NCOPY = 16;
        const bool aosm = getenv("AOS") != nullptr;
        const size_t qbytes = (size_t) M * nb * (aosm ? 18 : 16), dbytes = aosm ? 16 : (size_t) M * nb * 2;
        id<MTLBuffer> Q = [dev newBufferWithLength:qbytes * NCOPY options:MTLResourceStorageModeShared];
        id<MTLBuffer> D = [dev newBufferWithLength:dbytes * NCOPY options:MTLResourceStorageModeShared];
        id<MTLBuffer> X = [dev newBufferWithLength:K * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> Y = [dev newBufferWithLength:M * 4 options:MTLResourceStorageModeShared];
        srand(1);
        uint8_t * qp = (uint8_t *) Q.contents; __fp16 * dp = (__fp16 *) D.contents; float * xp = (float *) X.contents;
        for (size_t i = 0; i < qbytes * NCOPY; i++) qp[i] = rand() & 0xFF;
        if (aosm) { for (size_t i = 0; i < (size_t) M * nb * NCOPY; i++) *(__fp16 *)(qp + i * 18) = (__fp16)((rand() % 1000) / 1000.0f * 0.02f); }
        else { for (size_t i = 0; i < (size_t) M * nb * NCOPY; i++) dp[i] = (__fp16)((rand() % 1000) / 1000.0f * 0.02f); }
        for (uint i = 0; i < K; i++) xp[i] = (rand() % 2000 - 1000) / 1000.0f;
        // CPU reference for copy 0
        std::vector<float> ref(M);
        for (uint r = 0; r < M; r++) {
            double acc = 0;
            for (uint b = 0; b < nb; b++) {
                const uint8_t * blk = aosm ? qp + ((size_t) r * nb + b) * 18 + 2 : qp + ((size_t) r * nb + b) * 16;
                const double s = aosm ? (double) *(const __fp16 *)(qp + ((size_t) r * nb + b) * 18) : (double) dp[(size_t) r * nb + b];
                for (int j = 0; j < 16; j++) {
                    acc += s * (((blk[j] >> 0) & 0xF) - 8) * xp[b * 32 + (j % 4) * 1 + (j / 4) * 4 - (j / 4) * 4 + 0 * 0 + ((j / 4) * 4 + (j % 4)) - j + 0] * 0; // placeholder, replaced below
                }
                // match kernel nibble order: word i = bytes 4i..4i+3; byte k of word i -> lo nibble weight 4i+k, hi nibble weight 16+4i+k
                for (int i = 0; i < 4; i++) for (int k = 0; k < 4; k++) {
                    const uint8_t byte = blk[4 * i + k];
                    acc += s * ((byte & 0xF) - 8) * xp[b * 32 + 4 * i + k];
                    acc += s * ((byte >> 4) - 8) * xp[b * 32 + 16 + 4 * i + k];
                }
            }
            ref[r] = (float) acc;
        }
        const double bytes = qbytes + dbytes;
        printf("matrix %ux%u, %.1f MB per run\n", M, K, bytes / 1e6);
        for (int NR : {1, 2, 4, 8, 16}) for (int NSG : {1, 2, 4, 8}) {
            NSError * err = nil;
            id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:src(NR, NSG).c_str()] options:nil error:&err];
            if (!lib) { printf("compile: %s\n", err.localizedDescription.UTF8String); return 1; }
            id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:[NSString stringWithUTF8String:KNAME]] error:&err];
            const uint ntg = M / (NR * NSG);
            double best = 1e9; float maxerr = 0;
            for (int rep = 0; rep < 3; rep++) {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                [e setComputePipelineState:ps];
                for (int c = 0; c < NCOPY; c++) {
                    [e setBuffer:Q offset:qbytes * c atIndex:0]; [e setBuffer:D offset:(aosm ? 0 : dbytes * c) atIndex:1];
                    [e setBuffer:X offset:0 atIndex:2]; [e setBuffer:Y offset:0 atIndex:3];
                    [e setBytes:&M length:4 atIndex:4]; [e setBytes:&K length:4 atIndex:5];
                    [e dispatchThreadgroups:MTLSizeMake(ntg, 1, 1) threadsPerThreadgroup:MTLSizeMake(32 * NSG, 1, 1)];
                }
                [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
                const double us = (cb.GPUEndTime - cb.GPUStartTime) * 1e6 / NCOPY;
                if (us < best) best = us;
            }
            // correctness: run copy 0 alone
            {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                [e setComputePipelineState:ps];
                [e setBuffer:Q offset:0 atIndex:0]; [e setBuffer:D offset:0 atIndex:1];
                [e setBuffer:X offset:0 atIndex:2]; [e setBuffer:Y offset:0 atIndex:3];
                [e setBytes:&M length:4 atIndex:4]; [e setBytes:&K length:4 atIndex:5];
                [e dispatchThreadgroups:MTLSizeMake(ntg, 1, 1) threadsPerThreadgroup:MTLSizeMake(32 * NSG, 1, 1)];
                [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
                const float * yp = (const float *) Y.contents;
                for (uint r = 0; r < M; r++) maxerr = fmaxf(maxerr, fabsf(yp[r] - ref[r]) / (fabsf(ref[r]) + 1e-3f));
            }
            printf("NR %2d NSG %d: %7.1f us  %6.1f GB/s  max rel err %.2e\n", NR, NSG, best, bytes / best / 1e3, maxerr);
        }
    }
    return 0;
}
