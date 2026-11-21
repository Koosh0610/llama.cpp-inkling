// NVFP4 matvec decode-strategy comparison. Block: 4 UE4M3 scale bytes + 32 code bytes (64 weights).
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <string>
#include <initializer_list>

static std::string src(int NR, int NSG, int DEC) {
    char buf[16384];
    snprintf(buf, sizeof buf, R"(
#include <metal_stdlib>
using namespace metal;
#define NR %d
#define NSG %d
#define DEC %d
struct block_nvfp4 { uchar d[4]; uchar qs[32]; };
constant float kv[16] = { 0, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f };
static inline float sc(uchar x) { x &= 0x7F; return x == 0x7F ? 0.0f : (float) as_type<half>(ushort(ushort(x) << 7)); }
// DEC 0: per-nibble half subnormal (ggml mxfp4_dq1); 1: packed half2 (2 nibbles per op); 2: constant LUT
static inline half dq1(uint n) { const half c = as_type<half>(ushort((n & 7) << 9)); return (n & 8) ? -c : c; }
static inline float4 lo4(uint w) {
#if DEC == 0
    return float4(dq1(w), dq1(w >> 8), dq1(w >> 16), dq1(w >> 24));
#elif DEC == 1
    // nibbles at bits 0 and 16 -> half2 lanes; then bits 8 and 24
    const uint a = w, b = w >> 8;
    const half2 ha = as_type<half2>(((a & 0x00070007u) << 9) | ((a & 0x00080008u) << 12));
    const half2 hb = as_type<half2>(((b & 0x00070007u) << 9) | ((b & 0x00080008u) << 12));
    return float4(ha.x, hb.x, ha.y, hb.y);
#else
    return float4(kv[w & 15], kv[(w >> 8) & 15], kv[(w >> 16) & 15], kv[(w >> 24) & 15]);
#endif
}
static inline float4 hi4(uint w) { return lo4(w >> 4); }
#if DEC == 2
#define SCL 256.0f
#else
#define SCL 4194304.0f
#endif
kernel void mv(device const block_nvfp4 * W [[buffer(0)]], device const float4 * x [[buffer(2)]], device float * y [[buffer(3)]],
               constant uint & M [[buffer(4)]], constant uint & K [[buffer(5)]],
               uint tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]], ushort tid [[thread_index_in_simdgroup]]) {
    const uint nb = K / 64;
    const uint r0 = (tg * NSG + sg) * NR;
    float acc[NR];
    for (int r = 0; r < NR; r++) acc[r] = 0.f;
    for (uint u = tid; u < 2 * nb; u += 32) {
        const uint ib = u >> 1, h = u & 1;
        float4 yl[8];
        for (int j = 0; j < 8; j++) yl[j] = x[ib * 16 + h * 8 + j];
        for (int r = 0; r < NR; r++) {
            device const block_nvfp4 & xb = W[(r0 + r) * nb + ib];
            device const uint * qw = (device const uint *)(xb.qs + 16 * h);
            const uint w0 = qw[0], w1 = qw[1], w2 = qw[2], w3 = qw[3];
            const float4 a = yl[0]*lo4(w0) + yl[1]*lo4(w1) + yl[2]*hi4(w0) + yl[3]*hi4(w1);
            const float4 b = yl[4]*lo4(w2) + yl[5]*lo4(w3) + yl[6]*hi4(w2) + yl[7]*hi4(w3);
            acc[r] += sc(xb.d[2*h]) * ((a.x + a.y) + (a.z + a.w)) + sc(xb.d[2*h+1]) * ((b.x + b.y) + (b.z + b.w));
        }
    }
    for (int r = 0; r < NR; r++) { const float t = simd_sum(acc[r]); if (tid == 0 && r0 + r < M) y[r0 + r] = t * SCL; }
}
)", NR, NSG, DEC);
    return buf;
}

static float ue4m3(uint8_t x) {
    x &= 0x7F; if (x == 0 || x == 0x7F) return 0.f;
    int e = (x >> 3) & 0xF, m = x & 7;
    return e == 0 ? ldexpf((float) m, -9) : ldexpf(1.f + m / 8.f, e - 7);
}
static const float KV[16] = { 0, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f };

int main() {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> q = [dev newCommandQueue];
        const uint M = 18432, K = 2304, nb = K / 64;
        const int NCOPY = 16;
        const size_t wbytes = (size_t) M * nb * 36;
        id<MTLBuffer> Wb = [dev newBufferWithLength:wbytes * NCOPY options:MTLResourceStorageModeShared];
        id<MTLBuffer> X = [dev newBufferWithLength:K * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> Y = [dev newBufferWithLength:M * 4 options:MTLResourceStorageModeShared];
        srand(1);
        uint8_t * wp = (uint8_t *) Wb.contents; float * xp = (float *) X.contents;
        for (size_t blk = 0; blk < (size_t) M * nb * NCOPY; blk++) {
            uint8_t * b = wp + blk * 36;
            for (int s = 0; s < 4; s++) b[s] = 0x20 + rand() % 0x30;   // scales in a sane range
            for (int i = 0; i < 32; i++) b[4 + i] = rand() & 0xFF;
        }
        for (uint i = 0; i < K; i++) xp[i] = (rand() % 2000 - 1000) / 1000.0f;
        std::vector<float> ref(M);
        for (uint r = 0; r < M; r++) {
            double acc = 0;
            for (uint ib = 0; ib < nb; ib++) {
                const uint8_t * b = wp + ((size_t) r * nb + ib) * 36;
                for (int s = 0; s < 4; s++) {
                    const double d = ue4m3(b[s]);
                    for (int j = 0; j < 8; j++) {
                        const uint8_t v = b[4 + s * 8 + j];
                        acc += d * KV[v & 15] * xp[ib * 64 + s * 16 + j];
                        acc += d * KV[v >> 4] * xp[ib * 64 + s * 16 + 8 + j];
                    }
                }
            }
            ref[r] = (float) acc;
        }
        printf("NVFP4 %ux%u, %.1f MB per run\n", M, K, wbytes / 1e6);
        for (int DEC : {0, 1, 2}) for (int NR : {1, 2, 4}) for (int NSG : {2, 4, 8}) {
            NSError * err = nil;
            id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:src(NR, NSG, DEC).c_str()] options:nil error:&err];
            if (!lib) { printf("compile: %s\n", err.localizedDescription.UTF8String); return 1; }
            id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mv"] error:&err];
            const uint ntg = M / (NR * NSG);
            double best = 1e9;
            for (int rep = 0; rep < 4; rep++) {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                [e setComputePipelineState:ps];
                for (int c = 0; c < NCOPY; c++) {
                    [e setBuffer:Wb offset:wbytes * c atIndex:0]; [e setBuffer:X offset:0 atIndex:2]; [e setBuffer:Y offset:0 atIndex:3];
                    [e setBytes:&M length:4 atIndex:4]; [e setBytes:&K length:4 atIndex:5];
                    [e dispatchThreadgroups:MTLSizeMake(ntg, 1, 1) threadsPerThreadgroup:MTLSizeMake(32 * NSG, 1, 1)];
                }
                [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
                const double us = (cb.GPUEndTime - cb.GPUStartTime) * 1e6 / NCOPY;
                if (rep > 0 && us < best) best = us;
            }
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:ps];
            [e setBuffer:Wb offset:0 atIndex:0]; [e setBuffer:X offset:0 atIndex:2]; [e setBuffer:Y offset:0 atIndex:3];
            [e setBytes:&M length:4 atIndex:4]; [e setBytes:&K length:4 atIndex:5];
            [e dispatchThreadgroups:MTLSizeMake(ntg, 1, 1) threadsPerThreadgroup:MTLSizeMake(32 * NSG, 1, 1)];
            [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
            float maxerr = 0; const float * yp = (const float *) Y.contents;
            for (uint r = 0; r < M; r++) maxerr = fmaxf(maxerr, fabsf(yp[r] - ref[r]) / (fabsf(ref[r]) + 1e-2f));
            printf("DEC %d NR %d NSG %d: %7.1f us %6.1f GB/s err %.1e\n", DEC, NR, NSG, best, wbytes / best / 1e3, maxerr);
        }
    }
    return 0;
}
