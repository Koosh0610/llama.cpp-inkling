// Does MoE scatter (8 random experts out of 128 in one big buffer) slow the NVFP4 matvec vs a contiguous matrix?
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <initializer_list>

static const char * SRC = R"(
#include <metal_stdlib>
using namespace metal;
#define NR 2
#define NSG 4
struct block_nvfp4 { uchar d[4]; uchar qs[32]; };
static inline float sc(uchar x) { x &= 0x7F; return x == 0x7F ? 0.0f : (float) as_type<half>(ushort(ushort(x) << 7)); }
static inline half dq1(uint n) { const half c = as_type<half>(ushort((n & 7) << 9)); return (n & 8) ? -c : c; }
static inline float4 lo4(uint w) { return float4(dq1(w), dq1(w >> 8), dq1(w >> 16), dq1(w >> 24)); }
static inline float4 hi4(uint w) { return lo4(w >> 4); }
kernel void mv(device const block_nvfp4 * W [[buffer(0)]], device const float4 * x [[buffer(2)]], device float * y [[buffer(3)]],
               constant uint & M [[buffer(4)]], constant uint & K [[buffer(5)]], device const int * ids [[buffer(6)]],
               uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]], ushort tid [[thread_index_in_simdgroup]]) {
    const uint nb = K / 64;
    device const block_nvfp4 * We = W + (ulong) ids[tg.z] * M * nb;
    const uint r0 = (tg.x * NSG + sg) * NR;
    float acc[NR] = {0.f, 0.f};
    for (uint u = tid; u < 2 * nb; u += 32) {
        const uint ib = u >> 1, h = u & 1;
        float4 yl[8];
        for (int j = 0; j < 8; j++) yl[j] = x[ib * 16 + h * 8 + j];
        for (int r = 0; r < NR; r++) {
            device const block_nvfp4 & xb = We[(r0 + r) * nb + ib];
            device const uint * qw = (device const uint *)(xb.qs + 16 * h);
            const uint w0 = qw[0], w1 = qw[1], w2 = qw[2], w3 = qw[3];
            const float4 a = yl[0]*lo4(w0) + yl[1]*lo4(w1) + yl[2]*hi4(w0) + yl[3]*hi4(w1);
            const float4 b = yl[4]*lo4(w2) + yl[5]*lo4(w3) + yl[6]*hi4(w2) + yl[7]*hi4(w3);
            acc[r] += sc(xb.d[2*h]) * ((a.x + a.y) + (a.z + a.w)) + sc(xb.d[2*h+1]) * ((b.x + b.y) + (b.z + b.w));
        }
    }
    for (int r = 0; r < NR; r++) { const float t = simd_sum(acc[r]); if (tid == 0) y[tg.z * M + r0 + r] = t * 4194304.0f; }
}
)";

int main() {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> q = [dev newCommandQueue];
        NSError * err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:SRC] options:nil error:&err];
        if (!lib) { printf("%s\n", err.localizedDescription.UTF8String); return 1; }
        id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mv"] error:&err];
        const uint M = 2304, K = 2304, nb = K / 64, NE = 128, NU = 8;
        const size_t ebytes = (size_t) M * nb * 36;
        id<MTLBuffer> W = [dev newBufferWithLength:ebytes * NE options:MTLResourceStorageModeShared];
        uint8_t * wp = (uint8_t *) W.contents;
        for (size_t blk = 0; blk < (size_t) M * nb * NE; blk++) {
            uint8_t * b = wp + blk * 36;
            for (int s = 0; s < 4; s++) b[s] = 0x20 + rand() % 0x30;
            for (int i = 0; i < 32; i++) b[4 + i] = rand() & 0xFF;
        }
        id<MTLBuffer> X = [dev newBufferWithLength:K * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> Y = [dev newBufferWithLength:M * NU * 4 options:MTLResourceStorageModeShared];
        const int NIT = 64;
        id<MTLBuffer> I = [dev newBufferWithLength:NIT * NU * 4 options:MTLResourceStorageModeShared];
        int * ip = (int *) I.contents;
        printf("8 experts x %ux%u NVFP4 per dispatch, %.1f MB, expert buffer %.2f GB\n", M, K, ebytes * NU / 1e6, ebytes * NE / 1e9);
        for (int mode : {0, 1, 2}) {
            // 0: same 8 contiguous experts every iteration; 1: 8 contiguous experts, different block each iteration; 2: 8 random distinct experts each iteration
            for (int it = 0; it < NIT; it++) {
                int base = mode == 1 ? (it * NU) % NE : 0;
                for (int k = 0; k < (int) NU; k++) {
                    if (mode == 2) {
                        int e; bool dup;
                        do { e = rand() % NE; dup = false; for (int j = 0; j < k; j++) dup |= ip[it * NU + j] == e; } while (dup);
                        ip[it * NU + k] = e;
                    } else {
                        ip[it * NU + k] = base + k;
                    }
                }
            }
            double best = 1e9;
            for (int rep = 0; rep < 4; rep++) {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                [e setComputePipelineState:ps];
                for (int it = 0; it < NIT; it++) {
                    [e setBuffer:W offset:0 atIndex:0]; [e setBuffer:X offset:0 atIndex:2]; [e setBuffer:Y offset:0 atIndex:3];
                    [e setBytes:&M length:4 atIndex:4]; [e setBytes:&K length:4 atIndex:5]; [e setBuffer:I offset:it * NU * 4 atIndex:6];
                    [e dispatchThreadgroups:MTLSizeMake(M / 8, 1, NU) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                }
                [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
                const double us = (cb.GPUEndTime - cb.GPUStartTime) * 1e6 / NIT;
                if (rep > 0 && us < best) best = us;
            }
            const char * names[] = { "same 8 experts each run (cache-warm)", "8 contiguous, new block each run", "8 random of 128 each run (real MoE)" };
            printf("%-40s %7.1f us  %6.1f GB/s\n", names[mode], best, ebytes * NU / best / 1e3);
        }
    }
    return 0;
}
