// Stage 1 probes for a persistent decode kernel on Apple GPUs.
//   a:  cost of a dispatch boundary with a memory barrier in a concurrent encoder
//   b1: how many threadgroups stay co-resident (bounded spin, cannot hang)
//   b2: P dependent phases in one dispatch, synced grid-wide through a device-memory atomic counter
//   c:  persistent Q8_0-like matvec bandwidth vs one dispatch per matvec
// build: clang++ -O2 -std=c++17 -fobjc-arc sync.mm -framework Metal -framework Foundation -o sync
// usage: ../gpulock.sh ./sync [a|b1|b2|c|soak N]
// env:   DATA=0..6 data layout for b2/soak, MODE=0..4 sync protocol for soak, QUICK=1 one config,
//        ONE_TG=1 single threadgroup control, SYNC_DEBUG=1 print why a run failed
//
// Result on M5 (Nov 21): the counter sync itself never timed out up to 30720 co-resident threads, and a
// persistent matvec keeps full bandwidth, but plain stores from one threadgroup are not reliably visible to
// other threadgroups inside the same dispatch. Re-read or line-sharing layouts (DATA 0, 3, 4, 6) are wrong on
// every run, fresh regions (DATA 2) and atomics as the data channel (DATA 5) fail sporadically and much more
// often when another process uses the GPU. Failures are silent: no timeout, just stale values. The same
// layouts are always correct inside one threadgroup (ONE_TG=1). So a grid-synced megakernel is not safe.
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <algorithm>
#include <initializer_list>

static const char * SRC = R"(
#include <metal_stdlib>
using namespace metal;

kernel void step(device const float * x [[buffer(0)]], device float * y [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    y[gid] = x[gid] + 1.0f;
}

kernel void step2(device const float * x [[buffer(0)]], device float * y [[buffer(1)]], constant float & one [[buffer(2)]],
                  uint gid [[thread_position_in_grid]]) {
    y[gid] = x[gid] + one;
}

struct sargs { uint G; uint limit; uint P; uint n; uint nthreads; };

// every threadgroup checks in once, then all threads wait (bounded) until all G have arrived
kernel void probe(device atomic_uint * s [[buffer(0)]], device uint * out [[buffer(1)]], constant sargs & a [[buffer(2)]],
                  threadgroup float * sh [[threadgroup(0)]],
                  uint tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    sh[tid] = 1.0f;
    if (tid == 0) atomic_fetch_add_explicit(&s[0], 1, memory_order_relaxed);
    uint budget = a.limit, seen = 0;
    while ((seen = atomic_load_explicit(&s[0], memory_order_relaxed)) < a.G && budget) budget--;
    if (tid == 0) { out[2*tg] = seen; out[2*tg + 1] = a.limit - budget; }
}

// P dependent phases over n floats in one dispatch; each phase reads what other threads wrote in the previous one
// MODE 0: every thread checks in and spins
// MODE 1: threadgroup barrier, thread 0 checks in, every thread spins
// MODE 2: threadgroup barrier, thread 0 checks in and spins, threadgroup barrier releases the rest
// MODE 3: no sync (timing baseline, result is wrong)
kernel void phases(device atomic_uint * s [[buffer(0)]], device float * a [[buffer(1)]], device float * b [[buffer(2)]],
                   constant sargs & p [[buffer(3)]],
                   ushort tid [[thread_index_in_threadgroup]], uint gid [[thread_position_in_grid]]) {
    uint budget = p.limit;
    for (uint ph = 0; ph < p.P; ph++) {
        // DATA 0: ping-pong between two buffers, 1: same through volatile pointers, 2: a fresh region per phase
#if DATA == 2 || DATA == 6
        // DATA 6: regions start 5 floats in, so neighbouring threadgroups share cache lines
        device const float * src = a + (DATA == 6 ? 5 : 0) + ph*p.n;
        device       float * dst = a + (DATA == 6 ? 5 : 0) + (ph + 1)*p.n;
#elif DATA == 1
        device const volatile float * src = (ph & 1) ? b : a;
        device       volatile float * dst = (ph & 1) ? a : b;
#else
        device const float * src = (ph & 1) ? b : a;
        device       float * dst = (ph & 1) ? a : b;
#endif
#if DATA == 3
        // ping-pong, read offset loaded from memory
        const uint shift = ((device const uint *) s)[2];
#elif DATA == 4
        // ping-pong, read offset changes every phase: no thread reads an address twice in a row
        const uint shift = 977*(ph + 1);
#else
        const uint shift = 977;
#endif
#if DATA == 5
        // ping-pong, but every shared value moves through atomic load/store
        device atomic_uint * asrc = (device atomic_uint *) ((ph & 1) ? b : a);
        device atomic_uint * adst = (device atomic_uint *) ((ph & 1) ? a : b);
        for (uint i = gid; i < p.n; i += p.nthreads) {
            const float v = as_type<float>(atomic_load_explicit(&asrc[(i + shift) % p.n], memory_order_relaxed)) + 1.0f;
            atomic_store_explicit(&adst[i], as_type<uint>(v), memory_order_relaxed);
        }
#else
        for (uint i = gid; i < p.n; i += p.nthreads) dst[i] = src[(i + shift) % p.n] + 1.0f;
#endif
#if MODE == 0
        atomic_fetch_add_explicit(&s[0], 1, memory_order_relaxed);
        const uint target = p.nthreads * (ph + 1);
        while (atomic_load_explicit(&s[0], memory_order_relaxed) < target && budget) budget--;
#elif MODE == 1
        threadgroup_barrier(mem_flags::mem_none);
        if (tid == 0) atomic_fetch_add_explicit(&s[0], 1, memory_order_relaxed);
        const uint target = p.G * (ph + 1);
        while (atomic_load_explicit(&s[0], memory_order_relaxed) < target && budget) budget--;
#elif MODE == 2 || MODE == 4
        // MODE 4: as 2, with device-memory fences on both threadgroup barriers
        threadgroup_barrier(MODE == 4 ? mem_flags::mem_device : mem_flags::mem_none);
        if (tid == 0) {
            atomic_fetch_add_explicit(&s[0], 1, memory_order_relaxed);
            const uint target = p.G * (ph + 1);
            while (atomic_load_explicit(&s[0], memory_order_relaxed) < target && budget) budget--;
        }
        threadgroup_barrier(MODE == 4 ? mem_flags::mem_device : mem_flags::mem_none);
#endif
    }
    if (budget == 0) atomic_store_explicit(&s[1], 1, memory_order_relaxed);
}

#define NR0 2
#define NSG 4
struct blk { half d; char qs[32]; };
struct margs { uint M; uint nb; uint G; uint limit; uint ncopy; uint sync; };

// same layout as ggml kernel_mul_mv_q8_0_f32 (NR0 rows per threadgroup, NSG simdgroups share the row blocks)
static inline void mv_unit(device const blk * W, device const float * x, device float * y, threadgroup float * sh,
                           uint r0, uint M, uint nb, ushort tiisg, ushort sgitg) {
    device const blk * ax[NR0];
    for (short row = 0; row < NR0; ++row) ax[row] = W + (ulong) min(r0 + row, M - 1) * nb;
    float sumf[NR0] = { 0.f };
    const short ix = tiisg/4;
    const short il = tiisg%4;
    const int ib0 = sgitg*8 + ix;
    float yl[8];
    device const float * yb = x + ib0*32 + il*8;
    for (int ib = ib0; ib < (int) nb; ib += NSG*8) {
        for (short i = 0; i < 8; ++i) yl[i] = yb[i];
        for (short row = 0; row < NR0; row++) {
            device const char * qs = ax[row][ib].qs + il*8;
            float sumq = 0.f;
            for (short i = 0; i < 8; ++i) sumq += qs[i] * yl[i];
            sumf[row] += sumq*ax[row][ib].d;
        }
        yb += NSG*8*32;
    }
    for (short row = 0; row < NR0; ++row) {
        if (sgitg == 0) sh[32*row + tiisg] = 0.0f;
        sumf[row] = simd_sum(sumf[row]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short row = 0; row < NR0; ++row) if (tiisg == 0) sh[32*row + sgitg] = sumf[row];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short row = 0; row < NR0 && r0 + row < M; ++row) {
        float tot = simd_sum(sh[32*row + tiisg]);
        if (tiisg == 0 && sgitg == 0) y[r0 + row] = tot;
    }
}

kernel void mv(device const blk * W [[buffer(0)]], device const float * x [[buffer(1)]], device float * y [[buffer(2)]],
               constant margs & a [[buffer(3)]], threadgroup float * sh [[threadgroup(0)]],
               uint tg [[threadgroup_position_in_grid]], ushort tiisg [[thread_index_in_simdgroup]],
               ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    mv_unit(W, x, y, sh, tg*NR0, a.M, a.nb, tiisg, sgitg);
}

// persistent: G threadgroups stay alive across ncopy matvecs, with a grid sync after each
kernel void mvp(device const blk * W [[buffer(0)]], device const float * x [[buffer(1)]], device float * y [[buffer(2)]],
                constant margs & a [[buffer(3)]], device atomic_uint * s [[buffer(4)]], threadgroup float * sh [[threadgroup(0)]],
                uint tg [[threadgroup_position_in_grid]], ushort tiisg [[thread_index_in_simdgroup]],
                ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    uint budget = a.limit;
    const uint nu = (a.M + NR0 - 1)/NR0;
    for (uint c = 0; c < a.ncopy; c++) {
        device const blk * Wc = W + (ulong) c*a.M*a.nb;
        device float * yc = y + c*a.M;
        for (uint u = tg; u < nu; u += a.G) {
            mv_unit(Wc, x, yc, sh, u*NR0, a.M, a.nb, tiisg, sgitg);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (a.sync != 0) {
            if (tiisg == 0 && sgitg == 0) {
                atomic_fetch_add_explicit(&s[0], 1, memory_order_relaxed);
                const uint target = a.G*(c + 1);
                while (atomic_load_explicit(&s[0], memory_order_relaxed) < target && budget) budget--;
            }
            threadgroup_barrier(mem_flags::mem_none);
        }
    }
    if (budget == 0) atomic_store_explicit(&s[1], 1, memory_order_relaxed);
}
)";

static id<MTLDevice> dev;
static id<MTLCommandQueue> q;

static id<MTLComputePipelineState> pipe(const char * fn, int mode = 0, int data = 0) {
    NSError * err = nil;
    MTLCompileOptions * opt = [MTLCompileOptions new];
    opt.preprocessorMacros = @{ @"MODE" : @(mode), @"DATA" : @(data) };
    id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:SRC] options:opt error:&err];
    if (!lib) { printf("compile: %s\n", err.localizedDescription.UTF8String); exit(1); }
    id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:[NSString stringWithUTF8String:fn]] error:&err];
    if (!ps) { printf("pipeline: %s\n", err.localizedDescription.UTF8String); exit(1); }
    return ps;
}

static id<MTLBuffer> buf(size_t n) { return [dev newBufferWithLength:n options:MTLResourceStorageModeShared]; }

static double run(id<MTLCommandBuffer> cb) {
    [cb commit]; [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) { printf("command buffer failed: %s\n", cb.error.localizedDescription.UTF8String); exit(2); }
    return cb.GPUEndTime - cb.GPUStartTime;
}

struct sargs { uint32_t G, limit, P, n, nthreads; };
struct margs { uint32_t M, nb, G, limit, ncopy, sync; };

// a: N dependent tiny dispatches. kind 0 = concurrent + barrier, 1 = concurrent without barrier, 2 = serial encoder
static void test_a() {
    id<MTLComputePipelineState> ps = pipe("step"), ps2 = pipe("step2");
    const int N = 4000;
    const float one = 1.0f;
    printf("a: dispatch boundary cost, %d chained dispatches per command buffer\n", N);
    for (int n : {32, 2304}) {
        id<MTLBuffer> x = buf(n*4), y = buf(n*4);
        // kind 3 = concurrent + barrier, alternating two pipelines with fresh arguments (as a real graph does)
        for (int width : {1, 4}) for (int kind = 0; kind < 4; kind++) {
            double best = 1e9; bool ok = true;
            for (int r = 0; r < 6; r++) {
                memset(x.contents, 0, n*4); memset(y.contents, 0, n*4);
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> e = kind == 2 ? [cb computeCommandEncoder] : [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
                [e setComputePipelineState:ps];
                for (int i = 0; i < N; i++) {
                    if (kind == 3) { [e setComputePipelineState:(i & 1) ? ps2 : ps]; [e setBytes:&one length:4 atIndex:2]; }
                    [e setBuffer:(i & 1) ? y : x offset:0 atIndex:0]; [e setBuffer:(i & 1) ? x : y offset:0 atIndex:1];
                    // width > 1: independent dispatches at the same level (they rewrite the same result)
                    for (int w = 0; w < width; w++) [e dispatchThreadgroups:MTLSizeMake((n + 255)/256, 1, 1) threadsPerThreadgroup:MTLSizeMake(n < 256 ? n : 256, 1, 1)];
                    if (kind == 0 || kind == 3) [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
                }
                [e endEncoding];
                const double t = run(cb);
                if (r > 0) best = std::min(best, t);
                if (kind != 1) for (int i = 0; i < n; i++) ok &= ((float *) x.contents)[i] == (float) N;
            }
            printf("  n %4d width %d %-22s %6.3f us/level%s\n", n, width,
                kind == 0 ? "concurrent+barrier" : kind == 1 ? "concurrent no barrier" : kind == 2 ? "serial" : "barrier, 2 pipelines", best*1e6/N, ok ? "" : "  WRONG RESULT");
        }
    }
}

// returns true when all G threadgroups saw each other; spins = slowest wait in loop iterations
static bool probe_once(id<MTLComputePipelineState> ps, id<MTLBuffer> s, id<MTLBuffer> out, uint32_t G, int T, int tgmem, uint32_t * spins) {
    memset(s.contents, 0, 16);
    sargs a = { G, 1u << 18, 0, 0, 0 };
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
    [e setComputePipelineState:ps];
    [e setBuffer:s offset:0 atIndex:0]; [e setBuffer:out offset:0 atIndex:1]; [e setBytes:&a length:sizeof a atIndex:2];
    [e setThreadgroupMemoryLength:tgmem atIndex:0];
    [e dispatchThreadgroups:MTLSizeMake(G, 1, 1) threadsPerThreadgroup:MTLSizeMake(T, 1, 1)];
    [e endEncoding];
    run(cb);
    const uint32_t * o = (const uint32_t *) out.contents;
    bool ok = true; uint32_t mx = 0;
    for (uint32_t i = 0; i < G; i++) { ok &= o[2*i] >= G; mx = std::max(mx, o[2*i + 1]); }
    if (spins) *spins = mx;
    return ok;
}

static void test_b_probe() {
    id<MTLComputePipelineState> ps = pipe("probe");
    id<MTLBuffer> s = buf(16), out = buf(8 << 20);
    printf("b1: max co-resident threadgroups (all G check in before any proceeds; 5/5 trials)\n");
    for (int tgmem : {4096, 16384, 32768}) for (int T : {32, 64, 128, 256, 512, 1024}) {
        if (tgmem < T*4) continue;
        auto ok = [&](uint32_t G) { for (int r = 0; r < 5; r++) if (!probe_once(ps, s, out, G, T, tgmem, nullptr)) return false; return true; };
        uint32_t lo = 1, hi = 2;
        while (hi <= 65536 && ok(hi)) { lo = hi; hi *= 2; }
        while (hi - lo > 1) { const uint32_t mid = (lo + hi)/2; if (ok(mid)) lo = mid; else hi = mid; }
        uint32_t spins = 0; probe_once(ps, s, out, lo, T, tgmem, &spins);
        printf("  tgmem %5d threads/tg %4d: max G %5u (%6u threads), slowest wait %u spins\n", tgmem, T, lo, lo*T, spins);
    }
}

// returns seconds, or -1 on timeout / wrong result
static double phases_once(id<MTLComputePipelineState> ps, id<MTLBuffer> s, id<MTLBuffer> a, id<MTLBuffer> b, uint32_t G, int T, uint32_t P, uint32_t n, bool check, int data = 0) {
    memset(s.contents, 0, 16); memset(a.contents, 0, a.length); memset(b.contents, 0, n*4);
    ((uint32_t *) s.contents)[2] = 977;
    sargs p = { G, 1u << 20, P, n, G*T };
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
    [e setComputePipelineState:ps];
    [e setBuffer:s offset:0 atIndex:0]; [e setBuffer:a offset:0 atIndex:1]; [e setBuffer:b offset:0 atIndex:2]; [e setBytes:&p length:sizeof p atIndex:3];
    [e dispatchThreadgroups:MTLSizeMake(G, 1, 1) threadsPerThreadgroup:MTLSizeMake(T, 1, 1)];
    [e endEncoding];
    const double t = run(cb);
    if (!check) return t;
    const bool timeout = ((const uint32_t *) s.contents)[1] != 0;
    const float * r = data == 2 || data == 6 ? (const float *) a.contents + (data == 6 ? 5 : 0) + (size_t) P*n : (const float *) ((P & 1) ? b : a).contents;
    uint32_t wrong = 0;
    for (uint32_t i = 0; i < n; i++) wrong += r[i] != (float) P;
    if (getenv("SYNC_DEBUG") && (timeout || wrong)) {
        printf("    T %d G %u: timeout %d, wrong %u/%u (r[0] %.0f, want %u), counter %u, %.3f us/phase\n",
            T, G, timeout, wrong, n, r[0], P, ((const uint32_t *) s.contents)[0], t*1e6/P);
    }
    return timeout || wrong ? -1 : t;
}

static void test_b_phases() {
    const uint32_t P = 2000, n = 2304;
    const int data = getenv("DATA") ? atoi(getenv("DATA")) : 2;
    id<MTLBuffer> s = buf(16), a = buf((size_t) n*4*(P + 2)), b = buf(n*4);
    printf("b2: in-kernel grid sync, %u dependent phases over %u floats in one dispatch, data layout %d\n", P, n, data);
    id<MTLComputePipelineState> ps[5];
    for (int m = 0; m < 5; m++) ps[m] = pipe("phases", m, data);
    const char * names[5] = { "all threads inc+spin", "tg barrier, t0 inc, all spin", "tg barrier, t0 inc+spin", "no sync (baseline)", "mem_device barrier, t0 inc+spin" };
    const bool quick = getenv("QUICK") != nullptr;
    for (auto cfg : std::initializer_list<std::pair<int, int>>{{32, 8}, {32, 72}, {128, 18}, {128, 80}, {256, 9}, {256, 40}, {512, 40}, {1024, 3}, {1024, 10}, {1024, 20}}) {
        // ONE_TG: a single threadgroup, so the threadgroup barrier is the only sync that matters
        const int T = cfg.first, G = getenv("ONE_TG") ? 1 : cfg.second;
        if (getenv("ONE_TG") && cfg.second != 3) continue;
        if (quick && !(T == 128 && G == 18)) continue;
        for (int m = 0; m < 5; m++) {
            double best = 1e9; int bad = 0;
            for (int r = 0; r < 20; r++) {
                const double t = phases_once(ps[m], s, a, b, G, T, P, n, m != 3, data);
                if (t < 0) bad++; else if (r > 0) best = std::min(best, t);
            }
            printf("  T %4d G %3d %-32s %7.3f us/phase  failures %d/20\n", T, G, names[m], best < 1e8 ? best*1e6/P : NAN, bad);
        }
    }
}

static void soak(int runs) {
    const uint32_t P = 2000, n = 2304;
    id<MTLBuffer> s = buf(16), a = buf((size_t) n*4*(P + 2)), b = buf(n*4);
    const int data = getenv("DATA") ? atoi(getenv("DATA")) : 2;
    const int mode = getenv("MODE") ? atoi(getenv("MODE")) : 4;
    id<MTLComputePipelineState> ps = pipe("phases", mode, data);
    printf("soak: sync mode %d, data layout %d\n", mode, data);
    for (auto cfg : std::initializer_list<std::pair<int, int>>{{128, 18}, {256, 40}, {1024, 10}}) {
        int bad = 0; double worst = 0, sum = 0;
        for (int r = 0; r < runs; r++) {
            const double t = phases_once(ps, s, a, b, cfg.second, cfg.first, P, n, true, data);
            if (t < 0) bad++; else { worst = std::max(worst, t); sum += t; }
        }
        printf("soak T %4d G %3d: %d runs x %u syncs, failures %d, mean %.3f us/phase, worst run %.3f us/phase\n",
            cfg.first, cfg.second, runs, P, bad, sum*1e6/P/(runs - bad), worst*1e6/P);
    }
}

static void test_c() {
    const uint32_t M = 3072, K = 2304, nb = K/32, NCOPY = 32, NR0 = 2, NSG = 4;
    const size_t wbytes = (size_t) M*nb*34;
    id<MTLBuffer> W = buf(wbytes*NCOPY), X = buf(K*4), Y = buf(M*4*NCOPY), S = buf(16);
    uint8_t * wp = (uint8_t *) W.contents; float * xp = (float *) X.contents;
    srand(1);
    for (size_t i = 0; i < (size_t) M*nb*NCOPY; i++) {
        *(__fp16 *)(wp + i*34) = (__fp16)((rand() % 1000 + 1)/1000.0f*0.02f);
        for (int j = 0; j < 32; j++) wp[i*34 + 2 + j] = (uint8_t) rand();
    }
    for (uint32_t i = 0; i < K; i++) xp[i] = (rand() % 2000 - 1000)/1000.0f;
    std::vector<float> ref((size_t) M*NCOPY);
    for (size_t r = 0; r < (size_t) M*NCOPY; r++) {
        double acc = 0;
        for (uint32_t b = 0; b < nb; b++) {
            const uint8_t * p = wp + (r*nb + b)*34;
            double sq = 0;
            for (int j = 0; j < 32; j++) sq += (int8_t) p[2 + j]*(double) xp[b*32 + j];
            acc += sq*(double) *(const __fp16 *) p;
        }
        ref[r] = (float) acc;
    }
    auto maxerr = [&]() { float e = 0; const float * y = (const float *) Y.contents;
        for (size_t r = 0; r < (size_t) M*NCOPY; r++) e = fmaxf(e, fabsf(y[r] - ref[r])/(fabsf(ref[r]) + 1e-2f)); return e; };
    id<MTLComputePipelineState> pn = pipe("mv"), pp = pipe("mvp");
    printf("c: Q8_0-like matvec %ux%u, %.2f MB, %u copies streamed per run\n", M, K, wbytes/1e6, NCOPY);
    for (int barrier = 1; barrier >= 0; barrier--) {
        double best = 1e9;
        for (int r = 0; r < 8; r++) {
            memset(Y.contents, 0, M*4*NCOPY);
            margs a = { M, nb, 0, 0, NCOPY, 0 };
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
            [e setComputePipelineState:pn]; [e setBuffer:X offset:0 atIndex:1]; [e setBytes:&a length:sizeof a atIndex:3];
            [e setThreadgroupMemoryLength:32*4*NR0 atIndex:0];
            for (uint32_t c = 0; c < NCOPY; c++) {
                [e setBuffer:W offset:wbytes*c atIndex:0]; [e setBuffer:Y offset:(size_t) M*4*c atIndex:2];
                [e dispatchThreadgroups:MTLSizeMake(M/NR0, 1, 1) threadsPerThreadgroup:MTLSizeMake(32*NSG, 1, 1)];
                if (barrier) [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }
            [e endEncoding];
            const double t = run(cb);
            if (r > 0) best = std::min(best, t);
        }
        printf("  one dispatch per matvec, %-10s           %7.1f us/matvec %6.1f GB/s  max rel err %.1e\n",
            barrier ? "barriers" : "no barrier", best*1e6/NCOPY, wbytes*NCOPY/best/1e9, maxerr());
    }
    for (int sync = 1; sync >= 0; sync--) for (uint32_t G : {10, 20, 40, 60, 80, 120, 160, 240, 320}) {
        double best = 1e9; int bad = 0;
        for (int r = 0; r < 8; r++) {
            memset(Y.contents, 0, M*4*NCOPY); memset(S.contents, 0, 16);
            margs a = { M, nb, G, 1u << 18, NCOPY, (uint32_t) sync };
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
            [e setComputePipelineState:pp];
            [e setBuffer:W offset:0 atIndex:0]; [e setBuffer:X offset:0 atIndex:1]; [e setBuffer:Y offset:0 atIndex:2];
            [e setBytes:&a length:sizeof a atIndex:3]; [e setBuffer:S offset:0 atIndex:4];
            [e setThreadgroupMemoryLength:32*4*NR0 atIndex:0];
            [e dispatchThreadgroups:MTLSizeMake(G, 1, 1) threadsPerThreadgroup:MTLSizeMake(32*NSG, 1, 1)];
            [e endEncoding];
            const double t = run(cb);
            if (((const uint32_t *) S.contents)[1]) bad++; else if (r > 0) best = std::min(best, t);
        }
        printf("  persistent G %3u (%5u threads) %-8s  %7.1f us/matvec %6.1f GB/s  max rel err %.1e  timeouts %d/8\n",
            G, G*32*NSG, sync ? "sync" : "no sync", best*1e6/NCOPY, wbytes*NCOPY/best/1e9, maxerr(), bad);
    }
}

int main(int argc, char ** argv) {
    @autoreleasepool {
        setvbuf(stdout, nullptr, _IOLBF, 0);
        dev = MTLCreateSystemDefaultDevice();
        q = [dev newCommandQueue];
        const char * w = argc > 1 ? argv[1] : "abc";
        printf("GPU %s\n", dev.name.UTF8String);
        if (!strcmp(w, "soak")) { soak(argc > 2 ? atoi(argv[2]) : 1000); return 0; }
        if (strchr(w, 'a')) test_a();
        if (!strcmp(w, "b1") || !strcmp(w, "abc")) test_b_probe();
        if (!strcmp(w, "b2") || !strcmp(w, "abc")) test_b_phases();
        if (strchr(w, 'c')) test_c();
    }
    return 0;
}
