// Peak streaming-read bandwidth probe: each thread sums float4s with a grid stride.
#import <Metal/Metal.h>
#include <stdio.h>
#include <initializer_list>
static const char * SRC =
"#include <metal_stdlib>\nusing namespace metal;\n"
"kernel void rd(device const float4 * a [[buffer(0)]], device float * out [[buffer(1)]], constant uint & n [[buffer(2)]],\n"
"               uint gid [[thread_position_in_grid]], uint gsz [[threads_per_grid]]) {\n"
"  float4 s0 = 0, s1 = 0, s2 = 0, s3 = 0;\n"
"  uint i = gid;\n"
"  for (; i + 3*gsz < n; i += 4*gsz) { s0 += a[i]; s1 += a[i+gsz]; s2 += a[i+2*gsz]; s3 += a[i+3*gsz]; }\n"
"  for (; i < n; i += gsz) s0 += a[i];\n"
"  float4 s = s0 + s1 + s2 + s3;\n"
"  if (s.x + s.y + s.z + s.w == 12345.678f) out[0] = 1;\n"
"}\n";
int main() {
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    NSError * err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:SRC] options:nil error:&err];
    if (!lib) { printf("%s\n", err.localizedDescription.UTF8String); return 1; }
    id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"rd"] error:&err];
    const size_t bytes = 1ull << 30;
    id<MTLBuffer> a = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    memset(a.contents, 1, bytes);
    id<MTLBuffer> o = [dev newBufferWithLength:16 options:MTLResourceStorageModeShared];
    uint n = (uint)(bytes / 16);
    id<MTLCommandQueue> q = [dev newCommandQueue];
    printf("GPU %s, max threads/tg %lu\n", dev.name.UTF8String, (unsigned long)ps.maxTotalThreadsPerThreadgroup);
    for (int tg : {256, 1024}) for (int ntg : {80, 320, 1280, 5120, 20480}) {
      double best = 0;
      for (int r = 0; r < 6; r++) {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:ps]; [e setBuffer:a offset:0 atIndex:0]; [e setBuffer:o offset:0 atIndex:1]; [e setBytes:&n length:4 atIndex:2];
        [e dispatchThreadgroups:MTLSizeMake(ntg,1,1) threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        double gbs = bytes / (cb.GPUEndTime - cb.GPUStartTime) / 1e9;
        if (r > 0 && gbs > best) best = gbs;
      }
      printf("tg %4d x %6d threadgroups: %6.1f GB/s\n", tg, ntg, best);
    }
  }
  return 0;
}
