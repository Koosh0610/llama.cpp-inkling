# llama.cpp-inkling

llama.cpp tuned for Inkling 2 Mini on a base Apple M5 (10-core GPU, 24 GB unified memory). No change trades away quality.

**Write-up with diagrams of every change: https://koosh0610.github.io/llama.cpp-inkling/**

| Inkling 2 Mini UD-Q3_K_XL, base M5 | prefill (pp512) | decode |
|---|---|---|
| upstream llama.cpp [1] | 42.8 tok/s | 15.9 tok/s |
| MLX, 4-bit [2] | 411-421 tok/s | 27.2-27.7 tok/s |
| **this repo** | **362 tok/s** | **28.4 tok/s** |

- **8.5x faster prefill and 1.8x faster decode than upstream llama.cpp.** Upstream has no Metal kernels for NVFP4, the format the experts ship in, so they run on the CPU. Here they run on the GPU.
- **Faster decode than MLX on the same Mac**, while keeping 8-bit attention. MLX still leads on prefill.
- **Same quality as upstream:** perplexity 4.414 vs 4.423, KL divergence 0.0018 against upstream's logits, same top token 98.6% of the time.
- **78% of the hardware limit:** each token reads about 3.45 GB of weights. At the ~125 GB/s the GPU can stream (measured), decode tops out near 36 tok/s.
- **85 tok/s when rewriting a file**, 3.5x plain decoding, with n-gram drafts plus gated DSpark speculative decoding.

[1] Upstream 46eb174 plus the required fix #32704, llama-bench, flash attention. In a side-by-side server test on the same model file: upstream 16.9-17.8 tok/s decode, this repo 28.8-29.0 tok/s.

[2] mlx-community/Inkling-2-Mini-4bit with mlx-lm PR #2086 on the same Mac. Different quant: MLX uses NVFP4/MXFP8 experts and 4-bit attention, this repo uses Q8_0 attention.

## What made it fast

| change | measured effect |
|---|---|
| Native NVFP4 Metal kernels: mat-vec, MoE mat-vec, mat-mat | decode 15.9 -> 24.2, prefill 42.8 -> 251 |
| Fused row L2-norm clip (about 1000 dispatches per token) | decode 24.2 -> 25.1, prefill 251 -> 263 |
| Fused softcapped SITU gate (9 ops -> 1) | decode 25.1 -> 25.6, prefill 263 -> 280 |
| Fused loop controller (2 small mat-vecs and 6 ops -> 1) | decode +0.8% |
| F16 router and loop-control weights (shipped as F32) | decode +3.4%, KL divergence unchanged |
| Fused conv residual | decode +2.2% |
| Wider Metal graph reorder | decode 27.6 -> 28.4, identical logits |
| Fused depth mix (6 ops -> 1) | decode +2.1% |
| Depth-mix cache and fused mix for batches | prefill 288 -> 330 |
| NVFP4 dequant without table lookups | prefill 330 -> 343 |
| Fused layer tail and relative-position bias | prefill 343 -> 362 |

Each row is measured against the build just before it.

The lesson: decode on Apple GPUs is limited by memory-barrier depth (about 1.5 us per level), not by kernel launches. Dispatches that run concurrently are nearly free. The fusions and the reorder cut barrier levels per token from about 3,260 to about 1,780.

## Speculative decoding

Server, temperature 0, tok/s:

| | prose | code | rewrite a file |
|---|---|---|---|
| plain decoding | 27.1-28.6 | 27.9 | 24.6 |
| DSpark as in PR #32218 | 12.7-15.2 | 23.6 | 39.7 |
| DSpark, gated (this repo) | 25.7-26.6 | 28.2 | 39.5 |
| n-gram drafts (upstream `ngram-mod`) | 28.6-28.8 | 28.0 | 78.0 |
| n-gram, then gated DSpark | 26.5-26.9 | 28.4 | **85.1** |

The gate drafts only tokens the drafter gives probability >= 0.8, and pauses drafting after tries that lose time (32 tokens, doubling up to 256). Use n-gram drafts by default and add gated DSpark for copy-heavy work. Code and copy outputs are identical to plain decoding.

## What did not help

- Graph capture: at most 0.6 ms per token.
- A persistent megakernel: Metal atomics are relaxed-only, so one threadgroup can read stale data written by another.
- Caching the looped layers' weights: one loop pass reads about 780 MB, far more than anything on-chip.
- Splitting command buffers across more CPU threads.
- Lower-bit attention: Q6_K gave +4% decode but KL divergence went from 0.0017 to 0.0029, so it was dropped.

## Try it

```sh
cmake -B build -DGGML_METAL=ON
cmake --build build -j --target llama-bench llama-server
./build/bin/llama-bench -m Inkling-2-Mini-UD-Q3_K_XL.gguf -fa 1 -p 512 -n 128
./build/bin/llama-server -m Inkling-2-Mini-UD-Q3_K_XL.gguf -fa on --spec-type ngram-mod
```

- Model: the mainline llama.cpp UD-Q3_K_XL GGUF from unsloth/Inkling-2-Mini-GGUF (13.6 GB). The command for the F16 router requant is in the commit `metal, quantize : F16 router and loop-control weights`.
- DSpark: add `--spec-type ngram-mod,draft-dspark -md dspark-Inkling-2-Mini-Q8_0.gguf --spec-draft-n-max 7 --spec-draft-p-min 0.8`.
- To turn a fusion off, set one of `GGML_METAL_NO_LOOP_CTRL`, `GGML_METAL_NO_CONV_RES`, `GGML_METAL_NO_MIX`, `GGML_METAL_NO_CONV_TAIL`, `GGML_METAL_NO_REL_BIAS`, `GGML_METAL_NO_GROUPED` to 1.

## History

The first commit is upstream llama.cpp at 46eb174 (same tree hash). Each later commit is one kernel or patch, in the order it was built, with its measurements in the message, so `git log --reverse` reads as the story.

- `54ce6c4` native NVFP4 kernels, `7aca306` grouped NVFP4 MoE kernel for 2-8 tokens, `51bac5a` NVFP4 dequant without table lookups
- tag `m5-benchmarked-build`: the exact tree that was benchmarked
- `fc9388d`, `4b7c0dc`, `c757228`: DSpark (PR #32218), its Markov-step cap, and the gate
- `inkling-engine/`: the logit checker (dcheck), a side-by-side chat UI against upstream, microbenchmarks, and TODO.md with every number

## Caveats

- One machine and one quant. Benchmark noise is about +-0.6 tok/s. Measured on battery with Low Power Mode off (Low Power Mode cuts bandwidth to about 80 GB/s).
- UD-Q3_K_XL cuts 8 of the 40 released NVFP4 expert tensors to 2-3.4 bits. The quality numbers compare this engine with upstream on the same quant, not the quant with the released weights. UD-Q4_K_XL (17.6 GB) keeps all NVFP4 experts but needs a higher GPU memory limit on a 24 GB Mac (`sudo sysctl iogpu.wired_limit_mb=...`).
- Nothing here is upstream yet. Debug switches (`GGML_METAL_PROFILE`, `GGML_METAL_GRAPH_DUMP`, `INK_SKIP_*`) are still in the tree.

## Credits

The llama.cpp and ggml contributors. Daniel Han for the inkling2_loop fix (#32704), and Unsloth for the GGUFs. Piotr Wilkin for DSpark (PR #32218). [Locals](https://github.com/Generated-Computers/Locals) for the idea of fusing every stretch of small ops between weight reads, and for the gating design.

For everything else, see the [upstream README](https://github.com/ggml-org/llama.cpp/blob/46eb1742db6d2caec1bfa848c5702b1bdcc815e5/README.md).
