# Inkling 2 Mini on M5 (base, 10-core GPU, 24GB, ~125GB/s measured)

Goal: beat every existing Mac setup for Inkling 2 Mini.

IMPORTANT: benchmark only with Low Power Mode off (pmset -g | grep lowpowermode). LPM cut bandwidth to ~80GB/s and
invalidated an hour of results. Check before every benchmark run.

IMPORTANT: mainline needs upstream PR #32704 (applied locally) or Inkling 2 Mini outputs gibberish (PPL ~67 instead of 4.4).
CPU-only mode is still broken upstream (PPL ~140k), separate bug.

## Results (LPM off, on battery, Unsloth mainline UD-Q3_K_XL 13.6GB, flash attention on, all with #32704)
- llama-bench pp512 / tg128:
  - Stock (~/src/llama-stock = upstream + #32704, experts on CPU): 42.8 / 15.9
  - Mine (~/src/llama.cpp, native NVFP4 Metal): 251 / 24.2 (5.9x / 1.52x)
- Server, 4 chat prompts, temp 0, ~/src/llama-dspark/build-nv:
  - no speculation 23.84; DSpark n=3 as shipped 21.1; DSpark n=2 + Markov cap 24.6 (+3%); code prompt 30.1
- Quality: PPL 4.412 vs stock 4.423, KLD 0.0019, same top token 98.8%
- Ceiling: ~3.4GB read per token at ~125GB/s, about 36-37 tok/s single-stream

## Per-token budget at full power (40.6ms)
- NVFP4 experts 11.1ms (at peak), other expert types 2.6ms, attention Q8_0 9.8ms (at peak, 8-bit weights),
  output head 2.5ms (at peak), router/loop-ctrl F32 ~1.7ms, small ops/dispatch ~12.6ms

## Done
- [x] #1 Full-power profile (GPU timestamp profiler)
- [x] #2 Quality check: found and applied upstream #32704; KLD vs stock 0.0019
- [x] #3 DSpark: works with my kernels; found drafter computing 7 Markov steps (7 full-vocab head reads) regardless of
      n_max; capped via LLAMA_DSPARK_MARKOV_STEPS (identical drafts). Net +3% at n=2 on prose, +27% on code.
      Limit is structural: verify cost ~linear in tokens (experts rarely shared), each draft step reads the 318MB head.
- [x] #5 Output head: already at 129GB/s at full power
- [x] Grouped small-batch MoE kernel (kernel_mul_mv_id_nvfp4_grouped): correct, small gain at 8 tokens, ~neutral at 4
- [x] Native NVFP4 Metal kernels (mul_mv, mul_mv_id, mul_mm, mul_mm_id), 125/125 tests

## Next
4. [~] Fuse Inkling elementwise chains (~12.6ms of small ops per token)
   - [x] L2_CLIP (q/k/rel, ~1000 dispatches/token): tg 24.17 -> 25.13, pp 251 -> 263, KLD 0.0018
   - [x] SITU softcapped gate (9 ops -> 1): tg 25.56, pp 280, KLD 0.0017
   - [x] LOOP_CTRL (2 F32 matvecs + 6 ops -> 1, <=8 tokens): +0.8% only; outputs identical on 3/4 prompts,
         4th diverges at char 530/826 (near-tie). Low gain implies per-dispatch cost is small: the residual
         "small ops" bucket is probably not mostly dispatch overhead. Fable review launched to re-examine.
   - [x] CONV_RES: SSM_CONV+ADD+SCALE -> 1 (state CPY is emitted earlier in builder order, so not fusable with it).
         Needed a raw-span subgraph check (conv reaches ADD through a RESHAPE view). +2.2% (27.0 vs 26.4), outputs identical.
   - [x] mix prefix cache (single token): normalized block stack + transpose cached per block list. Outputs identical,
         speed gain not measurable (~180 dispatches); a real gain needs a fused mix kernel (6 ops -> 1).
   Upper bounds measured by skipping blocks entirely (wrong output, timing only): convolve ~2.0 ms, mix ~1.3-2.0 ms.

## Barrier work (Nov 21, evening)
- Finding: decode is GPU-bound (GPU busy ~35.0 of 35.6 ms/token); cost tracks memory barriers (~1.5 us each), not
  dispatches (removing 199 concurrent CONTs: 0 gain). Graph capture would recover <=0.6 ms, not worth it.
- [x] Reorder whitelist (ggml-metal-common.cpp h_safe += CONCAT, SOFT_MAX, FILL, PAD, SQR, SQRT, LOG, FLASH_ATTN_EXT):
      barriers 3257 -> 2656, 27.6 -> 28.4 tok/s, bit-identical logits
- [x] MIX fusion (rms_norm + 2 concat + scores + softmax + weighted sum -> kernel_mix_f32, 120/token):
      barriers -> 2213, +2.1% same-window A/B, mean KL vs unfused <= 8e-5, no top-1 change in 384 tokens
- [x] Tried and dropped: smaller first command buffer / more encoder threads (worse); control-down fusion (neutral);
      N_FORWARD 128/256 (neutral/worse); iq2 "707 us" was a profile-mode GPU clock ramp artifact
- [x] Attention Q6_K requant measured (+4%, KLD 0.0017 -> 0.0029): REJECTED by user, no quality trade-offs
- Tools: ~/src/inkling-engine/dcheck (check.sh greedy logits vs ~/models/inkling2-mini/eval/dref, ab.sh interleaved A/B)
- Bench noise is +-0.6 tok/s when warm/charging: use same-window interleaved A/B for anything under ~2%
- Next exact candidates: conv CONCAT+CPY folded into conv kernel (~190 barriers), attention-bias chain
  (mul_mat/sub/cont/get_rows/cont/add/cpy -> 1, ~240), NVFP4 ffn_down at ~84% BW (~0.5 ms)
- [x] Batched mix (prompt processing): INK_SKIP_MIX ablation showed mix = ~16% of pp512. Cached-prefix mix path now
      used for any n_tokens (+8.4%), MIX fusion extended to (query, token) grid with optional REPEAT (+5.8%):
      pp512 288 -> 330 tok/s. KLD vs stock 0.00180 (was 0.00171, ~1 SE), PPL 4.414, top-1 98.6%
- [x] MLX side-by-side (mlx-community/Inkling-2-Mini-4bit, mlx-lm PR #2086, ~/models/inkling2-mini/mlx-4bit):
      tg128 MLX 27.2-27.7 vs ours 28.1-28.3; pp512 MLX 411-421 vs ours 276-281 (before batched mix work).
      Quants differ: MLX NVFP4/MXFP8 experts + 4-bit attention; ours Q8_0 attention + some IQ2/IQ3 expert layers
- NVFP4 mul_mm_id at n=512 (32 tok/expert): 5.7 TFLOPS vs q4_0 6.9 (tensor API path, dequant-bound)
- [x] NVFP4 dequant without table lookups (e2m1 -> fp16 bits, x2^14 folded into scale, bit-identical): n=512 MoE
      matmul gate_up 7.58 -> 5.96 ms, down 3.81 -> 3.09 ms (7.0-7.3 TFLOPS, now above q4_0); pp512 330 -> 343
- [x] DSpark re-measured on new build (~/src/llama-ds2 = dspark branch + all of today, spec.sh/spec_bench.py):
      server plain decode 29.0 prose / 28.2-29.5 code; DSpark n=2 cap2 21-22.5 prose (42% accept), 25.7-28.8 code.
      Not worth it now: plain decode got ~20% faster, verify still costs ~a token. Fused-vs-unfused acceptance gap
      (57 vs 69% on code) was different completions (batched verify flips a temp-0 near-tie), not a bug.
- GPU shared with agents: wrap every timing run in ~/src/inkling-engine/gpulock.sh
- Loop weights cannot be cached: ~780 MB per loop pass vs tens of MB on-chip; only spec decoding amortizes reads

## Fable review (Nov 21) -> done
- [x] F16 router + loop-ctrl weights (mainline-...-f16ctrl.gguf, 168 tensors, bit-identical KLD): +3.4%
- [x] Identity ops removed in builder (mix query repeat at n=1, expert-scale repeat at n=1, shared zero sinks): ~0, pp +3%
- [-] Attention requant: rejected by user (see Barrier work)
- [ ] n-gram speculation on code; loop.count=2 KLD experiment
- [ ] Measure wall-minus-GPU per token in plain decode; re-run everything on AC power (battery drift seen at 36%)
6. [ ] Fast kernels for remaining non-NVFP4 expert tensors (IQ2_S, IQ3_XXS, IQ3_S, Q3_K, IQ4_NL): ~1ms
7. [-] Persistent per-layer decode kernel: REJECTED (Fable agent, Nov 21, mega/sync.mm). Grid sync via atomics works
      within 30720 co-resident threads, but cross-threadgroup data visibility inside a dispatch is unreliable (stale
      reads 1-8/300 quiet, up to 20/20 contended; Metal atomics are relaxed-only). Gain would be <=0.3 us/level anyway.
      Isolated dispatch+barrier = 0.7 us/level; real-graph level removal saves ~1.5 us (kernel ramp/tail), so fuse.
8. [ ] Self-draft with fewer loop passes (TokenSpeed loop num-passes)
9. [ ] Shared KV across loop passes (TokenSpeed loop share-cache)
10. [-] Attention weights Q8_0 -> Q6_K: measured, rejected (user: no quality compromise)

## Upstream candidates (user owns and submits; confirm per-file plan first; llama.cpp AGENTS.md rules apply)
- NVFP4 Metal kernels: best candidate, needs debug env vars stripped
- DSpark Markov cap: belongs as a comment on PR #32218, needs proper n_max plumbing instead of env var

## Local-only patches
- ~/src/llama.cpp: llama-quant.cpp LLAMA_QUANT_ONLY_MATCHED; ggml-metal-context.m GGML_METAL_PROFILE;
  ggml-metal-device.m profiler hook + NVFP4 enabled; NVFP4 kernels (common.h, dequantize.h, mul_mv.metal, mul_mm.metal,
  ggml-metal-impl.h, ggml-metal-device.cpp); GGML_METAL_MVID_NSG; GGML_METAL_NO_GROUPED; test-backend-ops INKLING_PERF;
  PR #32704
- ~/src/llama-dspark: all of the above + GGML_METAL_GRAPH_TIME, GGML_METAL_PROFILE_MIN/MAX, LLAMA_DSPARK_MARKOV_STEPS

## Spec decoding gating (Nov 22, after Locals review) -> ~/src/llama-ds2, patches/llama-ds2-20261122-gating.patch
- DSpark impl (common/speculative.cpp): p_min now works for Inkling2 (softmax prob of the drafted token on host; the old
  confidence check read context features from embeddings_nextn and broke at once); backoff after losing tries
  (32 doubling to 256 tokens, reset on a win; empty drafts count as losses); LLAMA_DSPARK_GATE=0 turns the gate off.
- Costs on M5: verify of n tokens ~0.92 + 0.32 n steps (pp1..pp8: 33.9, 52.5, 63.9, 73.9, 85.7, 95.3, -, 117 ms);
  drafter pass 29 ms with 7 Markov steps (each reads the target's 318 MB output head), 17.6 ms with 3.
- Server, temp 0, spec_bench.py (3 prose / code / copy = rewrite ui/server.py with one rename, thinking off, 1600 tok):
  | config                                   | prose     | code  | copy  |
  | plain                                    | 27.1-28.6 | 27.9  | 24.6  |
  | dspark n7 ungated                        | 12.7-15.2 | 23.6  | 39.7  |
  | dspark n7 p0.8 + backoff                 | 25.7-26.6 | 28.2  | 39.5  |
  | dspark n7 p0.8 + backoff, Markov cap 3   | 26.4-26.8 | 29.2  | 32.9  |
  | ngram-mod only                           | 28.6-28.8 | 28.0  | 78.0  |
  | ngram-mod, then dspark n7 p0.8 + backoff | 26.5-26.9 | 28.4  | 85.1  |
- Takeaway: ngram-mod alone is the default (free, 3.2x on copy). Gated DSpark on top: +9% copy, +1.5% code, -6% prose
  (the drafter's per-token context injection is paid even while skipped). Same as Locals: prose -4%.
- Outputs: copy and code identical to plain; 2 of 3 prose replies flip at a reasoning near-tie (batched verify).
- [ ] Lever left: Markov steps re-read the 318 MB head; bias only the base top-K rows to make wide drafts ~free.

## 2026-11-22: Locals-style fusions (#1)
- Two new Metal fusions, both on by default, off with GGML_METAL_NO_REL_BIAS / GGML_METAL_NO_CONV_TAIL:
  - REL_BIAS: bias-table matmul, sink, permute/gather and mask add in one kernel, written straight as the F16 mask.
  - CONV_TAIL: the post-attention and post-MoE tails (concat, conv state store, ssm_conv, residual, scale, residual add) in one kernel, one token.
- Graph changes in inkling2.cpp and llama-graph.cpp:
  - The F32 mask is cast once per graph, not 48 times.
  - The sinks are emitted early and each layer tail is emitted before the next layer, so the patterns stay consecutive.
- Barrier levels per token went from 2131 to 1783.
  - tg64 interleaved A/B: 28.0 vs 28.4 tok/s, within noise.
  - pp512: 343 vs 362 tok/s.
- dcheck p1–p3: no top-1 divergence over 128 tokens, mean KL ≤ 2.7e-4, same as with the fusions off. PPL was not rerun.
- Gotcha: the backend alloc deps are VIEW nodes inserted after each pattern's last node.
  - Deps from a long pattern broke nested patterns (MOE_REDUCE/CONV_RES), with garbage at token 0.
  - CONV_TAIL now takes no deps: it is alias-safe because each thread reads and writes only its own channel.
  - REL_BIAS keeps deps on base intermediates only.
- The 15-op MoE variant (expert sum folded into CONV_TAIL) was dropped: it is longer than the 16-node pack cap, and it is not alias-safe.
- Remaining levels are mostly the mix/loop-control prologue: about 37 levels per visit, against Locals' ~12.

## Quant check (2026-11-22)
- We run UD-Q3_K_XL. It cuts 8 of the 40 released NVFP4 expert tensors to 2–3.4 bits, and the MXFP8 layers 20–23 to 3–4.5 bits.
- UD-Q4_K_XL keeps all NVFP4 as released, with Q8_0 for layers 20–23.
- Bytes per token: Q3 3.45, IQ4_XS ~3.68, Q4_K_XL ~3.92, Q8_K_XL ~5.64 GB. Q8_K_XL (22.9 GB) does not fit a 24 GB Mac.
- Our KLD/dcheck numbers compare engine vs stock on the same file. They do not measure the quant's own loss.
