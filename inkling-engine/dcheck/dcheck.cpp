// greedy decode check: prints text, saves per-step logits, compares against a reference file
#include "llama.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>

int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s model prompt n_gen out.bin [ref.bin]\n", argv[0]); return 1; }
    const int n_gen = atoi(argv[3]);
    if (getenv("DCHECK_LOG")) {
        llama_log_set([](ggml_log_level, const char * t, void *) { fputs(t, stderr); }, nullptr);
    } else {
        llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
    }
    llama_backend_init();
    auto mp = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    auto cp = llama_context_default_params();
    cp.n_ctx = 4096; cp.n_batch = 512; cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    llama_context * ctx = llama_init_from_model(model, cp);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    std::string prompt = argv[2];
    std::vector<llama_token> toks(prompt.size() + 16);
    int n = llama_tokenize(vocab, prompt.c_str(), prompt.size(), toks.data(), toks.size(), true, true);
    toks.resize(n);

    FILE * fo = fopen(argv[4], "wb");
    FILE * fr = argc > 5 ? fopen(argv[5], "rb") : nullptr;
    std::vector<float> ref(n_vocab);
    double max_diff = 0; int first_div = -1; double kl_sum = 0; int kl_n = 0;

    llama_batch b = llama_batch_get_one(toks.data(), toks.size());
    std::string out;
    double t_dec = 0;
    for (int i = 0; i < n_gen; i++) {
        auto t0 = std::chrono::steady_clock::now();
        if (llama_decode(ctx, b)) { fprintf(stderr, "decode failed\n"); return 1; }
        const float * lg = llama_get_logits_ith(ctx, -1);
        if (i > 0) t_dec += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        fwrite(lg, sizeof(float), n_vocab, fo);
        if (fr && fread(ref.data(), sizeof(float), n_vocab, fr) == (size_t) n_vocab) {
            double md = 0;
            for (int k = 0; k < n_vocab; k++) md = fmax(md, fabs(lg[k] - ref[k]));
            max_diff = fmax(max_diff, md);
            double mr = -1e30, ml = -1e30;
            for (int k = 0; k < n_vocab; k++) { mr = fmax(mr, ref[k]); ml = fmax(ml, lg[k]); }
            double zr = 0, zl = 0;
            for (int k = 0; k < n_vocab; k++) { zr += exp(ref[k] - mr); zl += exp(lg[k] - ml); }
            double kl = 0;
            for (int k = 0; k < n_vocab; k++) {
                const double lp = ref[k] - mr - log(zr), lq = lg[k] - ml - log(zl);
                kl += exp(lp)*(lp - lq);
            }
            kl_sum += kl; kl_n++;
            if (getenv("DCHECK_STEPS") && i < atoi(getenv("DCHECK_STEPS"))) fprintf(stderr, "step %d max_diff %.6g\n", i, md);
        }
        llama_token best = 0;
        for (int k = 1; k < n_vocab; k++) if (lg[k] > lg[best]) best = k;
        if (fr) {
            llama_token rb = 0;
            for (int k = 1; k < n_vocab; k++) if (ref[k] > ref[rb]) rb = k;
            if (rb != best && first_div < 0) first_div = i;
        }
        char buf[256];
        int l = llama_token_to_piece(vocab, best, buf, sizeof(buf), 0, true);
        out.append(buf, l > 0 ? l : 0);
        if (llama_vocab_is_eog(vocab, best)) break;
        static llama_token cur; cur = best;
        b = llama_batch_get_one(&cur, 1);
    }
    fclose(fo);
    printf("%s\n---\n", out.c_str());
    printf("decode %.2f tok/s", (n_gen - 1)/t_dec);
    if (fr) printf("  max_logit_diff %.4g  mean_kl %.3g  first_top1_divergence %d", max_diff, kl_n ? kl_sum/kl_n : 0.0, first_div);
    printf("\n");
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
