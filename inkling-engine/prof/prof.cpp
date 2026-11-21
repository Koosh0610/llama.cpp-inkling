// Per-node timing of one decode step via the scheduler eval callback (forces a sync per node).
#include "llama.h"
#include "ggml.h"
#include <chrono>
#include <cstdio>
#include <map>
#include <regex>
#include <string>
#include <vector>

using clk = std::chrono::steady_clock;
struct Prof {
    bool on = false;
    clk::time_point last;
    std::map<std::string, std::pair<double, int>> by_key;
    double total = 0;
};

static bool cb(ggml_tensor * t, bool ask, void * ud) {
    auto * p = (Prof *) ud;
    if (ask) return true;
    auto now = clk::now();
    if (p->on) {
        double us = std::chrono::duration<double, std::micro>(now - p->last).count();
        std::string name = std::regex_replace(std::string(t->name), std::regex("-\\d+"), "-N");
        std::string key = std::string(ggml_op_desc(t)) + " | " + name;
        auto & e = p->by_key[key]; e.first += us; e.second++;
        p->total += us;
    }
    p->last = clk::now();
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: prof model.gguf [n_batch_tokens]\n"); return 1; }
    int n_tok = argc > 2 ? atoi(argv[2]) : 1;
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    Prof prof;
    auto cp = llama_context_default_params();
    cp.n_ctx = 2048; cp.n_batch = 512; cp.n_ubatch = 512;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    if (!getenv("GGML_METAL_PROFILE")) { cp.cb_eval = cb; cp.cb_eval_user_data = &prof; }
    llama_context * ctx = llama_init_from_model(model, cp);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::vector<llama_token> toks(64);
    std::string prompt = "The quick brown fox jumps over the lazy dog. Once upon a time in a land far away there lived";
    int n = llama_tokenize(vocab, prompt.c_str(), prompt.size(), toks.data(), toks.size(), true, false);
    toks.resize(n);
    llama_decode(ctx, llama_batch_get_one(toks.data(), n));

    // warm decode, then profiled decode of n_tok tokens
    llama_token t0 = toks.back();
    std::vector<llama_token> step(n_tok, t0);
    llama_decode(ctx, llama_batch_get_one(step.data(), n_tok));
    prof.on = true; prof.last = clk::now();
    auto a = clk::now();
    llama_decode(ctx, llama_batch_get_one(step.data(), n_tok));
    double wall = std::chrono::duration<double, std::milli>(clk::now() - a).count();
    prof.on = false;

    std::vector<std::pair<std::string, std::pair<double, int>>> v(prof.by_key.begin(), prof.by_key.end());
    std::sort(v.begin(), v.end(), [](auto & x, auto & y) { return x.second.first > y.second.first; });
    int nodes = 0; for (auto & e : v) nodes += e.second.second;
    printf("wall %.1f ms (synced), nodes %d, avg %.1f us/node\n", wall, nodes, prof.total / nodes);
    for (size_t i = 0; i < v.size() && i < 45; i++)
        printf("%8.2f ms %5.1f%% x%-4d %6.1f us  %s\n", v[i].second.first / 1000, 100 * v[i].second.first / prof.total,
               v[i].second.second, v[i].second.first / v[i].second.second, v[i].first.c_str());
    llama_free(ctx); llama_model_free(model);
    return 0;
}
