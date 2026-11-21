// Router overlap in small verify batches: for each MoE layer evaluation, distinct experts / (tokens * 8).
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

struct St { long pairs = 0, distinct = 0; int n_tok = 4; };

static bool cb(ggml_tensor * t, bool ask, void * ud) {
    const bool want = strncmp(t->name, "ffn_moe_topk", 12) == 0;
    if (ask) return want;
    if (!want) return true;
    auto * s = (St *) ud;
    const int k = t->ne[0], n = t->ne[1];
    std::vector<int32_t> ids(k * n);
    for (int j = 0; j < n; j++) ggml_backend_tensor_get(t, ids.data() + j * k, j * t->nb[1], k * sizeof(int32_t));
    for (int b0 = 0; b0 + s->n_tok <= n; b0 += s->n_tok) {
        std::set<int> u;
        for (int j = b0; j < b0 + s->n_tok; j++) for (int e = 0; e < k; e++) u.insert(ids[j * k + e]);
        s->pairs += (long) s->n_tok * k;
        s->distinct += u.size();
    }
    return true;
}

int main(int argc, char ** argv) {
    llama_backend_init();
    auto mp = llama_model_default_params(); mp.n_gpu_layers = 999;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    std::ifstream f(argv[2]); std::stringstream ss; ss << f.rdbuf(); std::string text = ss.str().substr(0, 20000);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> toks(text.size() + 16);
    int n = llama_tokenize(vocab, text.c_str(), text.size(), toks.data(), toks.size(), true, false);
    toks.resize(std::min(n, 1024));
    for (int nt : {2, 4, 8}) {
        St st; st.n_tok = nt;
        auto cp = llama_context_default_params();
        cp.n_ctx = 2048; cp.n_batch = 1024; cp.n_ubatch = 1024;
        cp.cb_eval = cb; cp.cb_eval_user_data = &st;
        llama_context * ctx = llama_init_from_model(model, cp);
        llama_decode(ctx, llama_batch_get_one(toks.data(), toks.size()));
        printf("batch %d tokens: %ld expert reads -> %ld distinct (%.1f%% saved by grouping)\n", nt, st.pairs, st.distinct, 100.0 * (1.0 - (double) st.distinct / st.pairs));
        llama_free(ctx);
    }
    llama_model_free(model);
}
