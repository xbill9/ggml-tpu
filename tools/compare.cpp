// Run one model on the CPU backend and on the TPU backend with identical tokens and compare logits.
// usage: tpu-compare -m model.gguf [-p prompt] [-n steps] [-c ctx]
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct Run {
    llama_model *   model = nullptr;
    llama_context * ctx   = nullptr;
};

static Run load(const char * path, int ngl, int n_ctx) {
    Run r;
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    r.model = llama_model_load_from_file(path, mp);
    if (!r.model) { fprintf(stderr, "load failed\n"); exit(1); }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = n_ctx;
    cp.n_ubatch = n_ctx;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.no_perf = true;
    r.ctx = llama_init_from_model(r.model, cp);
    if (!r.ctx) { fprintf(stderr, "context failed\n"); exit(1); }
    return r;
}

static std::vector<float> decode(Run & r, std::vector<llama_token> toks, int n_vocab) {
    llama_batch b = llama_batch_get_one(toks.data(), (int32_t) toks.size());
    if (llama_decode(r.ctx, b) != 0) { fprintf(stderr, "decode failed\n"); exit(1); }
    const float * l = llama_get_logits_ith(r.ctx, -1);
    return std::vector<float>(l, l + n_vocab);
}

int main(int argc, char ** argv) {
    const char * path = nullptr;
    std::string prompt = "The capital of France is";
    int steps = 8, n_ctx = 256;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m")) path = argv[++i];
        else if (!strcmp(argv[i], "-p")) prompt = argv[++i];
        else if (!strcmp(argv[i], "-n")) steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-c")) n_ctx = atoi(argv[++i]);
    }
    if (!path) { fprintf(stderr, "usage: %s -m model.gguf [-p prompt] [-n steps]\n", argv[0]); return 1; }

    ggml_backend_load_all();
    llama_backend_init();

    Run cpu = load(path, 0, n_ctx);
    Run tpu = load(path, 999, n_ctx);

    const llama_vocab * vocab = llama_model_get_vocab(cpu.model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    std::vector<llama_token> toks(prompt.size() + 8);
    int n = llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(), toks.data(), (int32_t) toks.size(), true, false);
    toks.resize(n);

    int agree = 0;
    double worst = 0;
    std::vector<llama_token> in = toks;
    for (int s = 0; s <= steps; s++) {
        std::vector<float> a = decode(cpu, in, n_vocab);
        std::vector<float> c = decode(tpu, in, n_vocab);
        double maxd = 0, maxa = 0, se = 0, sa = 0;
        for (int i = 0; i < n_vocab; i++) {
            const double d = (double) a[i] - c[i];
            maxd = std::max(maxd, std::fabs(d));
            maxa = std::max(maxa, (double) std::fabs(a[i]));
            se += d * d;
            sa += (double) a[i] * a[i];
        }
        const int ta = (int) (std::max_element(a.begin(), a.end()) - a.begin());
        const int tc = (int) (std::max_element(c.begin(), c.end()) - c.begin());
        const double nmse = sa > 0 ? se / sa : se;
        worst = std::max(worst, nmse);
        agree += ta == tc;
        printf("step %2d  tokens_in=%3zu  max|diff|=%.3e  max|logit|=%.3e  nmse=%.3e  argmax cpu=%d tpu=%d %s\n", s,
               in.size(), maxd, maxa, nmse, ta, tc, ta == tc ? "same" : "DIFFERENT");
        in = {ta};  // teacher-force the CPU's greedy token into both
    }
    printf("argmax agreement %d/%d, worst nmse %.3e\n", agree, steps + 1, worst);

    llama_free(cpu.ctx);
    llama_free(tpu.ctx);
    llama_model_free(cpu.model);
    llama_model_free(tpu.model);
    return 0;
}
