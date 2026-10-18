// llama-logits MODEL.gguf TOKENS.bin OUT.bin [options]
//
// Runs llama.cpp on a token sequence (int32, back to back) and writes the logits of every position
// as float32 [n_tokens, n_vocab]. With --gen N it then decodes N tokens greedily, one at a time,
// and writes their ids (int32) to OUT.bin.gen.
//
//   --ngl N        layers on the GPU (default 0; 99 for all)
//   --threads N    CPU threads (default 6)
//   --fa auto|on|off   flash attention (default auto, as llama.cpp)
//   --kv f16|f32   KV cache type (default f16, as llama.cpp)
//   --ubatch N     physical batch size (default 512)
//   --repack 0|1   CPU weight repacking into interleaved layouts (default 1, as llama.cpp)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "llama.h"

static void die(const char * msg) {
    fprintf(stderr, "llama-logits: %s\n", msg);
    exit(1);
}

int main(int argc, char ** argv) {
    if (argc < 4) die("usage: llama-logits MODEL.gguf TOKENS.bin OUT.bin [--ngl N] [--threads N] [--fa auto|on|off] [--kv f16|f32] [--ubatch N] [--gen N]");
    int ngl = 0, threads = 6, ubatch = 512, gen = 0, repack = 1;
    llama_flash_attn_type fa = LLAMA_FLASH_ATTN_TYPE_AUTO;
    ggml_type kv = GGML_TYPE_F16;
    for (int i = 4; i + 1 < argc; i += 2) {
        std::string k = argv[i], v = argv[i + 1];
        if (k == "--ngl") ngl = atoi(v.c_str());
        else if (k == "--threads") threads = atoi(v.c_str());
        else if (k == "--ubatch") ubatch = atoi(v.c_str());
        else if (k == "--gen") gen = atoi(v.c_str());
        else if (k == "--repack") repack = atoi(v.c_str());
        else if (k == "--fa") fa = v == "on" ? LLAMA_FLASH_ATTN_TYPE_ENABLED : v == "off" ? LLAMA_FLASH_ATTN_TYPE_DISABLED : LLAMA_FLASH_ATTN_TYPE_AUTO;
        else if (k == "--kv") kv = v == "f32" ? GGML_TYPE_F32 : GGML_TYPE_F16;
        else die(("unknown option " + k).c_str());
    }

    std::vector<llama_token> tokens;
    {
        FILE * f = fopen(argv[2], "rb");
        if (!f) die("cannot open tokens");
        int32_t t;
        while (fread(&t, sizeof t, 1, f) == 1) tokens.push_back(t);
        fclose(f);
    }
    const int n = (int) tokens.size();
    if (n == 0) die("no tokens");

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    mp.use_extra_bufts = repack != 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) die("cannot load model");
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n + gen + 1;
    cp.n_batch = n;
    cp.n_ubatch = ubatch < n ? ubatch : n;
    cp.n_threads = cp.n_threads_batch = threads;
    cp.flash_attn_type = fa;
    cp.type_k = cp.type_v = kv;
    cp.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) die("cannot create context");
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    llama_batch batch = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; i++) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1;
    }
    batch.n_tokens = n;
    if (llama_decode(ctx, batch) != 0) die("llama_decode failed");

    FILE * out = fopen(argv[3], "wb");
    if (!out) die("cannot open output");
    for (int i = 0; i < n; i++) fwrite(llama_get_logits_ith(ctx, i), sizeof(float), n_vocab, out);
    fclose(out);

    if (gen > 0) {
        std::vector<int32_t> ids;
        const float * last = llama_get_logits_ith(ctx, n - 1);
        for (int s = 0; s < gen; s++) {
            int best = 0;
            for (int j = 1; j < n_vocab; j++) if (last[j] > last[best]) best = j;
            ids.push_back(best);
            batch.n_tokens = 1;
            batch.token[0] = best;
            batch.pos[0] = n + s;
            batch.logits[0] = 1;
            if (llama_decode(ctx, batch) != 0) die("llama_decode failed while generating");
            last = llama_get_logits_ith(ctx, 0);
        }
        FILE * g = fopen((std::string(argv[3]) + ".gen").c_str(), "wb");
        fwrite(ids.data(), sizeof(int32_t), ids.size(), g);
        fclose(g);
    }
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
