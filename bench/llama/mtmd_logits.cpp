// mtmd-logits MODEL.gguf MMPROJ.gguf PIXELS.bin TOKENS.bin OUT.bin [options]
//
// llama.cpp on an image prompt, from the same preprocessed pixels as transformers: PIXELS.bin is
// float32 [tiles, 3, S, S] (the processor's normalized pixel_values), or with --image one image
// [3, H, W]; TOKENS.bin the prompt's ids (int32) with one image token per embedding row. The image
// encoder is llama.cpp's own (clip, from the mmproj GGUF), run one tile per call as mtmd does for
// sliced images; the k-th run of image tokens takes tile k's rows. M-RoPE models (Qwen-VL) get
// mtmd's positions: (t, y, x) = start + (0, row, col) per image row, and text after an image
// continues from start + max(grid). Writes the logits of positions [--from, n) as float32
// [n - from, n_vocab] to OUT.bin; with --gen N, N greedy tokens (int32) to OUT.bin.gen; timings as
// JSON to OUT.bin.json.
//
//   --image-token ID   the image placeholder id (required)
//   --size S           tile side in pixels (default 512)
//   --image WxH        one W x H image instead of tiles
//   --from N           first position whose logits are written (default: after the last image token)
//   --ngl N            layers on the GPU (default 0; 99 for all; > 0 also puts clip on the GPU)
//   --threads N        CPU threads (default 6)
//   --fa auto|on|off   flash attention for the LM (default auto, as llama.cpp)
//   --gen N            greedy tokens after the prompt
//   --reps N           timed repetitions of the image encode and the prompt (default 0)
//   --dump-embd PATH   also write the image rows clip produced (float32 [rows, n_embd_inp])
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "clip.h"
#include "clip-impl.h"
#include "llama.h"

static void die(const char * msg) {
    fprintf(stderr, "mtmd-logits: %s\n", msg);
    exit(1);
}

template <typename T> static std::vector<T> read_all(const char * path) {
    FILE * f = fopen(path, "rb");
    if (!f) die((std::string("cannot open ") + path).c_str());
    std::vector<T> v;
    T x;
    while (fread(&x, sizeof x, 1, f) == 1) v.push_back(x);
    fclose(f);
    return v;
}

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char ** argv) {
    if (argc < 6) die("usage: mtmd-logits MODEL.gguf MMPROJ.gguf PIXELS.bin TOKENS.bin OUT.bin --image-token ID [options]");
    int ngl = 0, threads = 6, gen = 0, reps = 0, from = -1, image_token = -1, S = 512, W = 0, H = 0;
    std::string dump_embd;
    llama_flash_attn_type fa = LLAMA_FLASH_ATTN_TYPE_AUTO;
    for (int i = 6; i + 1 < argc; i += 2) {
        std::string k = argv[i], v = argv[i + 1];
        if (k == "--ngl") ngl = atoi(v.c_str());
        else if (k == "--threads") threads = atoi(v.c_str());
        else if (k == "--gen") gen = atoi(v.c_str());
        else if (k == "--reps") reps = atoi(v.c_str());
        else if (k == "--from") from = atoi(v.c_str());
        else if (k == "--image-token") image_token = atoi(v.c_str());
        else if (k == "--size") S = atoi(v.c_str());
        else if (k == "--dump-embd") dump_embd = v;
        else if (k == "--image") { if (sscanf(v.c_str(), "%dx%d", &W, &H) != 2) die("--image wants WxH"); }
        else if (k == "--fa") fa = v == "on" ? LLAMA_FLASH_ATTN_TYPE_ENABLED : v == "off" ? LLAMA_FLASH_ATTN_TYPE_DISABLED : LLAMA_FLASH_ATTN_TYPE_AUTO;
        else die(("unknown option " + k).c_str());
    }
    if (image_token < 0) die("--image-token is required");
    std::vector<float> pixels = read_all<float>(argv[3]);
    std::vector<int32_t> tokens = read_all<int32_t>(argv[4]);
    const int n = (int) tokens.size();
    int last_img = -1, n_img = 0;
    for (int i = 0; i < n; i++) if (tokens[i] == image_token) { last_img = i; n_img++; }
    if (from < 0) from = last_img + 1;
    if (from >= n) die("--from is past the prompt");

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) die("cannot load model");
    const int n_embd = llama_model_n_embd_inp(model);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    clip_context_params clp{};
    clp.use_gpu = ngl > 0;
    clp.device = nullptr;
    clp.flash_attn_type = CLIP_FLASH_ATTN_TYPE_AUTO;  // mtmd's default (it follows the LM's, auto)
    clp.image_min_tokens = clp.image_max_tokens = -1;
    clp.warmup = true;
    clip_ctx * clip = clip_init(argv[2], clp).ctx_v;
    if (!clip) die("cannot load mmproj");
    if (clip_n_mmproj_embd(clip) != n_embd) die("mmproj and model embedding sizes differ");

    // tiles: [T, 3, h, w] planar -> clip_image_f32 (interleaved RGB, already normalized)
    const int iw = W ? W : S, ih = W ? H : S;
    const size_t per_tile = (size_t) 3 * iw * ih;
    if (pixels.size() % per_tile) die("pixels are not a whole number of tiles");
    const int tiles = (int) (pixels.size() / per_tile);
    std::vector<clip_image_f32> imgs(tiles);
    for (int t = 0; t < tiles; t++) {
        std::vector<float> hwc(per_tile);
        const float * src = pixels.data() + t * per_tile;
        for (int c = 0; c < 3; c++)
            for (size_t p = 0; p < (size_t) iw * ih; p++) hwc[p * 3 + c] = src[(size_t) c * iw * ih + p];
        imgs[t].set_size({iw, ih}, false, false);
        imgs[t].cpy_buf(hwc);
    }
    const llama_rope_type rt = llama_model_rope_type(model);
    const bool mrope = rt == LLAMA_ROPE_TYPE_MROPE || rt == LLAMA_ROPE_TYPE_IMROPE;
    const int gx = clip_n_output_tokens_x(clip, &imgs[0]), gy = clip_n_output_tokens_y(clip, &imgs[0]);
    if (mrope && tiles != 1) die("M-RoPE models take one --image");
    std::vector<float> embd;  // [n_img, n_embd]
    auto encode = [&]() {
        embd.clear();
        for (int t = 0; t < tiles; t++) {
            clip_image_f32_batch b;
            b.entries.push_back(imgs[t]);
            std::vector<float> out((size_t) clip_n_output_tokens(clip, &imgs[t]) * n_embd);
            if (!clip_image_batch_encode(clip, threads, &b, out)) die("clip encode failed");
            embd.insert(embd.end(), out.begin(), out.end());
        }
    };
    encode();
    if (!dump_embd.empty()) {
        FILE * f = fopen(dump_embd.c_str(), "wb");
        if (!f) die("cannot open --dump-embd");
        fwrite(embd.data(), sizeof(float), embd.size(), f);
        fclose(f);
    }
    if ((int) (embd.size() / n_embd) != n_img) {
        fprintf(stderr, "mtmd-logits: %d tiles gave %zu rows for %d image tokens\n", tiles, embd.size() / n_embd, n_img);
        return 1;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n + gen + 1;
    cp.n_batch = cp.n_ubatch = 512;
    cp.n_threads = cp.n_threads_batch = threads;
    cp.flash_attn_type = fa;
    cp.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) die("cannot create context");
    llama_batch tb = llama_batch_init(512, 0, 1), eb = llama_batch_init(512, n_embd, 1);
    std::vector<llama_pos> epos(4 * 512);  // an embedding batch's M-RoPE positions, section-major
    llama_pos * eb_pos = eb.pos;
    llama_pos lpos = 0;  // llama's position of the next token (M-RoPE: not its index after an image)

    // the prompt in spans of text or image rows, at most 512 per llama_decode
    std::vector<float> logits((size_t) (n - from) * n_vocab);
    auto prompt = [&]() {
        llama_memory_clear(llama_get_memory(ctx), true);
        int i = 0, row = 0;
        llama_pos img0 = 0;
        lpos = 0;
        while (i < n) {
            const bool img = tokens[i] == image_token;
            int j = i;
            while (j < n && j - i < 512 && (tokens[j] == image_token) == img) j++;
            llama_batch & b = img ? eb : tb;
            b.pos = img && mrope ? epos.data() : img ? eb_pos : b.pos;
            const int nb = j - i;
            for (int k = i; k < j; k++) {
                const int r = k - i;
                if (img) {
                    if (row == 0) img0 = lpos;
                    if (mrope) {  // mtmd_image_tokens_get_decoder_pos and set_position_mrope_2d
                        epos[r] = img0;
                        epos[r + nb] = img0 + row / gx;
                        epos[r + 2 * nb] = img0 + row % gx;
                        epos[r + 3 * nb] = 0;
                    } else {
                        b.pos[r] = lpos++;
                    }
                    memcpy(b.embd + (size_t) r * n_embd, embd.data() + (size_t) (row++) * n_embd, n_embd * sizeof(float));
                    if (mrope && row == n_img) lpos = img0 + std::max(gx, gy);
                } else {
                    b.token[r] = tokens[k];
                    b.pos[r] = lpos++;
                }
                b.n_seq_id[r] = 1;
                b.seq_id[r][0] = 0;
                b.logits[r] = k >= from;
            }
            b.n_tokens = nb;
            if (llama_decode(ctx, b) != 0) die("llama_decode failed");
            for (int k = std::max(i, from); k < j; k++)
                memcpy(logits.data() + (size_t) (k - from) * n_vocab, llama_get_logits_ith(ctx, k - i), n_vocab * sizeof(float));
            i = j;
        }
    };
    prompt();
    FILE * out = fopen(argv[5], "wb");
    if (!out) die("cannot open output");
    fwrite(logits.data(), sizeof(float), logits.size(), out);
    fclose(out);

    double enc_ms = 0, prompt_ms = 0, tg_ms = 0;
    for (int r = 0; r < reps; r++) {
        double t0 = now_ms();
        encode();
        double t1 = now_ms();
        prompt();
        double t2 = now_ms();
        enc_ms += (t1 - t0) / reps;
        prompt_ms += (t2 - t1) / reps;
    }

    if (gen > 0) {
        std::vector<int32_t> ids;
        std::vector<float> last(logits.end() - n_vocab, logits.end());
        const float * lp = last.data();
        double t0 = now_ms();
        for (int s = 0; s < gen; s++) {
            int best = 0;
            for (int j = 1; j < n_vocab; j++) if (lp[j] > lp[best]) best = j;
            ids.push_back(best);
            tb.n_tokens = 1;
            tb.token[0] = best;
            tb.pos[0] = lpos + s;
            tb.n_seq_id[0] = 1;
            tb.seq_id[0][0] = 0;
            tb.logits[0] = 1;
            if (llama_decode(ctx, tb) != 0) die("llama_decode failed while generating");
            lp = llama_get_logits_ith(ctx, 0);
        }
        tg_ms = now_ms() - t0;
        FILE * g = fopen((std::string(argv[5]) + ".gen").c_str(), "wb");
        fwrite(ids.data(), sizeof(int32_t), ids.size(), g);
        fclose(g);
    }
    FILE * j = fopen((std::string(argv[5]) + ".json").c_str(), "w");
    fprintf(j, "{\"tiles\": %d, \"prompt_tokens\": %d, \"encode_ms\": %.3f, \"prompt_ms\": %.3f, \"gen\": %d, \"gen_ms\": %.3f}\n",
            tiles, n, enc_ms, prompt_ms, gen, tg_ms);
    fclose(j);
    eb.pos = eb_pos;
    llama_batch_free(tb);
    llama_batch_free(eb);
    clip_free(clip);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
