// A computation the device fails must fail what reads its results, never hand them back. GGML_BACKEND_FAULT_TEST=N skips
// the N-th graph queued in the process and fails its backend from then on, as a GPU fault does: the device accepted the
// work, so the decode that queued it returns 0, and its outputs are whatever the buffers held before. This decodes a
// prompt and then one token at a time and checks, at every step, that the read of the outputs either succeeds with the
// logits a run without the fault computed (--reference, written by --write-reference), bit for bit, or reports the
// failure: the getters return NULL, llama_get_compute_status returns -3, the sequence's state does not save, and every
// later decode returns -3, and common's sampler picks no token instead of asserting on the missing logits. A run whose
// fault falls within its graphs must see the failure.

#include "llama.h"
#include "sampling.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int fail(const char * what) {
    fprintf(stderr, "FAIL: %s\n", what);
    return 1;
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string reference_path;
    bool        write_reference = false;
    for (int i = 1; i + 1 < argc; i++) {
        if (strcmp(argv[i], "-m") == 0) {
            model_path = argv[i + 1];
        } else if (strcmp(argv[i], "--reference") == 0) {
            reference_path = argv[i + 1];
        } else if (strcmp(argv[i], "--write-reference") == 0) {
            reference_path  = argv[i + 1];
            write_reference = true;
        }
    }
    if (model_path.empty() || reference_path.empty()) {
        fprintf(stderr, "usage: %s -m <model.gguf> (--write-reference | --reference) <file>, the latter with GGML_BACKEND_FAULT_TEST=<n>\n", argv[0]);
        return 1;
    }
    const char * fault_at = getenv("GGML_BACKEND_FAULT_TEST");
    const bool   faulted  = fault_at != nullptr && atoi(fault_at) > 0;
    if (faulted == write_reference) {
        return fail(write_reference ? "the reference is written without GGML_BACKEND_FAULT_TEST" : "GGML_BACKEND_FAULT_TEST is not set: nothing would fail");
    }
    std::vector<float> reference;
    if (!write_reference) {
        FILE * f = fopen(reference_path.c_str(), "rb");
        if (f == nullptr) {
            return fail("the reference does not open");
        }
        float v;
        while (fread(&v, sizeof(v), 1, f) == 1) {
            reference.push_back(v);
        }
        fclose(f);
    }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (model == nullptr) {
        return fail("the model does not load");
    }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx     = 256;
    cp.n_batch   = 64;
    cp.n_ubatch  = 64;
    cp.n_seq_max = 1;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (ctx == nullptr) {
        return fail("the context does not initialize");
    }
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    common_params_sampling sparams;
    common_sampler * smpl = common_sampler_init(model, sparams);
    if (smpl == nullptr) {
        return fail("common's sampler does not initialize");
    }
    // what reads a failed context's outputs through common: it must pick nothing, not abort
    auto common_picks_nothing = [&]() {
        return common_sampler_sample(smpl, ctx, -1) == LLAMA_TOKEN_NULL;
    };

    const int n_prompt = 16;
    const int n_steps  = 16;
    llama_batch batch = llama_batch_init(n_prompt, 0, 1);
    std::vector<float> all; // every step's logits, for --write-reference
    bool failed = false;
    int  rc_fail = 0;
    for (int step = 0; step <= n_steps; step++) {
        const int n = step == 0 ? n_prompt : 1;
        batch.n_tokens = n;
        for (int i = 0; i < n; i++) {
            const int pos = step == 0 ? i : n_prompt + step - 1;
            batch.token[i]     = (llama_token) (1 + (pos*7) % (n_vocab - 1));
            batch.pos[i]       = pos;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = i == n - 1;
        }
        const int32_t rc = llama_decode(ctx, batch);
        if (failed) {
            // a failed context refuses the work instead of computing on what the failure left
            if (rc != -3) {
                fprintf(stderr, "step %d: decode returned %d after the failure, want -3\n", step, rc);
                return fail("a decode after the failure did not fail");
            }
            rc_fail++;
            continue;
        }
        if (rc == -3) {
            // the skipped graph was one of this decode's, and a later one on the failed backend refused to run
            llama_synchronize(ctx);
            if (llama_get_compute_status(ctx) != -3) {
                return fail("a decode failed, but the context reports no failure");
            }
            if (!common_picks_nothing()) {
                return fail("common's sampler picked a token from a failed decode");
            }
            printf("step %d: the decode failed (-3)\n", step);
            failed = true;
            continue;
        }
        if (rc != 0) {
            fprintf(stderr, "step %d: decode returned %d\n", step, rc);
            return fail("a decode failed for another reason");
        }
        const float * logits = llama_get_logits_ith(ctx, -1);
        const int32_t status = llama_get_compute_status(ctx);
        if (logits == nullptr || status != 0) {
            // the failure: reported, and no output or state to read
            if (logits != nullptr || status != -3) {
                fprintf(stderr, "step %d: logits %s, status %d\n", step, logits ? "returned" : "NULL", status);
                return fail("a read after the failure returned logits, or reported no failure");
            }
            std::vector<uint8_t> state(llama_state_seq_get_size(ctx, 0) + 1);
            if (llama_state_seq_get_data(ctx, state.data(), state.size(), 0) != 0) {
                return fail("the sequence's state saved after the failure");
            }
            if (!common_picks_nothing()) {
                return fail("common's sampler picked a token from a failed decode");
            }
            printf("step %d: the read of the outputs reports the failure (-3)\n", step);
            failed = true;
            continue;
        }
        if (write_reference) {
            all.insert(all.end(), logits, logits + n_vocab);
            continue;
        }
        if (reference.size() < (size_t) (step + 1)*n_vocab) {
            return fail("the reference is shorter than this run");
        }
        const float * want = reference.data() + (size_t) step*n_vocab;
        if (memcmp(want, logits, n_vocab*sizeof(float)) != 0) {
            int32_t k = 0;
            while (k < n_vocab && memcmp(want + k, logits + k, sizeof(float)) == 0) {
                k++;
            }
            fprintf(stderr, "step %d: logit %d is %g, the computation's is %g\n", step, k, logits[k], want[k]);
            return fail("a read returned logits that are not the computation's, and reported no failure");
        }
    }
    llama_batch_free(batch);
    common_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    if (write_reference) {
        FILE * f = fopen(reference_path.c_str(), "wb");
        if (f == nullptr || fwrite(all.data(), sizeof(float), all.size(), f) != all.size()) {
            return fail("the reference does not write");
        }
        fclose(f);
        printf("OK (reference of %d steps written)\n", n_steps + 1);
        return 0;
    }
    if (!failed) {
        return fail("the skipped graph was among this run's, yet no decode or read reported the failure");
    }
    printf("OK (%d later decodes refused)\n", rc_fail);
    return 0;
}
