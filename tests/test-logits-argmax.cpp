// common_logits_argmax must pick what the greedy sampler picks from the same row
#include "llama.h"
#include "sampling.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

static llama_token greedy_pick(llama_sampler * greedy, const std::vector<float> & row) {
    std::vector<llama_token_data> cur(row.size());
    for (size_t t = 0; t < row.size(); ++t) {
        cur[t] = llama_token_data{ (llama_token) t, row[t], 0.0f };
    }
    llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };
    llama_sampler_apply(greedy, &cur_p);
    return cur[cur_p.selected].id;
}

static void check(llama_sampler * greedy, const std::vector<float> & row, int & n_rows) {
    const llama_token expected = greedy_pick(greedy, row);
    const llama_token got      = common_logits_argmax(row.data(), (int32_t) row.size());
    if (got != expected) {
        fprintf(stderr, "row of %zu: common_logits_argmax %d, greedy %d\n", row.size(), got, expected);
        for (const float x : row) {
            fprintf(stderr, " %g", x);
        }
        fprintf(stderr, "\n");
        assert(false);
    }
    n_rows++;
}

int main(void) {
    llama_sampler * greedy = llama_sampler_init_greedy();
    std::mt19937 rng(42);
    int n_rows = 0;

    // few values, so most rows tie for their largest logit; lengths around the 16-lane and 64-logit block edges
    const float vals[] = { -INFINITY, -1.0f, -0.0f, 0.0f, 0.5f, 1.0f, INFINITY, NAN };
    const int lens[] = { 1, 2, 3, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 200, 1000 };
    int n_late = 0;
    for (const int n : lens) {
        for (int rep = 0; rep < 400; ++rep) {
            std::vector<float> row(n);
            for (auto & x : row) {
                x = vals[rng() % 8u];
            }
            if (rep % 2 == 1) {
                // the largest logit once or twice, anywhere: the first can fall in any lane or block
                for (auto & x : row) {
                    x = x == INFINITY ? 1.0f : x;
                }
                const int at = (int) (rng() % (uint32_t) n);
                row[at] = INFINITY;
                if (rng() % 2u == 0) {
                    row[at + (int) (rng() % (uint32_t) (n - at))] = INFINITY;
                }
            }
            check(greedy, row, n_rows);
            n_late += common_logits_argmax(row.data(), n) >= 64 ? 1 : 0;
        }
    }
    assert(n_late > 400); // 530 with this seed: the first max lands past the first block in many rows

    // explicit cases
    check(greedy, { 1.0f, 3.0f, 2.0f, 3.0f }, n_rows);  // first max on a tie
    check(greedy, { NAN, 5.0f }, n_rows);               // a NaN first logit is never replaced
    check(greedy, { -0.0f, 0.0f }, n_rows);             // equal zeros: the first
    check(greedy, { -INFINITY, -INFINITY }, n_rows);

    // full rows of normal logits, the size of a large vocabulary
    std::normal_distribution<float> nd(0.0f, 3.0f);
    for (int rep = 0; rep < 8; ++rep) {
        std::vector<float> row(129280);
        for (auto & x : row) {
            x = nd(rng);
        }
        check(greedy, row, n_rows);
    }

    llama_sampler_free(greedy);
    printf("common_logits_argmax matches the greedy sampler on %d rows (%d picks past logit 63)\n", n_rows, n_late);
    return 0;
}
