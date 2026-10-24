# Spice.ai packaging

This fork carries upstream llama.cpp releases, plus this directory and the root
`Cargo.toml`. Together they make a release fetchable by Cargo as the `llama-cpp-src` git
dependency, which Spice.ai's `llama-cpp` crate builds with CMake.

It also carries these source patches, which each Spice branch must keep until upstream has them:

- **A failed check while loading a model throws instead of aborting.** libllama defines
  `GGML_ABORT_THROWS` (`src/CMakeLists.txt`), so its `GGML_ASSERT`s go to `ggml_abort_throw`
  (`src/llama-impl.cpp`, declared in `ggml/include/ggml.h`). That throws a
  `std::runtime_error` only inside a `llama_abort_throws_scope` (`src/llama-impl.h`), which
  `llama_model_load` (`src/llama.cpp`) opens in the try block that turns an exception into a
  failed load, so a GGUF whose metadata fails a check fails its load rather than aborting the
  process. Everywhere else, and on every other thread, a failed check aborts as upstream: the
  C API functions that reach those checks do not catch exceptions. Spice's `llama-cpp` crate
  tests the load path with crafted and corrupted GGUFs.
- **DeepSeek-V4.1-Flash (`deepseek41`)**, upstream PR ggml-org/llama.cpp#30817 at `b72050d03`, not yet
  merged upstream. It adds the model, the DSpark drafter, and the pairwise all-reduce between two RPC
  servers that `-sm tensor` uses when the model is split across two machines. Drop it once a release
  contains the PR.
- **Fixes on top of #30817**, not upstream. A GGUF whose `attention.output_group_count` is 0 fails its load
  instead of dividing by zero (`src/models/deepseek4.cpp`, which `deepseek41` shares), and one whose Engram head
  offset plus bucket count passes `INT32_MAX` fails its load instead of wrapping and aborting in the first decode
  (`src/models/deepseek41.cpp`), whose output projection also applies a LoRA adapter on `output.weight`.
  `ggml_validate_row_data` rejects MXFP8 blocks with an E4M3 NaN code (`ggml/src/ggml-quants.c`), so
  `--check-tensors` catches them. `deepseek41v` image preprocessing (`tools/mtmd/mtmd-image.cpp`) counts the tokens
  of a placeholder bitmap, and pads an image too thin to keep a pixel at the planned size with the pad color alone,
  as Pillow's `ImageOps.pad` does, instead of failing. With `convert_hf_to_gguf.py --remote`,
  `conversion/deepseek41.py` reads the Engram tables by HTTP range instead of from shards it never downloaded.
  When dropping #30817, keep whichever of these the release still lacks.
- **DFlash drafters over Qwen3.8-Flash-Next (`qwen4exp`)**, upstream #30927 (`d9ee007a0`), merged after
  b11803. The target records the mean over its hyper-connection copies as each layer input, the
  `[n_embd, n_tokens]` rows a DFlash drafter reads; without it the first drafted request aborts in
  `llama_context::extract_layer_inputs`. Drop it at the first release that contains it.
- **DFlash drafters over GLM-5.3-Flash (`glm5-next`)**, `src/models/glm5-next.cpp`, not upstream. The same
  layer-input recording for `glm5-next`, which has no hook for it, so a DFlash draft over it aborts in
  `llm_graph_result::set_outputs` (layer input tensor is null).
- **`GGML_RPC_COMM_HOST`**, `ggml/src/ggml-rpc/ggml-rpc.cpp`, not upstream, on top of #30817. Rank 1 of
  the pairwise all-reduce reaches rank 0 at this address instead of the one rank 0 serves on, so two
  servers that also share a direct link exchange over that link, and over RDMA when the address is a
  RoCE GID.

- **Two-rank DeepSeek-V4 decode switches**, not upstream, each off by default and measured on two DGX Sparks:
  `GGML_CUDA_MMVQ_IQ3_SMALL_K=1` lets IQ3 expert down projections with few K blocks take the small-K
  `mul_mat_vec_q` path (`ggml/src/ggml-cuda/mmvq.cu`); `LLAMA_DSV4_HC_WEIGHTS=1` computes the
  hyper-connection weights in one op, `ggml_dsv4_hc_weights` (`src/models/deepseek4.cpp`,
  `ggml/src/ggml-cuda/dsv4-hc.cu`); on the client, `GGML_RPC_COMM_NCCL=1` runs the pairwise all-reduce
  through NCCL when both rpc-servers can, and `GGML_RPC_COMM_WIRE_F32=1` keeps the host all-reduce's partials in
  F32, both sent to the two servers in `COMM_INIT` (`ggml/src/ggml-rpc/ggml-rpc.cpp`); `GGML_RPC_SET_ASYNC=1` on
  the client queues buffer writes without waiting; `LLAMA_DSV4_SHEXP_SHIFT=<rows>` moves shared-expert rows to the rank with the lighter expert split
  (`src/llama-model.cpp`). Drop each once upstream has an equivalent.

Each Spice branch is `spiceai-<release>`, cut from the upstream release tag with the
Spice commits on top. Its tip is tagged `<release>-spiceai.<n>`, a name apart from the
branch's so neither is ambiguous, with `<n>` counting re-packagings of one release. To move
to a new release:

```bash
git fetch https://github.com/ggml-org/llama.cpp refs/tags/<release>:refs/tags/<release>
git checkout -b spiceai-<release> <release>
# the previous branch's Spice commits, oldest first, without the merge commits PRs leave
git rev-list --reverse --topo-order --no-merges <previous release>..spiceai-<previous release> | git cherry-pick --stdin
# a commit picked from upstream ends in "(cherry picked from commit <sha>)": skip it when the new release
# already contains <sha> (git merge-base --is-ancestor <sha> <release>), as `git cherry-pick --skip`
# set the release in spiceai/release.rs, and the version and description in Cargo.toml
git tag <release>-spiceai.1
git push origin spiceai-<release> refs/tags/<release>-spiceai.1
```

Then pin the tagged commit in spiceai/spiceai by its full SHA.
