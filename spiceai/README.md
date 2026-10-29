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
- **Fixes on top of upstream's DeepSeek-V4.1-Flash (`deepseek41`, ggml-org/llama.cpp#31086)**, not upstream. A GGUF
  whose `attention.output_group_count` is 0 fails its load instead of dividing by zero (`src/models/deepseek4.cpp`,
  which `deepseek41` shares), and `deepseek41`'s output projection applies a LoRA adapter on `output.weight`
  (`src/models/deepseek41.cpp`). `ggml_validate_row_data` rejects MXFP8 blocks with an E4M3 NaN code
  (`ggml/src/ggml-quants.c`), so `--check-tensors` catches them. `deepseek41v` image preprocessing
  (`tools/mtmd/mtmd-image.cpp`) counts the tokens of a placeholder bitmap, and pads an image too thin to keep a pixel at
  the planned size with the pad color alone, as Pillow's `ImageOps.pad` does, instead of failing. With
  `convert_hf_to_gguf.py --remote`, `conversion/deepseek41.py` reads the Engram tables by HTTP range instead of from
  shards it never downloaded. Since b11984 the model itself is upstream's; the fork's earlier copy (from #30817)
  is gone, along with its `LLAMA_DSV4_HC_WEIGHTS`, `LLAMA_DSV4_VERIFY_REUSE` and `LLAMA_DSV41_KV_MIRROR` switches,
  which upstream's implementation has no place for.
- **DFlash drafters over GLM-5.3-Flash (`glm5-next`)**, `src/models/glm5-next.cpp`, not upstream. The same
  layer-input recording for `glm5-next`, which has no hook for it, so a DFlash draft over it aborts in
  `llm_graph_result::set_outputs` (layer input tensor is null).
- **`GGML_RPC_COMM_HOST`**, `ggml/src/ggml-rpc/ggml-rpc.cpp`, not upstream, on top of upstream's RPC all-reduce. Rank 1 of
  the pairwise all-reduce reaches rank 0 at this address instead of the one rank 0 serves on, so two
  servers that also share a direct link exchange over that link, and over RDMA when the address is a
  RoCE GID.

- **Two-rank DeepSeek-V4 decode switches**, not upstream, each off by default and measured on two DGX Sparks:
  `GGML_CUDA_MMVQ_IQ3_SMALL_K=1` lets IQ3 expert down projections with few K blocks take the small-K
  `mul_mat_vec_q` path (`ggml/src/ggml-cuda/mmvq.cu`); the `ggml_dsv4_hc_weights` op
  (`ggml/src/ggml-cuda/dsv4-hc.cu`) remains, though since b11984 no model builds it; on the client, `GGML_RPC_COMM_NCCL=1` runs the pairwise all-reduce
  through NCCL when both rpc-servers can, and `GGML_RPC_COMM_WIRE_F32=1` keeps the host all-reduce's partials in
  F32, both sent to the two servers in `COMM_INIT` (`ggml/src/ggml-rpc/ggml-rpc.cpp`); `GGML_RPC_SET_ASYNC=1` on
  the client queues buffer writes without waiting; `LLAMA_DSV4_SHEXP_SHIFT=<rows>` moves shared-expert rows to the
  rank with the lighter expert split, only when `tensor_split` is unset or equal, and leaves an uneven split as given
  (`src/llama-model.cpp`). With `GGML_RPC_COMM_NCCL` or `GGML_RPC_COMM_WIRE_F32` set, both rpc-servers must come
  from this branch, since an older server refuses the longer `COMM_INIT`; with neither set, `COMM_INIT` is
  upstream's. Drop each once upstream has an equivalent.

Each Spice branch is `spiceai-<release>`: the upstream release tag with the Spice commits on top, made by
merging the tag into the previous Spice branch (`spiceai-b11999` is `spiceai-b11803` with `b11984`, then `b11999`, merged in). Its tip is tagged `<release>-spiceai.<n>`, a name apart from the
branch's so neither is ambiguous, with `<n>` counting re-packagings of one release. To move
to a new release:

```bash
git fetch https://github.com/ggml-org/llama.cpp refs/tags/<release>:refs/tags/<release>
git checkout -b spiceai-<release> spiceai-<previous release>
git merge <release>
# resolve the conflicts; where the release now has a patch this branch carries (an upstream PR it picked, or an
# equivalent), take the release's and keep only what it still lacks, then update the list above
# set the release in spiceai/release.rs, and the version and description in Cargo.toml
git tag <release>-spiceai.1
git push origin spiceai-<release> refs/tags/<release>-spiceai.1
```

Then pin the tagged commit in spiceai/spiceai by its full SHA.
