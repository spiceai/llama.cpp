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
- **A GGUF opened by its path whose data section runs past its tensors fails its load**, `ggml/src/gguf.cpp`, not
  upstream. GGML type 43 was this fork's gather-only MXFP8 (33 bytes per 32 values) before v1.0.0 and is F8_E4M3 (one
  byte per value) since. Upstream's offset checks refuse such a tensor only when another tensor follows it; read last,
  it would load with the wrong values. A GGUF opened from a `FILE` may be followed by other data, so that path is not
  checked.
- **The RPC protocol's major is upstream's plus 100 or more** (`ggml/include/ggml-rpc.h`: 110 at v1.0.0, and 111 since
  b12600, where upstream's `GGML_OP_SLEEP` moved this branch's two ops up by one), since this branch's
  ops (`GGML_OP_ARGMAX_PART` and `GGML_OP_ARGMAX_MERGE`, appended after upstream's) and commands differ from
  upstream's: an upstream peer refuses a peer from this branch at the handshake. The HELLO reply's flags byte carries
  upstream's `rpc_dispatcher_flags` in its low bits and this branch's `rpc_op_cap` bits in its high bits. Bump it with
  upstream's, and whenever this branch changes an op or a command.
- **Inkling 2 Mini's looped layers keep their NVFP4 expert scales** (`src/llama-model.cpp`), upstream PR 32704, picked
  before upstream merged it. `inkling2_loop` copies each physical core layer into the cache slots of its later passes
  before the generic pass attaches the optional `.scale` tensors, so without it the second and third passes run the
  NVFP4 experts unscaled (Unsloth reports perplexity about 130 instead of 7.65 for its UD-Q8_K_XL). Take upstream's
  once a release has it.
- **Fixes on top of upstream's DeepSeek-V4.1-Flash (`deepseek41`, ggml-org/llama.cpp#31086)**, not upstream. A GGUF
  whose `attention.output_group_count` is 0 fails its load instead of dividing by zero (`src/models/deepseek4.cpp`,
  which `deepseek41` shares), and `deepseek41`'s output projection applies a LoRA adapter on `output.weight`
  (`src/models/deepseek41.cpp`). `deepseek41v` image preprocessing
  (`tools/mtmd/mtmd-image.cpp`) counts the tokens of a placeholder bitmap, and pads an image too thin to keep a pixel at
  the planned size with the pad color alone, as Pillow's `ImageOps.pad` does, instead of failing. With
  `convert_hf_to_gguf.py --remote`, `conversion/deepseek41.py` reads the Engram tables by HTTP range instead of from
  shards it never downloaded. Since b11984 the model itself is upstream's; the fork's earlier copy (from #30817)
  is gone, along with its `LLAMA_DSV4_HC_WEIGHTS`, `LLAMA_DSV4_VERIFY_REUSE` and `LLAMA_DSV41_KV_MIRROR` switches,
  which upstream's implementation has no place for.
- **A grammar repetition whose maximum is below its minimum fails to parse** (`src/llama-grammar.cpp`,
  `llama_grammar_parser::parse_sequence`), upstream's 46eb1742d (ggml-org/llama.cpp#32551, in b12604), picked before
  this branch's upstream release has it. Without it `S{3,1}` passes `max - min`, which underflows, to
  `handle_repetitions`, which then loops without end (CVE-2026-137057). `tests/test-grammar-parser.cpp` checks that
  such grammars fail. Take upstream's once the branch moves to b12604 or later.
- **An RPC `COPY_TENSOR` must fit in its destination tensor** (`ggml/src/ggml-rpc/ggml-rpc.cpp`,
  `rpc_server::copy_tensor`), the copy check of upstream's 03813c56e (ggml-org/llama.cpp#32651, in b12658), ported
  rather than picked: that commit rewrites the whole RPC server (+513 -747), which this branch's RPC patches conflict
  with. The server checks a tensor's data against its buffer only when the tensor has elements, so a destination with
  none could point anywhere, and the copy was checked only against the end of the destination's buffer: a client that
  learned an address with `BUFFER_GET_BASE` could write below it (CVE-2026-137567). `tests/test-rpc-copy-tensor.cpp`
  sends such a copy over the wire. The same commit's region check is taken for `SET_TENSOR`, `SET_TENSOR_MULTI`,
  `SET_TENSOR_HASH` and `GET_TENSOR` (`rpc_check_tensor_region`): they checked `data + offset`, a sum that wraps, and a
  region inside the buffer but past the tensor reached `ggml_backend_tensor_set`/`get`'s assert, which ended the
  server. The test covers each, and that `MEMSET_TENSOR`, `SET_TENSOR_2D` and `GET_TENSOR_2D` still refuse both.
  The wire format is unchanged. Take upstream's once the branch moves to b12658 or later.
- **DFlash drafters over GLM-5.3-Flash (`glm5-next`)**, `src/models/glm5-next.cpp`, not upstream. The same
  layer-input recording for `glm5-next`, which has no hook for it, so a DFlash draft over it aborts in
  `llm_graph_result::set_outputs` (layer input tensor is null).
- **A computation the device failed fails what reads its results**, not upstream. A backend may report a failure of
  queued work that it finds only when waiting for it (`get_status` in `ggml/src/ggml-backend-impl.h`, which bumps
  `GGML_BACKEND_API_VERSION` to 4 so a backend library built without it is refused, `ggml_backend_get_status`): Metal reports every command buffer that failed since the last synchronize, a graph's
  included after the next graph reuses its slot (`ggml/src/ggml-metal/ggml-metal-context.m`), and a tensor split
  reports its devices'. llama fails the context at the first synchronize that sees one, or at a graph a backend
  returns `GGML_STATUS_FAILED` for where it is queued, a memory update's (K-shift) included (`llama_get_compute_status`,
  `include/llama.h`): the output getters return NULL, the state functions 0, every later decode -3, and
  `llama_sampler_sample` and common's `common_sampler_sample` `LLAMA_TOKEN_NULL`, instead of the values the failed work
  left in the buffers. A read an event orders (the MTP draft's hidden states, a DFlash drafter's layer inputs) checks the
  source context too, and common's drafters stop at the first read of a failed context with an exception
  (`common/speculative.cpp`). common's prompt checkpoint (`common_prompt_checkpoint::update_tgt` and `update_dft`,
  `common/common.cpp`) returns false and keeps nothing on a failed context instead of aborting on the size mismatch.
  The callers in tools Spice does not ship (llama-server, llama-completion, the mtmd/TTS/realtime examples) do not
  handle the failure yet. A Metal event wait waits on the command buffer that signals it, so a failed one ends the
  wait instead of aborting after 60 s.
  `GGML_BACKEND_FAULT_TEST=N` (`ggml/src/ggml-backend.cpp`) skips the N-th graph of the process and fails its
  backend, for tests (with `GGML_BACKEND_FAULT_TEST_SYNC=1` the backend refuses that graph where it is queued and
  reports nothing later, as one without `get_status`): `tests/test-compute-fault.cpp` checks each read against a run
  without it.
- **A tensor split keeps the offset of a view that picks the same part of every element along the split axis**,
  `ggml/src/ggml-backend-meta.cpp`, not upstream. Each device's copy of a view of a split tensor starts at the view's
  offset scaled by the device's share of the split axis, unless the offset lies inside the data split; the check
  missed a view strided by the element that starts past part of each element, the rope part of each head of an MLA
  query (Mistral Large 4), so each device's rope read another part of its heads, and from an address no float starts
  at when its share does not divide the offset (NaN logits on three devices). The offset is kept only when the
  source's split is one segment that the view's elements fill; a source that splits something before them
  independently (a fused prefix in its own segment, a gated delta net's attention output before its states) still
  scales. `tests/test-argmax-split.cpp` checks both parts of each head, a gated delta net's new state and a
  same-stride crop of it, and a view of the heads past a segmented prefix, against one device.
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
merging the tag into the previous Spice branch (`spiceai-b12600` is `spiceai-v1.0.0` with `b12600` merged in, `spiceai-v1.0.0` is `spiceai-b11999` with `v1.0.0` merged in, and
`spiceai-b11999` is `spiceai-b11803` with `b11984`, then `b11999`, merged in). Its tip is tagged `<release>-spiceai.<n>`, a name apart from the
branch's so neither is ambiguous, with `<n>` counting re-packagings of one release. To move
to a new release:

```bash
git fetch https://github.com/ggml-org/llama.cpp refs/tags/<release>:refs/tags/<release>
git checkout -b spiceai-<release> spiceai-<previous release>
git merge <release>
# resolve the conflicts; where the release now has a patch this branch carries (an upstream PR it picked, or an
# equivalent), take the release's and keep only what it still lacks, then update the list above
# set the release in spiceai/release.rs, and the version and description in Cargo.toml
# fork PRs keep landing on the previous Spice branch while this one is made: merge it again just before tagging,
# then check that every PR merged into it is in this branch (a PR stacked on another one's branch reaches it through
# the PR that landed the stack, so it is not listed here)
git fetch origin spiceai-<previous release> && git merge origin/spiceai-<previous release>
gh pr list -R spiceai/llama.cpp --state merged --base spiceai-<previous release> --limit 500 --json number,mergeCommit \
  --jq '.[] | "\(.number) \(.mergeCommit.oid)"' | while read -r n c; do git merge-base --is-ancestor "$c" HEAD || echo "missing #$n"; done
git tag <release>-spiceai.1
git push origin spiceai-<release> refs/tags/<release>-spiceai.1
```

Then pin the tagged commit in spiceai/spiceai by its full SHA.
