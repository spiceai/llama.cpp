# Spice.ai packaging

This fork carries upstream llama.cpp releases, plus this directory and the root
`Cargo.toml`. Together they make a release fetchable by Cargo as the `llama-cpp-src` git
dependency, which Spice.ai's `llama-cpp` crate builds with CMake.

It also carries one source patch, which each Spice branch must keep:

- **A failed check while loading a model throws instead of aborting.** libllama defines
  `GGML_ABORT_THROWS` (`src/CMakeLists.txt`), so its `GGML_ASSERT`s go to `ggml_abort_throw`
  (`src/llama-impl.cpp`, declared in `ggml/include/ggml.h`). That throws a
  `std::runtime_error` only inside a `llama_abort_throws_scope` (`src/llama-impl.h`), which
  `llama_model_load` (`src/llama.cpp`) opens in the try block that turns an exception into a
  failed load, so a GGUF whose metadata fails a check fails its load rather than aborting the
  process. Everywhere else, and on every other thread, a failed check aborts as upstream: the
  C API functions that reach those checks do not catch exceptions. Spice's `llama-cpp` crate
  tests the load path with crafted and corrupted GGUFs.

Each Spice branch is `spiceai-<release>`, cut from the upstream release tag with the
Spice commits on top. Its tip is tagged `<release>-spiceai.<n>`, a name apart from the
branch's so neither is ambiguous, with `<n>` counting re-packagings of one release. To move
to a new release:

```bash
git fetch https://github.com/ggml-org/llama.cpp refs/tags/<release>:refs/tags/<release>
git checkout -b spiceai-<release> <release>
# the previous branch's Spice commits, oldest first, without the merge commits PRs leave
git rev-list --reverse --topo-order --no-merges <previous release>..spiceai-<previous release> | git cherry-pick --stdin
# set the release in spiceai/release.rs, and the version and description in Cargo.toml
git tag <release>-spiceai.1
git push origin spiceai-<release> refs/tags/<release>-spiceai.1
```

Then pin the tagged commit in spiceai/spiceai by its full SHA.
