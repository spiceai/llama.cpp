# Spice.ai packaging

This fork carries upstream llama.cpp releases unchanged, plus this directory and the root
`Cargo.toml`. Together they make a release fetchable by Cargo as the `llama-cpp-src` git
dependency, which Spice.ai's `llama-cpp` crate builds with CMake.

Each Spice branch is `spiceai-<release>`, cut from the upstream release tag with the
packaging commits on top. Its tip is tagged `<release>-spiceai.<n>`, a name apart from the
branch's so neither is ambiguous, with `<n>` counting re-packagings of one release. To move
to a new release:

```bash
git fetch https://github.com/ggml-org/llama.cpp refs/tags/<release>:refs/tags/<release>
git checkout -b spiceai-<release> <release>
# the previous branch's packaging commits, oldest first, without the merge commits PRs leave
git rev-list --reverse --topo-order --no-merges <previous release>..spiceai-<previous release> | git cherry-pick --stdin
# set the release in spiceai/release.rs, and the version and description in Cargo.toml
git tag <release>-spiceai.1
git push origin spiceai-<release> refs/tags/<release>-spiceai.1
```

Then pin the tagged commit in spiceai/spiceai by its full SHA.
