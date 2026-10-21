# Spice.ai packaging

This fork carries upstream llama.cpp releases unchanged, plus this directory and the root
`Cargo.toml`. Together they make a release fetchable by Cargo as the `llama-cpp-src` git
dependency, which Spice.ai's `llama-cpp` crate builds with CMake.

Each Spice branch is `spiceai-<release>`, cut from the upstream release tag with this one
commit on top, and tagged `spiceai-<release>`. To move to a new release:

```bash
git fetch https://github.com/ggml-org/llama.cpp refs/tags/<release>:refs/tags/<release>
git checkout -b spiceai-<release> <release>
git cherry-pick <the packaging commit of the previous branch>
# set the release in spiceai/release.rs, and the version and description in Cargo.toml
git tag spiceai-<release>
git push origin spiceai-<release> refs/tags/spiceai-<release>
```

Then pin the tagged commit in spiceai/spiceai by its full SHA.
