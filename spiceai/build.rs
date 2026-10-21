// Tells the build script of a crate that depends on this one where the llama.cpp sources
// are: it reads them from `DEP_LLAMA_CPP_SRC_ROOT` and `DEP_LLAMA_CPP_SRC_RELEASE`.
fn main() {
    let root = std::env::var("CARGO_MANIFEST_DIR").expect("Cargo sets CARGO_MANIFEST_DIR");
    println!("cargo:root={root}");
    println!("cargo:release={}", llama_cpp_src_release());
    println!("cargo:rerun-if-changed=spiceai/build.rs");
    println!("cargo:rerun-if-changed=spiceai/lib.rs");
}

include!("release.rs");
